"""Success-cost evaluation and feedback-driven tool-budget allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .harness import EvidenceHarness
from .metrics import EvaluationSummary, summarize_episodes
from .models import Document, MemoryMode, StrategyMemory, TaskSpec
from .verifier import DeterministicVerifier


class HarnessPolicy(Protocol):
    name: str

    def run(self, harness: EvidenceHarness, task: TaskSpec, *, episode_id: str) -> None: ...


@dataclass(frozen=True)
class BudgetPoint:
    budget_cap: int
    episodes: int
    strict_success: float
    task_success: float
    evidence_coverage: float
    citation_rate: float
    evidence_unsupported_answer_rate: float | None
    mean_tool_calls: float
    mean_spent_budget: float
    budget_exhaustion_rate: float
    mean_total_reward: float
    abstention_precision: float | None = None
    abstention_recall: float | None = None
    answer_coverage: float = 0.0
    selective_risk_at_observed_coverage: float | None = None
    conflict_resolution_rate: float | None = None


@dataclass(frozen=True)
class BudgetSweepResult:
    policy: str
    memory_mode: str
    points: tuple[BudgetPoint, ...]
    pareto_budget_caps: tuple[int, ...]


class BudgetSweepEvaluator:
    def __init__(
        self,
        documents: Iterable[Document],
        memories: Iterable[StrategyMemory],
        tasks: Iterable[TaskSpec],
        *,
        memory_mode: MemoryMode | str = MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION,
    ) -> None:
        self.documents = tuple(documents)
        self.memories = tuple(memories)
        self.tasks = tuple(tasks)
        self.memory_mode = MemoryMode(memory_mode)
        if not self.tasks:
            raise ValueError("budget sweep requires at least one task")

    def evaluate(
        self, policy: HarnessPolicy, *, budgets: Iterable[int] = (4, 8, 12)
    ) -> BudgetSweepResult:
        caps = tuple(int(budget) for budget in budgets)
        if not caps or any(cap <= 0 for cap in caps) or len(set(caps)) != len(caps):
            raise ValueError("budgets must be unique positive integers")
        verifier = DeterministicVerifier()
        points: list[BudgetPoint] = []
        for cap in sorted(caps):
            states = []
            scores = []
            strict = []
            spent = []
            exhausted = []
            for task in self.tasks:
                harness = EvidenceHarness(
                    self.documents,
                    self.memories,
                    max_budget=cap,
                    memory_mode=self.memory_mode,
                )
                policy.run(
                    harness,
                    task,
                    episode_id=f"budget:{policy.name}:{self.memory_mode.value}:{cap}:{task.task_id}",
                )
                score = verifier.score(harness.state)
                states.append(harness.state)
                scores.append(score)
                strict.append(verifier.is_success(score))
                spent.append(cap - harness.state.remaining_budget)
                exhausted.append(
                    any(step.observation.kind == "budget_exhausted" for step in harness.state.trace)
                )
            summary = summarize_episodes(states)
            points.append(
                BudgetPoint(
                    budget_cap=cap,
                    episodes=len(states),
                    strict_success=sum(strict) / len(strict),
                    task_success=summary.task_success,
                    evidence_coverage=summary.evidence_coverage,
                    citation_rate=summary.citation_rate,
                    evidence_unsupported_answer_rate=summary.evidence_unsupported_answer_rate,
                    mean_tool_calls=summary.mean_tool_calls,
                    mean_spent_budget=sum(spent) / len(spent),
                    budget_exhaustion_rate=sum(exhausted) / len(exhausted),
                    mean_total_reward=sum(score.total for score in scores) / len(scores),
                    abstention_precision=summary.abstention_precision,
                    abstention_recall=summary.abstention_recall,
                    answer_coverage=summary.answer_coverage,
                    selective_risk_at_observed_coverage=(
                        summary.selective_risk_at_observed_coverage
                    ),
                    conflict_resolution_rate=summary.conflict_resolution_rate,
                )
            )
        frontier = pareto_frontier(points)
        return BudgetSweepResult(
            policy.name,
            self.memory_mode.value,
            tuple(points),
            tuple(point.budget_cap for point in frontier),
        )


def pareto_frontier(points: Iterable[BudgetPoint]) -> tuple[BudgetPoint, ...]:
    point_list = tuple(points)
    frontier: list[BudgetPoint] = []
    for point in point_list:
        dominated = any(
            other.mean_spent_budget <= point.mean_spent_budget
            and other.strict_success >= point.strict_success
            and (
                other.mean_spent_budget < point.mean_spent_budget
                or other.strict_success > point.strict_success
                or (
                    other.mean_spent_budget == point.mean_spent_budget
                    and other.strict_success == point.strict_success
                    and other.budget_cap < point.budget_cap
                )
            )
            for other in point_list
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return tuple(sorted(frontier, key=lambda item: (item.mean_spent_budget, -item.strict_success)))


@dataclass(frozen=True)
class AdaptiveBudgetDecision:
    task_id: str
    budget_cap: int
    reason: str


class AdaptiveBudgetAllocator:
    """Promote caps only from observed exhaustion; demote from cheap success streaks."""

    def __init__(
        self,
        *,
        levels: tuple[int, ...] = (4, 8, 12),
        success_streak_for_demotion: int = 3,
    ) -> None:
        if not levels or tuple(sorted(set(levels))) != levels or any(level <= 0 for level in levels):
            raise ValueError("budget levels must be unique increasing positive integers")
        if success_streak_for_demotion <= 0:
            raise ValueError("success streak must be positive")
        self.levels = levels
        self.success_streak_for_demotion = success_streak_for_demotion
        self._level_by_task: dict[str, int] = {}
        self._cheap_success_streak: dict[str, int] = {}

    def allocation(self, task_id: str) -> AdaptiveBudgetDecision:
        if not task_id.strip():
            raise ValueError("task_id cannot be empty")
        index = self._level_by_task.get(task_id, 0)
        reason = "initial_or_retained"
        return AdaptiveBudgetDecision(task_id, self.levels[index], reason)

    def update(
        self,
        task_id: str,
        *,
        strict_success: bool,
        budget_exhausted: bool,
        spent_budget: int,
    ) -> AdaptiveBudgetDecision:
        current = self._level_by_task.get(task_id, 0)
        if budget_exhausted and not strict_success and current < len(self.levels) - 1:
            current += 1
            self._level_by_task[task_id] = current
            self._cheap_success_streak[task_id] = 0
            return AdaptiveBudgetDecision(task_id, self.levels[current], "promoted_after_exhaustion")
        lower_cap = self.levels[current - 1] if current > 0 else None
        if strict_success and lower_cap is not None and spent_budget <= lower_cap:
            streak = self._cheap_success_streak.get(task_id, 0) + 1
            self._cheap_success_streak[task_id] = streak
            if streak >= self.success_streak_for_demotion:
                current -= 1
                self._level_by_task[task_id] = current
                self._cheap_success_streak[task_id] = 0
                return AdaptiveBudgetDecision(task_id, self.levels[current], "demoted_after_cheap_success")
        else:
            self._cheap_success_streak[task_id] = 0
        self._level_by_task[task_id] = current
        return AdaptiveBudgetDecision(task_id, self.levels[current], "retained")


@dataclass(frozen=True)
class AdaptiveTaskResult:
    task_id: str
    budget_cap: int
    spent_budget: int
    strict_success: bool
    budget_exhausted: bool
    next_budget_cap: int
    allocation_reason: str


@dataclass(frozen=True)
class AdaptiveBudgetRound:
    round_index: int
    strict_success: float
    mean_budget_cap: float
    mean_spent_budget: float
    budget_exhaustion_rate: float
    tasks: tuple[AdaptiveTaskResult, ...]


class AdaptiveBudgetEvaluator:
    """Repeated policy evaluation whose caps change only from prior Harness outcomes."""

    def __init__(
        self,
        documents: Iterable[Document],
        memories: Iterable[StrategyMemory],
        tasks: Iterable[TaskSpec],
        *,
        allocator: AdaptiveBudgetAllocator | None = None,
        memory_mode: MemoryMode | str = MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION,
    ) -> None:
        self.documents = tuple(documents)
        self.memories = tuple(memories)
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("adaptive budget evaluation requires at least one task")
        self.allocator = allocator or AdaptiveBudgetAllocator()
        self.memory_mode = MemoryMode(memory_mode)
        self.verifier = DeterministicVerifier()

    def run_round(self, policy: HarnessPolicy, *, round_index: int) -> AdaptiveBudgetRound:
        if round_index < 0:
            raise ValueError("round_index cannot be negative")
        results: list[AdaptiveTaskResult] = []
        for task in self.tasks:
            allocation = self.allocator.allocation(task.task_id)
            harness = EvidenceHarness(
                self.documents,
                self.memories,
                max_budget=allocation.budget_cap,
                memory_mode=self.memory_mode,
            )
            policy.run(
                harness,
                task,
                episode_id=(
                    f"adaptive:{round_index}:{policy.name}:"
                    f"{self.memory_mode.value}:{task.task_id}"
                ),
            )
            score = self.verifier.score(harness.state)
            success = self.verifier.is_success(score)
            spent = allocation.budget_cap - harness.state.remaining_budget
            exhausted = any(
                step.observation.kind == "budget_exhausted"
                for step in harness.state.trace
            )
            next_allocation = self.allocator.update(
                task.task_id,
                strict_success=success,
                budget_exhausted=exhausted,
                spent_budget=spent,
            )
            results.append(
                AdaptiveTaskResult(
                    task.task_id,
                    allocation.budget_cap,
                    spent,
                    success,
                    exhausted,
                    next_allocation.budget_cap,
                    next_allocation.reason,
                )
            )
        count = len(results)
        return AdaptiveBudgetRound(
            round_index,
            sum(result.strict_success for result in results) / count,
            sum(result.budget_cap for result in results) / count,
            sum(result.spent_budget for result in results) / count,
            sum(result.budget_exhausted for result in results) / count,
            tuple(results),
        )
