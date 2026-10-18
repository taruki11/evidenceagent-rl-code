"""Framework-neutral rollout scoring, group advantages, and task curriculum."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable

from .harness import EvidenceHarness
from .models import Action, Document, EpisodeState, MemoryMode, StrategyMemory, TaskSpec
from .react import DecisionProvider, PolicyDecision, ReActRunner
from .verifier import DeterministicVerifier, RewardBreakdown
from .policy_skill import PolicySkillContract, coerce_policy_skill


class RewardMode(StrEnum):
    OUTCOME_ONLY = "outcome_only"
    COMPOSITE = "composite"
    MEMORY_BALANCED_COMPOSITE = "memory_balanced_composite"
    SUCCESS_DOMINANT_MEMORY_BALANCED_COMPOSITE = (
        "success_dominant_memory_balanced_composite"
    )


class AdvantageMethod(StrEnum):
    GRPO = "grpo"
    RLOO = "rloo"


class PoolName(StrEnum):
    ACTIVE = "active"
    MONITORING = "monitoring"


def memory_decision_reward(state: EpisodeState) -> float:
    """Symmetric reward for accepting matched and rejecting mismatched memory."""

    if state.task.mismatch_role not in {"matched", "mismatched"}:
        return 0.0
    if not state.guidance_history:
        return -0.5
    accepted = any(guidance.kind.value != "<EMPTY>" for guidance in state.guidance_history)
    rejected = any(guidance.kind.value == "<EMPTY>" for guidance in state.guidance_history)
    if accepted == rejected:
        return -1.0
    correct = accepted if state.task.mismatch_role == "matched" else rejected
    return 1.0 if correct else -1.0


@dataclass(frozen=True)
class RolloutCandidate:
    rollout_id: str
    task_id: str
    actions: tuple[Action, ...]
    token_count: int = 0


@dataclass(frozen=True)
class ScoredRollout:
    rollout_id: str
    task_id: str
    state: EpisodeState
    score: RewardBreakdown
    reward: float
    success: bool


@dataclass(frozen=True)
class AdvantageRollout:
    rollout: ScoredRollout
    advantage: float


@dataclass(frozen=True)
class CollectedRollout:
    candidate: RolloutCandidate
    decisions: tuple[PolicyDecision, ...]


@dataclass(frozen=True)
class CollectedRolloutGroup:
    collected: tuple[CollectedRollout, ...]
    scored: tuple[AdvantageRollout, ...]


@dataclass(frozen=True)
class TaskAllocation:
    task_id: str
    pool: PoolName
    success_rate: float | None
    rollout_budget: int


class HarnessRolloutEvaluator:
    """Replay policy actions in the authoritative environment before scoring."""

    def __init__(
        self,
        documents: Iterable[Document],
        memories: Iterable[StrategyMemory],
        tasks: Iterable[TaskSpec],
        *,
        reward_mode: RewardMode | str = RewardMode.COMPOSITE,
        max_budget: int = 8,
        memory_mode: MemoryMode | str = MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION,
        policy_context_version: str = "v2",
    ) -> None:
        if policy_context_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            raise ValueError("policy_context_version must be v1, v2, v3, v4, v5, or v6")
        self.documents = tuple(documents)
        self.memories = tuple(memories)
        self.tasks = {task.task_id: task for task in tasks}
        self.reward_mode = RewardMode(reward_mode)
        self.max_budget = max_budget
        self.memory_mode = MemoryMode(memory_mode)
        self.policy_context_version = policy_context_version
        self.verifier = DeterministicVerifier()

    def evaluate(self, candidate: RolloutCandidate) -> ScoredRollout:
        task = self.tasks.get(candidate.task_id)
        if task is None:
            raise ValueError(f"unknown rollout task: {candidate.task_id}")
        if not candidate.rollout_id.strip():
            raise ValueError("rollout_id cannot be empty")
        harness = EvidenceHarness(
            self.documents,
            self.memories,
            max_budget=self.max_budget,
            memory_mode=self.memory_mode,
            policy_context_version=self.policy_context_version,
        )
        state = harness.replay(
            task,
            candidate.actions,
            episode_id=candidate.rollout_id,
        )
        score = self.verifier.score(state, token_count=candidate.token_count)
        success = self.verifier.is_success(score)
        if self.reward_mode is RewardMode.OUTCOME_ONLY:
            reward = float(success)
        elif self.reward_mode in {
            RewardMode.MEMORY_BALANCED_COMPOSITE,
            RewardMode.SUCCESS_DOMINANT_MEMORY_BALANCED_COMPOSITE,
        }:
            reward = score.total + memory_decision_reward(state)
            if self.reward_mode is RewardMode.SUCCESS_DOMINANT_MEMORY_BALANCED_COMPOSITE:
                reward += 10.0 * success
        else:
            reward = score.total
        return ScoredRollout(
            candidate.rollout_id, candidate.task_id, state, score, reward, success
        )

    def evaluate_group(
        self,
        candidates: Iterable[RolloutCandidate],
        *,
        method: AdvantageMethod | str = AdvantageMethod.GRPO,
    ) -> tuple[AdvantageRollout, ...]:
        candidate_list = tuple(candidates)
        if len(candidate_list) < 2:
            raise ValueError("a rollout group requires at least two candidates")
        if len({candidate.rollout_id for candidate in candidate_list}) != len(candidate_list):
            raise ValueError("rollout_id values must be unique within a group")
        task_ids = {candidate.task_id for candidate in candidate_list}
        if len(task_ids) != 1:
            raise ValueError("all candidates in a group must share one task_id")
        scored = tuple(self.evaluate(candidate) for candidate in candidate_list)
        advantages = compute_group_advantages(
            [rollout.reward for rollout in scored], method=method
        )
        return tuple(
            AdvantageRollout(rollout, advantage)
            for rollout, advantage in zip(scored, advantages)
        )


class HarnessRolloutCollector:
    """Collect policy actions and prove they replay through the common Harness."""

    def __init__(
        self,
        documents: Iterable[Document],
        memories: Iterable[StrategyMemory],
        tasks: Iterable[TaskSpec],
        *,
        instructions: str | PolicySkillContract,
        reward_mode: RewardMode | str = RewardMode.COMPOSITE,
        max_budget: int = 8,
        max_decisions: int = 12,
        memory_mode: MemoryMode | str = MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION,
        policy_context_version: str = "v2",
    ) -> None:
        if policy_context_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            raise ValueError("policy_context_version must be v1, v2, v3, v4, v5, or v6")
        policy_skill = coerce_policy_skill(instructions)
        self.documents = tuple(documents)
        self.memories = tuple(memories)
        self.tasks = {task.task_id: task for task in tasks}
        self.instructions = policy_skill
        self.max_budget = max_budget
        self.max_decisions = max_decisions
        self.memory_mode = MemoryMode(memory_mode)
        self.policy_context_version = policy_context_version
        self.evaluator = HarnessRolloutEvaluator(
            self.documents,
            self.memories,
            self.tasks.values(),
            reward_mode=reward_mode,
            max_budget=max_budget,
            memory_mode=self.memory_mode,
            policy_context_version=self.policy_context_version,
        )

    def collect(
        self,
        provider: DecisionProvider,
        *,
        task_id: str,
        rollout_id: str,
    ) -> CollectedRollout:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown rollout task: {task_id}")
        harness = EvidenceHarness(
            self.documents,
            self.memories,
            max_budget=self.max_budget,
            memory_mode=self.memory_mode,
            policy_context_version=self.policy_context_version,
        )
        result = ReActRunner(provider, max_decisions=self.max_decisions).run(
            harness,
            task,
            episode_id=rollout_id,
            instructions=self.instructions,
        )
        candidate = RolloutCandidate(
            rollout_id=rollout_id,
            task_id=task_id,
            actions=harness.cloned_action_trace(),
            token_count=result.token_count,
        )
        replayed = self.evaluator.evaluate(candidate)
        if replayed.state != result.state:
            raise AssertionError(f"collected rollout replay diverged: {rollout_id}")
        return CollectedRollout(candidate, result.decisions)

    def collect_group(
        self,
        providers: Iterable[tuple[str, DecisionProvider]],
        *,
        task_id: str,
        method: AdvantageMethod | str = AdvantageMethod.GRPO,
        max_workers: int = 1,
    ) -> CollectedRolloutGroup:
        provider_list = tuple(providers)
        if len(provider_list) < 2:
            raise ValueError("a rollout group requires at least two providers")
        rollout_ids = [rollout_id for rollout_id, _ in provider_list]
        if len(set(rollout_ids)) != len(rollout_ids):
            raise ValueError("rollout_id values must be unique within a group")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        def collect_one(item: tuple[str, DecisionProvider]) -> CollectedRollout:
            rollout_id, provider = item
            return self.collect(provider, task_id=task_id, rollout_id=rollout_id)

        if max_workers == 1:
            collected = tuple(map(collect_one, provider_list))
        else:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(provider_list))
            ) as executor:
                collected = tuple(executor.map(collect_one, provider_list))
        scored = self.evaluator.evaluate_group(
            [item.candidate for item in collected], method=method
        )
        return CollectedRolloutGroup(collected, scored)

    def collect_group_with_retries(
        self,
        provider_factory: Callable[[int, int], tuple[str, DecisionProvider]],
        *,
        task_id: str,
        group_size: int,
        max_attempts: int,
        method: AdvantageMethod | str = AdvantageMethod.GRPO,
        max_workers: int = 1,
    ) -> CollectedRolloutGroup:
        """Replace malformed sampled-token responses without weakening validation."""

        if group_size < 2 or max_attempts <= 0 or max_workers <= 0:
            raise ValueError("invalid retry group size, attempts, or worker count")

        def collect_slot(slot: int) -> CollectedRollout:
            errors: list[str] = []
            for attempt in range(max_attempts):
                rollout_id, provider = provider_factory(slot, attempt)
                try:
                    return self.collect(
                        provider,
                        task_id=task_id,
                        rollout_id=rollout_id,
                    )
                except ValueError as exc:
                    errors.append(str(exc))
            raise RuntimeError(
                f"rollout slot {slot} failed after {max_attempts} attempts: "
                + " | ".join(errors)
            )

        if max_workers == 1:
            collected = tuple(collect_slot(slot) for slot in range(group_size))
        else:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, group_size)
            ) as executor:
                collected = tuple(executor.map(collect_slot, range(group_size)))
        rollout_ids = [item.candidate.rollout_id for item in collected]
        if len(set(rollout_ids)) != len(rollout_ids):
            raise ValueError("provider factory produced duplicate rollout IDs")
        scored = self.evaluator.evaluate_group(
            [item.candidate for item in collected], method=method
        )
        return CollectedRolloutGroup(collected, scored)


def compute_group_advantages(
    rewards: Iterable[float],
    *,
    method: AdvantageMethod | str = AdvantageMethod.GRPO,
    epsilon: float = 1e-6,
) -> tuple[float, ...]:
    values = tuple(float(reward) for reward in rewards)
    if len(values) < 2:
        raise ValueError("group advantages require at least two rewards")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rewards must be finite")
    selected = AdvantageMethod(method)
    if selected is AdvantageMethod.RLOO:
        total = sum(values)
        return tuple(
            value - (total - value) / (len(values) - 1) for value in values
        )
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0.0:
        return tuple(0.0 for _ in values)
    denominator = math.sqrt(variance) + epsilon
    return tuple((value - mean) / denominator for value in values)


class DynamicTaskPool:
    """Qwen-UI-Agent-style active/monitoring allocation from recent success."""

    def __init__(
        self,
        *,
        active_lower: float = 0.2,
        active_upper: float = 0.8,
        window_size: int = 32,
        active_rollouts: int = 4,
        monitoring_rollouts: int = 1,
    ) -> None:
        if not 0.0 <= active_lower < active_upper <= 1.0:
            raise ValueError("active thresholds must satisfy 0 <= lower < upper <= 1")
        if window_size <= 0 or active_rollouts < 2 or monitoring_rollouts <= 0:
            raise ValueError("invalid task-pool budgets")
        self.active_lower = active_lower
        self.active_upper = active_upper
        self.window_size = window_size
        self.active_rollouts = active_rollouts
        self.monitoring_rollouts = monitoring_rollouts
        self._history: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, task_id: str, outcomes: Iterable[bool]) -> TaskAllocation:
        if not task_id.strip():
            raise ValueError("task_id cannot be empty")
        history = self._history[task_id]
        history.extend(bool(outcome) for outcome in outcomes)
        return self.allocation(task_id)

    def allocation(self, task_id: str) -> TaskAllocation:
        history = self._history.get(task_id)
        if not history:
            return TaskAllocation(
                task_id, PoolName.MONITORING, None, self.monitoring_rollouts
            )
        success_rate = sum(history) / len(history)
        active = self.active_lower <= success_rate <= self.active_upper
        return TaskAllocation(
            task_id,
            PoolName.ACTIVE if active else PoolName.MONITORING,
            success_rate,
            self.active_rollouts if active else self.monitoring_rollouts,
        )

    def snapshot(self) -> tuple[TaskAllocation, ...]:
        return tuple(self.allocation(task_id) for task_id in sorted(self._history))
