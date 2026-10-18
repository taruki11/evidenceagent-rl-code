"""Paired checkpoint evaluation across memory modes and tool budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Any, Callable, Iterable

from .harness import EvidenceHarness
from .metrics import EvaluationSummary, summarize_episodes
from .models import Action, Document, EpisodeState, MemoryMode, StrategyMemory, TaskSpec
from .react import DecisionProvider, ReActRunner
from .verifier import DeterministicVerifier, RewardBreakdown
from .policy_skill import PolicySkillContract, coerce_policy_skill
from .io import dump_trajectory


def select_evaluation_tasks(
    tasks: Iterable[TaskSpec],
    *,
    split: str,
    task_ids: Iterable[str] | None = None,
) -> tuple[TaskSpec, ...]:
    """Select an ordered canary subset without changing the underlying dataset."""

    split_tasks = tuple(task for task in tasks if task.split == split)
    if task_ids is None:
        return split_tasks
    requested = tuple(task_ids)
    if not requested or len(set(requested)) != len(requested) or any(
        not isinstance(task_id, str) or not task_id.strip() for task_id in requested
    ):
        raise ValueError("evaluation task_ids must be unique non-empty strings")
    by_id = {task.task_id: task for task in split_tasks}
    unknown = [task_id for task_id in requested if task_id not in by_id]
    if unknown:
        raise ValueError(f"evaluation task_ids are absent from split {split}: {unknown}")
    return tuple(by_id[task_id] for task_id in requested)


@dataclass(frozen=True)
class EvaluationCondition:
    memory_mode: str
    budget_cap: int

    def __post_init__(self) -> None:
        MemoryMode(self.memory_mode)
        if not isinstance(self.budget_cap, int) or isinstance(self.budget_cap, bool) or self.budget_cap <= 0:
            raise ValueError("evaluation budget_cap must be a positive integer")

    @property
    def key(self) -> str:
        return f"memory={self.memory_mode}|budget={self.budget_cap}"


@dataclass(frozen=True)
class EvaluationEpisode:
    task_id: str
    task_type: str
    mismatch_role: str | None
    state: EpisodeState
    score: RewardBreakdown
    strict_success: bool
    prompt_tokens: int
    completion_tokens: int
    decisions: tuple[dict[str, Any], ...]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "mismatch_role": self.mismatch_role,
            "strict_success": self.strict_success,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "score": {**asdict(self.score), "total": self.score.total},
            "decisions": list(self.decisions),
            "trajectory": dump_trajectory(self.state),
        }


@dataclass(frozen=True)
class EvaluationConditionResult:
    condition: EvaluationCondition
    summary: EvaluationSummary
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_total_tokens: float
    mean_spent_budget: float
    budget_exhaustion_rate: float
    episodes: tuple[EvaluationEpisode, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": asdict(self.condition),
            "condition_key": self.condition.key,
            "summary": asdict(self.summary),
            "mean_prompt_tokens": self.mean_prompt_tokens,
            "mean_completion_tokens": self.mean_completion_tokens,
            "mean_total_tokens": self.mean_total_tokens,
            "mean_spent_budget": self.mean_spent_budget,
            "budget_exhaustion_rate": self.budget_exhaustion_rate,
            "episodes": [episode.to_json() for episode in self.episodes],
        }


@dataclass(frozen=True)
class PairedConditionComparison:
    comparison_axis: str
    control_key: str
    treatment_key: str
    strict_success_delta: float
    task_success_delta: float
    evidence_unsupported_answer_rate_delta: float | None
    invalid_memory_acceptance_rate_delta: float | None
    mean_tool_calls_delta: float
    mean_total_tokens_delta: float
    treatment_wins: int
    control_wins: int
    ties: int
    discordant_exact_p_value: float | None


@dataclass(frozen=True)
class EvaluationMatrixReport:
    version: int
    epistemic_status: str
    experiment_id: str
    model_id: str
    model_revision: str
    split: str
    control_key: str
    task_ids: tuple[str, ...]
    conditions: tuple[EvaluationConditionResult, ...]
    comparisons: tuple[PairedConditionComparison, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "epistemic_status": self.epistemic_status,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "split": self.split,
            "control_key": self.control_key,
            "task_ids": list(self.task_ids),
            "conditions": [condition.to_json() for condition in self.conditions],
            "comparisons": [asdict(comparison) for comparison in self.comparisons],
        }


def _nullable_delta(treatment: float | None, control: float | None) -> float | None:
    if treatment is None or control is None:
        return None
    return treatment - control


def _two_sided_sign_test(wins: int, losses: int) -> float | None:
    discordant = wins + losses
    if discordant == 0:
        return None
    tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_comparisons(
    results: tuple[EvaluationConditionResult, ...],
    control_key: str,
) -> tuple[PairedConditionComparison, ...]:
    by_key = {result.condition.key: result for result in results}
    if control_key not in by_key:
        raise ValueError("control_key must identify one evaluation condition")
    global_control = by_key[control_key]

    def compare(
        control: EvaluationConditionResult,
        treatment: EvaluationConditionResult,
        axis: str,
    ) -> PairedConditionComparison:
        control_success = {item.task_id: item.strict_success for item in control.episodes}
        treatment_success = {item.task_id: item.strict_success for item in treatment.episodes}
        if set(treatment_success) != set(control_success):
            raise ValueError("paired evaluation task sets diverged")
        wins = sum(treatment_success[key] and not control_success[key] for key in control_success)
        losses = sum(control_success[key] and not treatment_success[key] for key in control_success)
        return PairedConditionComparison(
            comparison_axis=axis,
            control_key=control.condition.key,
            treatment_key=treatment.condition.key,
            strict_success_delta=(
                treatment.summary.strict_success_rate - control.summary.strict_success_rate
            ),
            task_success_delta=treatment.summary.task_success - control.summary.task_success,
            evidence_unsupported_answer_rate_delta=_nullable_delta(
                treatment.summary.evidence_unsupported_answer_rate,
                control.summary.evidence_unsupported_answer_rate,
            ),
            invalid_memory_acceptance_rate_delta=_nullable_delta(
                treatment.summary.invalid_memory_acceptance_rate,
                control.summary.invalid_memory_acceptance_rate,
            ),
            mean_tool_calls_delta=(
                treatment.summary.mean_tool_calls - control.summary.mean_tool_calls
            ),
            mean_total_tokens_delta=treatment.mean_total_tokens - control.mean_total_tokens,
            treatment_wins=wins,
            control_wins=losses,
            ties=len(control_success) - wins - losses,
            discordant_exact_p_value=_two_sided_sign_test(wins, losses),
        )

    comparisons: list[PairedConditionComparison] = []
    modes = tuple(dict.fromkeys(result.condition.memory_mode for result in results))
    budgets = tuple(dict.fromkeys(result.condition.budget_cap for result in results))
    control_mode = global_control.condition.memory_mode
    control_budget = global_control.condition.budget_cap
    for budget in budgets:
        baseline_key = EvaluationCondition(control_mode, budget).key
        if baseline_key not in by_key:
            continue
        for mode in modes:
            treatment_key = EvaluationCondition(mode, budget).key
            if mode != control_mode and treatment_key in by_key:
                comparisons.append(compare(by_key[baseline_key], by_key[treatment_key], "memory"))
    for mode in modes:
        baseline_key = EvaluationCondition(mode, control_budget).key
        if baseline_key not in by_key:
            continue
        for budget in budgets:
            treatment_key = EvaluationCondition(mode, budget).key
            if budget != control_budget and treatment_key in by_key:
                comparisons.append(compare(by_key[baseline_key], by_key[treatment_key], "budget"))
    return tuple(comparisons)


class EvaluationMatrixRunner:
    def __init__(
        self,
        documents: Iterable[Document],
        memories: Iterable[StrategyMemory],
        tasks: Iterable[TaskSpec],
        *,
        instructions: str | PolicySkillContract,
        model_id: str,
        model_revision: str,
        experiment_id: str,
        max_decisions: int = 12,
        policy_context_version: str = "v2",
    ) -> None:
        self.documents = tuple(documents)
        self.memories = tuple(memories)
        self.tasks = tuple(tasks)
        self.instructions = coerce_policy_skill(instructions)
        self.model_id = model_id.strip()
        self.model_revision = model_revision.strip()
        self.experiment_id = experiment_id.strip()
        self.max_decisions = max_decisions
        self.policy_context_version = policy_context_version
        if not self.tasks or not self.model_id or not self.model_revision or not self.experiment_id:
            raise ValueError("evaluation requires tasks and immutable identities")
        if self.model_revision.casefold() in {"main", "latest"} or any(
            marker in value
            for value in (self.model_id, self.model_revision)
            for marker in ("REPLACE_", "PLACEHOLDER")
        ):
            raise ValueError("evaluation model revision must be immutable")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("evaluation task IDs must be unique")
        if len({task.split for task in self.tasks}) != 1:
            raise ValueError("evaluation matrix requires exactly one split")
        if not isinstance(max_decisions, int) or isinstance(max_decisions, bool) or max_decisions <= 0:
            raise ValueError("max_decisions must be a positive integer")
        if self.policy_context_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            raise ValueError("policy_context_version must be v1, v2, v3, v4, v5, or v6")

    def run(
        self,
        conditions: Iterable[EvaluationCondition],
        provider_factory: Callable[[EvaluationCondition, TaskSpec, int], DecisionProvider],
        *,
        base_seed: int,
        epistemic_status: str,
        control_key: str,
    ) -> EvaluationMatrixReport:
        condition_list = tuple(conditions)
        if not condition_list or len({condition.key for condition in condition_list}) != len(condition_list):
            raise ValueError("evaluation conditions must be non-empty and unique")
        if control_key not in {condition.key for condition in condition_list}:
            raise ValueError("control_key must identify one evaluation condition")
        if epistemic_status not in {"draft_model_smoke", "human_reviewed_evaluation"}:
            raise ValueError("invalid evaluation epistemic status")
        verifier = DeterministicVerifier()
        results: list[EvaluationConditionResult] = []
        for condition in condition_list:
            episodes: list[EvaluationEpisode] = []
            states: list[EpisodeState] = []
            spent: list[int] = []
            exhausted: list[bool] = []
            for task_index, task in enumerate(self.tasks):
                # Common per-task seeds reduce sampling noise in paired conditions.
                seed = base_seed + task_index
                provider = provider_factory(condition, task, seed)
                harness = EvidenceHarness(
                    self.documents,
                    self.memories,
                    max_budget=condition.budget_cap,
                    memory_mode=condition.memory_mode,
                    policy_context_version=self.policy_context_version,
                )
                run = ReActRunner(provider, max_decisions=self.max_decisions).run(
                    harness,
                    task,
                    episode_id=(
                        f"eval:{self.experiment_id}:{condition.key}:{task.task_id}:seed-{seed}"
                    ),
                    instructions=self.instructions,
                )
                score = verifier.score(run.state, token_count=run.token_count)
                states.append(run.state)
                spent.append(condition.budget_cap - run.state.remaining_budget)
                exhausted.append(
                    any(step.observation.kind == "budget_exhausted" for step in run.state.trace)
                )
                episodes.append(
                    EvaluationEpisode(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        mismatch_role=task.mismatch_role,
                        state=run.state,
                        score=score,
                        strict_success=verifier.is_success(score),
                        prompt_tokens=sum(item.prompt_tokens for item in run.decisions),
                        completion_tokens=sum(item.completion_tokens for item in run.decisions),
                        decisions=tuple(
                            {
                                "raw_output": decision.raw_output,
                                "decoded_action": {
                                    "name": decision.action.name,
                                    "arguments": decision.action.arguments,
                                },
                                "decode_error": decision.decode_error,
                                "prompt_tokens": decision.prompt_tokens,
                                "completion_tokens": decision.completion_tokens,
                            }
                            for decision in run.decisions
                        ),
                    )
                )
            summary = summarize_episodes(states)
            count = len(episodes)
            results.append(
                EvaluationConditionResult(
                    condition=condition,
                    summary=summary,
                    mean_prompt_tokens=sum(item.prompt_tokens for item in episodes) / count,
                    mean_completion_tokens=sum(item.completion_tokens for item in episodes) / count,
                    mean_total_tokens=sum(item.total_tokens for item in episodes) / count,
                    mean_spent_budget=sum(spent) / count,
                    budget_exhaustion_rate=sum(exhausted) / count,
                    episodes=tuple(episodes),
                )
            )
        result_tuple = tuple(results)
        comparisons = _paired_comparisons(result_tuple, control_key)
        return EvaluationMatrixReport(
            version=1,
            epistemic_status=epistemic_status,
            experiment_id=self.experiment_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            split=self.tasks[0].split,
            control_key=control_key,
            task_ids=tuple(task.task_id for task in self.tasks),
            conditions=result_tuple,
            comparisons=comparisons,
        )


def validate_evaluation_report(
    raw: object,
    *,
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    tasks: Iterable[TaskSpec],
    policy_context_version: str = "v2",
) -> EvaluationMatrixReport:
    """Replay every recorded action and recompute all reported aggregates."""

    if not isinstance(raw, dict):
        raise ValueError("evaluation report must be an object")
    if policy_context_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
        raise ValueError("policy_context_version must be v1, v2, v3, v4, v5, or v6")
    required = {
        "version", "epistemic_status", "experiment_id", "model_id", "model_revision",
        "split", "control_key", "task_ids", "conditions", "comparisons",
    }
    if set(raw) != required or raw.get("version") != 1:
        raise ValueError("evaluation report fields differ from version 1 schema")
    if raw.get("epistemic_status") not in {"draft_model_smoke", "human_reviewed_evaluation"}:
        raise ValueError("evaluation report has invalid epistemic status")
    task_map = {task.task_id: task for task in tasks}
    task_ids = raw["task_ids"]
    if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("evaluation report task_ids must be unique and non-empty")
    if unknown := set(task_ids) - set(task_map):
        raise ValueError(f"evaluation report contains unknown tasks: {sorted(unknown)}")
    selected_tasks = tuple(task_map[task_id] for task_id in task_ids)
    if {task.split for task in selected_tasks} != {raw["split"]}:
        raise ValueError("evaluation report split does not match tasks")
    document_list = tuple(documents)
    memory_list = tuple(memories)
    raw_conditions = raw["conditions"]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("evaluation report conditions must be non-empty")
    verifier = DeterministicVerifier()
    rebuilt_conditions: list[EvaluationConditionResult] = []
    for raw_result in raw_conditions:
        if not isinstance(raw_result, dict):
            raise ValueError("evaluation condition result must be an object")
        condition_raw = raw_result.get("condition")
        if not isinstance(condition_raw, dict) or set(condition_raw) != {"memory_mode", "budget_cap"}:
            raise ValueError("evaluation condition is invalid")
        condition = EvaluationCondition(
            str(condition_raw["memory_mode"]), condition_raw["budget_cap"]
        )
        if raw_result.get("condition_key") != condition.key:
            raise ValueError("evaluation condition key drift")
        raw_episodes = raw_result.get("episodes")
        if not isinstance(raw_episodes, list) or [item.get("task_id") for item in raw_episodes if isinstance(item, dict)] != task_ids:
            raise ValueError("evaluation condition task ordering or coverage drift")
        episodes: list[EvaluationEpisode] = []
        states: list[EpisodeState] = []
        spent: list[int] = []
        exhausted: list[bool] = []
        for raw_episode, task in zip(raw_episodes, selected_tasks):
            if not isinstance(raw_episode, dict):
                raise ValueError("evaluation episode must be an object")
            trajectory = raw_episode.get("trajectory")
            decisions = raw_episode.get("decisions")
            if not isinstance(trajectory, dict) or not isinstance(decisions, list):
                raise ValueError("evaluation episode lacks trajectory or decisions")
            harness = EvidenceHarness(
                document_list,
                memory_list,
                max_budget=condition.budget_cap,
                memory_mode=condition.memory_mode,
                policy_context_version=policy_context_version,
            )
            harness.reset(task, episode_id=str(trajectory.get("episode_id", "")))
            steps = trajectory.get("steps")
            if not isinstance(steps, list):
                raise ValueError("evaluation trajectory steps must be an array")
            for step in steps:
                if not isinstance(step, dict) or not isinstance(step.get("action"), dict):
                    raise ValueError("evaluation trajectory action is invalid")
                action = step["action"]
                if set(action) != {"name", "arguments"}:
                    raise ValueError("evaluation trajectory action schema drift")
                harness.step(Action(action["name"], action["arguments"]))
            if dump_trajectory(harness.state) != trajectory:
                raise ValueError(f"evaluation trajectory replay diverged: {task.task_id}")
            decision_prompt_tokens = sum(int(item.get("prompt_tokens", -1)) for item in decisions)
            decision_completion_tokens = sum(
                int(item.get("completion_tokens", -1)) for item in decisions
            )
            if decision_prompt_tokens < 0 or decision_completion_tokens < 0:
                raise ValueError("evaluation decision token counts are invalid")
            score = verifier.score(
                harness.state,
                token_count=decision_prompt_tokens + decision_completion_tokens,
            )
            episode = EvaluationEpisode(
                task_id=task.task_id,
                task_type=task.task_type,
                mismatch_role=task.mismatch_role,
                state=harness.state,
                score=score,
                strict_success=verifier.is_success(score),
                prompt_tokens=decision_prompt_tokens,
                completion_tokens=decision_completion_tokens,
                decisions=tuple(decisions),
            )
            if episode.to_json() != raw_episode:
                raise ValueError(f"evaluation episode metadata or score drift: {task.task_id}")
            episodes.append(episode)
            states.append(harness.state)
            spent.append(condition.budget_cap - harness.state.remaining_budget)
            exhausted.append(
                any(step.observation.kind == "budget_exhausted" for step in harness.state.trace)
            )
        count = len(episodes)
        rebuilt = EvaluationConditionResult(
            condition=condition,
            summary=summarize_episodes(states),
            mean_prompt_tokens=sum(item.prompt_tokens for item in episodes) / count,
            mean_completion_tokens=sum(item.completion_tokens for item in episodes) / count,
            mean_total_tokens=sum(item.total_tokens for item in episodes) / count,
            mean_spent_budget=sum(spent) / count,
            budget_exhaustion_rate=sum(exhausted) / count,
            episodes=tuple(episodes),
        )
        if rebuilt.to_json() != raw_result:
            raise ValueError(f"evaluation aggregate drift: {condition.key}")
        rebuilt_conditions.append(rebuilt)
    condition_tuple = tuple(rebuilt_conditions)
    comparisons = _paired_comparisons(condition_tuple, str(raw["control_key"]))
    report = EvaluationMatrixReport(
        version=1,
        epistemic_status=str(raw["epistemic_status"]),
        experiment_id=str(raw["experiment_id"]),
        model_id=str(raw["model_id"]),
        model_revision=str(raw["model_revision"]),
        split=str(raw["split"]),
        control_key=str(raw["control_key"]),
        task_ids=tuple(task_ids),
        conditions=condition_tuple,
        comparisons=comparisons,
    )
    if report.to_json() != raw:
        raise ValueError("evaluation comparison or top-level metadata drift")
    return report
