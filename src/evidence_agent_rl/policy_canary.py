"""Hard behavioral canaries over replay-validated Harness evaluation episodes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .evaluation import EvaluationMatrixReport
from .harness import EvidenceHarness
from .models import Document, EpisodeState, GuidanceKind, StrategyMemory, TaskSpec
from .policies import OracleEvidenceMemoryPolicy
from .verifier import DeterministicVerifier


CANARY_REQUIREMENTS = frozenset(
    {
        "terminal",
        "strict_success",
        "answer",
        "abstain",
        "memory_accept",
        "memory_reject",
        "no_invalid_action",
        "no_firewall_violation",
        "no_budget_exhaustion",
    }
)


@dataclass(frozen=True)
class PolicyCanaryTask:
    task_id: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class PolicyCanarySpec:
    version: int
    experiment_id: str
    epistemic_status: str
    split: str
    minimum_review: str
    memory_mode: str
    budget_cap: int
    max_gpu_hours: float
    seed: int
    tasks: tuple[PolicyCanaryTask, ...]


@dataclass(frozen=True)
class PolicyCanaryTaskResult:
    task_id: str
    passed: bool
    checks: dict[str, bool]


@dataclass(frozen=True)
class PolicyCanaryGate:
    passed: bool
    passed_tasks: int
    total_tasks: int
    task_results: tuple[PolicyCanaryTaskResult, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "passed_tasks": self.passed_tasks,
            "total_tasks": self.total_tasks,
            "task_results": [asdict(item) for item in self.task_results],
        }


def load_policy_canary_spec(path: str | Path) -> PolicyCanarySpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version", "experiment_id", "epistemic_status", "split", "minimum_review",
        "memory_mode", "budget_cap", "max_gpu_hours", "seed", "tasks",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw.get("version") != 1:
        raise ValueError("policy canary spec fields differ from version 1 schema")
    if raw["epistemic_status"] != "draft_model_behavior_canary_not_evaluation":
        raise ValueError("policy canary must retain its draft epistemic status")
    if raw["split"] != "train" or raw["minimum_review"] != "draft":
        raise ValueError("policy canaries are train/draft diagnostics, not evaluation")
    if raw["memory_mode"] != "evidence_bound_reconstruction":
        raise ValueError("policy canary requires the full evidence-bound memory mode")
    if not isinstance(raw["budget_cap"], int) or not 1 <= raw["budget_cap"] <= 12:
        raise ValueError("policy canary budget_cap must be in [1, 12]")
    if not isinstance(raw["max_gpu_hours"], (int, float)) or not 0 < raw["max_gpu_hours"] <= 1:
        raise ValueError("policy canary can reserve at most one GPU hour")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool) or raw["seed"] < 0:
        raise ValueError("policy canary seed must be non-negative")
    if not isinstance(raw["experiment_id"], str) or not raw["experiment_id"].strip():
        raise ValueError("policy canary experiment_id is required")
    rows = raw["tasks"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("policy canary tasks must be non-empty")
    tasks: list[PolicyCanaryTask] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"task_id", "requirements"}:
            raise ValueError(f"policy canary task {index} has invalid fields")
        requirements = row["requirements"]
        if not isinstance(row["task_id"], str) or not row["task_id"]:
            raise ValueError(f"policy canary task {index} lacks task_id")
        if not isinstance(requirements, list) or not requirements or any(
            not isinstance(value, str) or value not in CANARY_REQUIREMENTS
            for value in requirements
        ):
            raise ValueError(f"policy canary task {index} has invalid requirements")
        if len(set(requirements)) != len(requirements):
            raise ValueError(f"policy canary task {index} repeats requirements")
        tasks.append(PolicyCanaryTask(row["task_id"], tuple(requirements)))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("policy canary task IDs must be unique")
    return PolicyCanarySpec(
        version=1,
        experiment_id=raw["experiment_id"],
        epistemic_status=raw["epistemic_status"],
        split="train",
        minimum_review="draft",
        memory_mode=raw["memory_mode"],
        budget_cap=raw["budget_cap"],
        max_gpu_hours=float(raw["max_gpu_hours"]),
        seed=raw["seed"],
        tasks=tuple(tasks),
    )


def evaluate_policy_canaries(
    report: EvaluationMatrixReport, spec: PolicyCanarySpec
) -> PolicyCanaryGate:
    if report.experiment_id != spec.experiment_id or report.split != spec.split:
        raise ValueError("policy canary report identity or split does not match spec")
    if report.epistemic_status != "draft_model_smoke" or len(report.conditions) != 1:
        raise ValueError("policy canary requires one draft-model evaluation condition")
    condition = report.conditions[0]
    if (
        condition.condition.memory_mode != spec.memory_mode
        or condition.condition.budget_cap != spec.budget_cap
    ):
        raise ValueError("policy canary report condition does not match spec")
    expected_ids = tuple(task.task_id for task in spec.tasks)
    if report.task_ids != expected_ids:
        raise ValueError("policy canary report task order or coverage does not match spec")
    by_id = {episode.task_id: episode for episode in condition.episodes}
    results: list[PolicyCanaryTaskResult] = []
    for expected in spec.tasks:
        episode = by_id[expected.task_id]
        results.append(_evaluate_task(expected, episode.state, episode.strict_success))
    passed = sum(result.passed for result in results)
    return PolicyCanaryGate(passed == len(results), passed, len(results), tuple(results))


def _evaluate_task(
    expected: PolicyCanaryTask, state: EpisodeState, strict_success: bool
) -> PolicyCanaryTaskResult:
    actual = {
        "terminal": state.done,
        "strict_success": strict_success,
        "answer": bool(state.final_claims) and state.abstention_reason is None,
        "abstain": state.abstention_reason is not None and not state.final_claims,
        "memory_accept": any(
            guidance.kind is not GuidanceKind.EMPTY for guidance in state.guidance_history
        ),
        "memory_reject": any(
            guidance.kind is GuidanceKind.EMPTY for guidance in state.guidance_history
        ),
        "no_invalid_action": not any(
            step.observation.kind == "invalid_action" for step in state.trace
        ),
        "no_firewall_violation": state.firewall_violations == 0,
        "no_budget_exhaustion": not any(
            step.observation.kind == "budget_exhausted" for step in state.trace
        ),
    }
    checks = {requirement: actual[requirement] for requirement in expected.requirements}
    return PolicyCanaryTaskResult(expected.task_id, all(checks.values()), checks)


def validate_policy_canary_reference(
    spec: PolicyCanarySpec,
    *,
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    tasks: Iterable[TaskSpec],
) -> PolicyCanaryGate:
    """Prove the canary is achievable with the gold-aware Harness oracle."""

    document_list = tuple(documents)
    memory_list = tuple(memories)
    task_map = {task.task_id: task for task in tasks}
    if missing := [item.task_id for item in spec.tasks if item.task_id not in task_map]:
        raise ValueError(f"policy canary reference tasks are missing: {missing}")
    policy = OracleEvidenceMemoryPolicy()
    verifier = DeterministicVerifier()
    results: list[PolicyCanaryTaskResult] = []
    for expected in spec.tasks:
        task = task_map[expected.task_id]
        harness = EvidenceHarness(
            document_list,
            memory_list,
            max_budget=spec.budget_cap,
            memory_mode=spec.memory_mode,
        )
        policy.run(harness, task, episode_id=f"canary-reference:{task.task_id}")
        score = verifier.score(harness.state)
        results.append(_evaluate_task(expected, harness.state, verifier.is_success(score)))
    passed = sum(result.passed for result in results)
    gate = PolicyCanaryGate(passed == len(results), passed, len(results), tuple(results))
    if not gate.passed:
        failures = [result.task_id for result in results if not result.passed]
        raise ValueError(f"policy canary requirements are not oracle-achievable: {failures}")
    return gate
