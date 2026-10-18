"""CPU-only oracle feasibility gate for evaluation canaries."""

from __future__ import annotations

from typing import Any, Iterable

from .evaluation import EvaluationCondition, select_evaluation_tasks
from .harness import EvidenceHarness
from .models import Document, MemoryMode, StrategyMemory, TaskSpec
from .policies import OracleEvidenceMemoryPolicy, OracleEvidencePolicy
from .verifier import DeterministicVerifier


def audit_evaluation_achievability(
    config: dict[str, Any],
    *,
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    tasks: Iterable[TaskSpec],
) -> dict[str, Any]:
    """Prove every configured task/condition can finish within the decision cap."""

    if config.get("stage") != "evaluation":
        raise ValueError("evaluation achievability requires stage=evaluation")
    settings = config["training"]
    data = config["data"]
    max_decisions = settings["max_decisions"]
    if not isinstance(max_decisions, int) or isinstance(max_decisions, bool) or max_decisions <= 0:
        raise ValueError("training.max_decisions must be a positive integer")

    selected_tasks = select_evaluation_tasks(
        tasks,
        split=data["split"],
        task_ids=settings.get("task_ids"),
    )
    document_list = tuple(documents)
    memory_list = tuple(memories)
    verifier = DeterministicVerifier()
    memory_oracle = OracleEvidenceMemoryPolicy()
    evidence_oracle = OracleEvidencePolicy()
    cases: list[dict[str, Any]] = []

    for mode in settings["memory_modes"]:
        for budget in settings["budgets"]:
            condition = EvaluationCondition(mode, budget)
            for task in selected_tasks:
                harness = EvidenceHarness(
                    document_list,
                    memory_list,
                    max_budget=condition.budget_cap,
                    memory_mode=condition.memory_mode,
                )
                use_memory = (
                    condition.memory_mode is MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION
                    or (
                        condition.memory_mode is not MemoryMode.NONE
                        and task.mismatch_role != "mismatched"
                    )
                )
                oracle = memory_oracle if use_memory else evidence_oracle
                oracle.run(
                    harness,
                    task,
                    episode_id=f"achievability:{condition.key}:{task.task_id}",
                )
                strict_success = verifier.is_success(verifier.score(harness.state))
                decisions_required = len(harness.state.trace)
                feasible = strict_success and decisions_required <= max_decisions
                cases.append(
                    {
                        "condition_key": condition.key,
                        "task_id": task.task_id,
                        "decisions_required": decisions_required,
                        "max_decisions": max_decisions,
                        "strict_success": strict_success,
                        "feasible": feasible,
                    }
                )

    failures = [case for case in cases if not case["feasible"]]
    return {
        "version": 1,
        "epistemic_status": "oracle_environment_achievability_not_model_performance",
        "experiment_id": config["experiment_id"],
        "split": data["split"],
        "tasks": len(selected_tasks),
        "conditions": len(settings["memory_modes"]) * len(settings["budgets"]),
        "max_decisions": max_decisions,
        "ready": not failures,
        "failures": failures,
        "cases": cases,
    }
