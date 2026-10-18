"""Paired held-out development comparison for base and SFT checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .evaluation import EvaluationMatrixReport
from .models import TaskSpec


def _delta(treatment: float | None, control: float | None) -> float | None:
    if treatment is None or control is None:
        return None
    return treatment - control


def _action_validity(episodes) -> tuple[float, float]:
    decisions = [decision for episode in episodes for decision in episode.decisions]
    valid_decisions = [
        decision
        for decision in decisions
        if decision.get("decode_error") is None
        and decision.get("decoded_action", {}).get("name") != "__invalid_format__"
    ]
    valid_episodes = sum(
        all(
            decision.get("decode_error") is None
            and decision.get("decoded_action", {}).get("name") != "__invalid_format__"
            for decision in episode.decisions
        )
        for episode in episodes
    )
    return (
        len(valid_decisions) / len(decisions) if decisions else 0.0,
        valid_episodes / len(episodes),
    )


def _compare_checkpoints(
    base: EvaluationMatrixReport,
    sft: EvaluationMatrixReport,
    tasks: Iterable[TaskSpec],
    *,
    expected_split: str,
    epistemic_status: str,
) -> dict[str, Any]:
    """Compare identical free-running tasks without treating memory pairs as IID."""

    if base.split != expected_split or sft.split != expected_split:
        raise ValueError(f"checkpoint diagnostic is restricted to the {expected_split} split")
    if (base.model_id, base.model_revision) != (sft.model_id, sft.model_revision):
        raise ValueError("base and SFT reports must share the same base model identity")
    if base.task_ids != sft.task_ids:
        raise ValueError("base and SFT reports must use identical task ordering")
    base_conditions = {item.condition.key: item for item in base.conditions}
    sft_conditions = {item.condition.key: item for item in sft.conditions}
    if set(base_conditions) != set(sft_conditions):
        raise ValueError("base and SFT reports must use identical conditions")
    task_map = {task.task_id: task for task in tasks}
    if set(base.task_ids) - set(task_map):
        raise ValueError("checkpoint diagnostic tasks are missing from the task catalog")

    cluster_by_task = {
        task_id: (task_map[task_id].mismatch_pair_id or task_id)
        for task_id in base.task_ids
    }
    conditions: list[dict[str, Any]] = []
    for key in sorted(base_conditions):
        base_result = base_conditions[key]
        sft_result = sft_conditions[key]
        base_episodes = {item.task_id: item for item in base_result.episodes}
        sft_episodes = {item.task_id: item for item in sft_result.episodes}
        base_action_validity, base_episode_validity = _action_validity(
            base_result.episodes
        )
        sft_action_validity, sft_episode_validity = _action_validity(sft_result.episodes)

        cluster_scores: dict[str, list[tuple[bool, bool]]] = {}
        for task_id in base.task_ids:
            cluster_scores.setdefault(cluster_by_task[task_id], []).append(
                (
                    base_episodes[task_id].strict_success,
                    sft_episodes[task_id].strict_success,
                )
            )
        wins = losses = ties = 0
        for values in cluster_scores.values():
            base_score = sum(item[0] for item in values) / len(values)
            sft_score = sum(item[1] for item in values) / len(values)
            if sft_score > base_score:
                wins += 1
            elif sft_score < base_score:
                losses += 1
            else:
                ties += 1

        base_summary = base_result.summary
        sft_summary = sft_result.summary
        conditions.append(
            {
                "condition_key": key,
                "question_clusters": len(cluster_scores),
                "cluster_wins": wins,
                "cluster_losses": losses,
                "cluster_ties": ties,
                "base": {
                    **asdict(base_summary),
                    "action_validity_rate": base_action_validity,
                    "episode_action_validity_rate": base_episode_validity,
                    "mean_total_tokens": base_result.mean_total_tokens,
                },
                "sft": {
                    **asdict(sft_summary),
                    "action_validity_rate": sft_action_validity,
                    "episode_action_validity_rate": sft_episode_validity,
                    "mean_total_tokens": sft_result.mean_total_tokens,
                },
                "delta_sft_minus_base": {
                    "strict_success_rate": (
                        sft_summary.strict_success_rate - base_summary.strict_success_rate
                    ),
                    "task_success": sft_summary.task_success - base_summary.task_success,
                    "evidence_coverage": (
                        sft_summary.evidence_coverage - base_summary.evidence_coverage
                    ),
                    "evidence_unsupported_answer_rate": _delta(
                        sft_summary.evidence_unsupported_answer_rate,
                        base_summary.evidence_unsupported_answer_rate,
                    ),
                    "invalid_memory_acceptance_rate": _delta(
                        sft_summary.invalid_memory_acceptance_rate,
                        base_summary.invalid_memory_acceptance_rate,
                    ),
                    "memory_mismatch_rejection_rate": _delta(
                        sft_summary.memory_mismatch_rejection_rate,
                        base_summary.memory_mismatch_rejection_rate,
                    ),
                    "mean_tool_calls": (
                        sft_summary.mean_tool_calls - base_summary.mean_tool_calls
                    ),
                    "action_validity_rate": sft_action_validity - base_action_validity,
                    "episode_action_validity_rate": (
                        sft_episode_validity - base_episode_validity
                    ),
                    "mean_total_tokens": (
                        sft_result.mean_total_tokens - base_result.mean_total_tokens
                    ),
                },
            }
        )

    return {
        "version": 1,
        "epistemic_status": epistemic_status,
        "model_id": base.model_id,
        "model_revision": base.model_revision,
        "split": expected_split,
        "base_experiment_id": base.experiment_id,
        "sft_experiment_id": sft.experiment_id,
        "task_ids": list(base.task_ids),
        "conditions": conditions,
    }


def compare_dev_checkpoints(
    base: EvaluationMatrixReport,
    sft: EvaluationMatrixReport,
    tasks: Iterable[TaskSpec],
) -> dict[str, Any]:
    """Compare paired held-out dev reports."""

    return _compare_checkpoints(
        base,
        sft,
        tasks,
        expected_split="dev",
        epistemic_status="ai_reviewed_dev_checkpoint_diagnostic_not_formal_evaluation",
    )


def compare_test_checkpoints(
    parent: EvaluationMatrixReport,
    candidate: EvaluationMatrixReport,
    tasks: Iterable[TaskSpec],
) -> dict[str, Any]:
    """Compare the once-opened source-paper-disjoint test reports."""

    return _compare_checkpoints(
        parent,
        candidate,
        tasks,
        expected_split="test",
        epistemic_status="heldout_test_checkpoint_comparison",
    )


def compare_train_canaries(
    base: EvaluationMatrixReport,
    sft: EvaluationMatrixReport,
    tasks: Iterable[TaskSpec],
) -> dict[str, Any]:
    """Compare paired train canaries without presenting them as held-out evaluation."""

    return _compare_checkpoints(
        base,
        sft,
        tasks,
        expected_split="train",
        epistemic_status="descriptive_train_checkpoint_canary_not_held_out_evaluation",
    )
