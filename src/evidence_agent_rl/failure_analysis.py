"""Failure taxonomy for free-running evaluation trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .evaluation import EvaluationConditionResult, EvaluationEpisode, EvaluationMatrixReport
from .models import GuidanceKind, MemoryMode


def _failure_tags(
    episode: EvaluationEpisode,
    condition: EvaluationConditionResult,
) -> tuple[str, ...]:
    if episode.strict_success:
        return ()

    state = episode.state
    score = episode.score
    tags: set[str] = set()
    if any(decision.get("decode_error") is not None for decision in episode.decisions):
        tags.add("invalid_json_action")
    if any(step.observation.kind == "provider_error" for step in state.trace):
        tags.add("model_service_error")
    if any(step.observation.kind == "budget_exhausted" for step in state.trace):
        tags.add("budget_exhausted")
    if any(step.observation.kind == "decision_limit_exhausted" for step in state.trace):
        tags.add("decision_limit_exhausted")
    if any(
        not step.observation.ok
        and step.observation.kind not in {"budget_exhausted", "decision_limit_exhausted"}
        for step in state.trace
    ):
        tags.add("invalid_tool_action")

    memory_eligible = (
        condition.condition.memory_mode != MemoryMode.NONE.value
        and bool(state.task.available_memory_ids)
    )
    if memory_eligible and not state.inspected_memory_ids:
        tags.add("memory_not_inspected")
    if (
        memory_eligible
        and condition.condition.memory_mode
        == MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION.value
        and not state.guidance_history
    ):
        tags.add("memory_not_reconstructed")
    if state.task.mismatch_role == "mismatched" and any(
        guidance.kind is not GuidanceKind.EMPTY for guidance in state.guidance_history
    ):
        tags.add("invalid_memory_acceptance")

    if state.abstention_reason is not None and not state.task.insufficient:
        tags.add("incorrect_abstention")
    if state.task.insufficient and state.abstention_reason is None:
        tags.add("missing_abstention")
    if score.unsupported > 0:
        tags.add("unsupported_answer")
    if not state.task.insufficient and score.evidence < 1.0:
        tags.add("incomplete_evidence")
    if state.final_claims and score.citation < 1.0:
        tags.add("incomplete_citation")
    if not state.done:
        tags.add("no_terminal_action")
    if score.task < 1.0 and not tags:
        tags.add("task_failure_unclassified")
    return tuple(sorted(tags))


def analyze_rollout_failures(report: EvaluationMatrixReport) -> dict[str, Any]:
    """Summarize overlapping failure causes without turning dev/test into training data."""

    condition_rows: list[dict[str, Any]] = []
    total_episodes = 0
    total_failed = 0
    for condition in report.conditions:
        counts: Counter[str] = Counter()
        failed = 0
        all_memory_eligible = 0
        all_memory_inspected = 0
        all_memory_decided = 0
        matched_eligible = 0
        matched_decided = 0
        matched_accepted = 0
        matched_rejected = 0
        mismatched_eligible = 0
        mismatched_decided = 0
        mismatched_rejected = 0
        mismatched_invalid_acceptance = 0
        memory_eligible = 0
        memory_inspected = 0
        memory_reconstructed = 0
        for episode in condition.episodes:
            total_episodes += 1
            state = episode.state
            is_memory_eligible = (
                condition.condition.memory_mode != MemoryMode.NONE.value
                and bool(state.task.available_memory_ids)
            )
            if is_memory_eligible:
                all_memory_eligible += 1
                all_memory_inspected += bool(state.inspected_memory_ids)
                all_memory_decided += bool(state.guidance_history)
                if state.task.mismatch_role == "matched":
                    matched_eligible += 1
                    matched_decided += bool(state.guidance_history)
                    matched_accepted += any(
                        guidance.kind is not GuidanceKind.EMPTY
                        for guidance in state.guidance_history
                    )
                    matched_rejected += any(
                        guidance.kind is GuidanceKind.EMPTY
                        for guidance in state.guidance_history
                    )
                elif state.task.mismatch_role == "mismatched":
                    mismatched_eligible += 1
                    mismatched_decided += bool(state.guidance_history)
                    mismatched_rejected += any(
                        guidance.kind is GuidanceKind.EMPTY
                        for guidance in state.guidance_history
                    )
                    mismatched_invalid_acceptance += any(
                        guidance.kind is not GuidanceKind.EMPTY
                        for guidance in state.guidance_history
                    )
            if episode.strict_success:
                continue
            failed += 1
            total_failed += 1
            counts.update(_failure_tags(episode, condition))
            if (
                condition.condition.memory_mode != MemoryMode.NONE.value
                and episode.state.task.available_memory_ids
            ):
                memory_eligible += 1
                memory_inspected += bool(episode.state.inspected_memory_ids)
                memory_reconstructed += bool(episode.state.guidance_history)
        condition_rows.append(
            {
                "condition_key": condition.condition.key,
                "episodes": len(condition.episodes),
                "failed_episodes": failed,
                "failure_tag_counts": dict(sorted(counts.items())),
                "memory_eligible_episodes": all_memory_eligible,
                "memory_inspection_coverage_rate": (
                    all_memory_inspected / all_memory_eligible
                    if all_memory_eligible
                    else None
                ),
                "memory_decision_coverage_rate": (
                    all_memory_decided / all_memory_eligible
                    if all_memory_eligible
                    else None
                ),
                "matched_memory_eligible_episodes": matched_eligible,
                "matched_memory_decision_coverage_rate": (
                    matched_decided / matched_eligible
                    if matched_eligible
                    else None
                ),
                "matched_memory_acceptance_rate_over_eligible": (
                    matched_accepted / matched_eligible
                    if matched_eligible
                    else None
                ),
                "matched_memory_rejection_rate_over_eligible": (
                    matched_rejected / matched_eligible
                    if matched_eligible
                    else None
                ),
                "mismatched_memory_eligible_episodes": mismatched_eligible,
                "mismatched_memory_decision_coverage_rate": (
                    mismatched_decided / mismatched_eligible
                    if mismatched_eligible
                    else None
                ),
                "mismatch_rejection_rate_over_eligible": (
                    mismatched_rejected / mismatched_eligible
                    if mismatched_eligible
                    else None
                ),
                "invalid_memory_acceptance_rate_over_eligible": (
                    mismatched_invalid_acceptance / mismatched_eligible
                    if mismatched_eligible
                    else None
                ),
                "memory_eligible_failed_episodes": memory_eligible,
                "memory_inspected_failed_episodes": memory_inspected,
                "memory_reconstructed_failed_episodes": memory_reconstructed,
            }
        )
    return {
        "version": 1,
        "epistemic_status": "descriptive_failure_audit_not_model_performance_claim",
        "source_experiment_id": report.experiment_id,
        "source_split": report.split,
        "episodes": total_episodes,
        "failed_episodes": total_failed,
        "conditions": condition_rows,
    }
