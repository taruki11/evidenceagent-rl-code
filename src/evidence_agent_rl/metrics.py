"""Aggregate evaluation metrics with explicit eligibility denominators."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .models import EpisodeState, GuidanceKind
from .verifier import DeterministicVerifier


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    strict_success_rate: float
    task_success: float
    evidence_coverage: float
    citation_rate: float
    evidence_unsupported_answer_rate: float | None
    invalid_memory_acceptance_rate: float | None
    memory_mismatch_rejection_rate: float | None
    mean_tool_calls: float
    abstention_precision: float | None
    abstention_recall: float | None
    answer_coverage: float
    selective_risk_at_observed_coverage: float | None
    conflict_resolution_rate: float | None


@dataclass(frozen=True)
class ConfidenceScoredEpisode:
    """An episode paired with an explicit policy confidence, never a reward proxy."""

    state: EpisodeState
    confidence: float


@dataclass(frozen=True)
class RiskCoveragePoint:
    confidence_threshold: float
    coverage: float
    risk: float
    retained_answers: int
    total_episodes: int


def summarize_episodes(states: Iterable[EpisodeState]) -> EvaluationSummary:
    episodes = tuple(states)
    if not episodes:
        raise ValueError("at least one episode is required")
    verifier = DeterministicVerifier()
    scores = [verifier.score(state) for state in episodes]
    strict_success = [verifier.is_success(score) for score in scores]
    answered_claims = sum(len(state.final_claims) for state in episodes)
    answered_indices = [index for index, state in enumerate(episodes) if state.final_claims]
    abstained_indices = [
        index for index, state in enumerate(episodes) if state.abstention_reason is not None
    ]
    insufficient_indices = [index for index, state in enumerate(episodes) if state.task.insufficient]
    conflict_indices = [
        index for index, state in enumerate(episodes) if state.task.task_type == "conflict"
    ]
    unsupported_claims = sum(score.unsupported for score in scores)

    mismatch_decisions = [
        state
        for state in episodes
        if state.task.mismatch_role == "mismatched"
        and state.inspected_memory_ids
        and state.guidance_history
    ]
    invalid_acceptances = sum(
        any(guidance.kind is not GuidanceKind.EMPTY for guidance in state.guidance_history)
        for state in mismatch_decisions
    )
    rejections = sum(
        any(guidance.kind is GuidanceKind.EMPTY for guidance in state.guidance_history)
        for state in mismatch_decisions
    )
    tool_calls = [
        sum(step.action.name not in {"answer", "abstain"} for step in state.trace)
        for state in episodes
    ]
    return EvaluationSummary(
        episodes=len(episodes),
        strict_success_rate=sum(strict_success) / len(strict_success),
        task_success=sum(score.task for score in scores) / len(scores),
        evidence_coverage=sum(score.evidence for score in scores) / len(scores),
        citation_rate=sum(score.citation for score in scores) / len(scores),
        evidence_unsupported_answer_rate=(
            unsupported_claims / answered_claims if answered_claims else None
        ),
        invalid_memory_acceptance_rate=(
            invalid_acceptances / len(mismatch_decisions) if mismatch_decisions else None
        ),
        memory_mismatch_rejection_rate=(
            rejections / len(mismatch_decisions) if mismatch_decisions else None
        ),
        mean_tool_calls=sum(tool_calls) / len(tool_calls),
        abstention_precision=(
            sum(episodes[index].task.insufficient for index in abstained_indices)
            / len(abstained_indices)
            if abstained_indices
            else None
        ),
        abstention_recall=(
            sum(episodes[index].abstention_reason is not None for index in insufficient_indices)
            / len(insufficient_indices)
            if insufficient_indices
            else None
        ),
        answer_coverage=len(answered_indices) / len(episodes),
        selective_risk_at_observed_coverage=(
            sum(not strict_success[index] for index in answered_indices) / len(answered_indices)
            if answered_indices
            else None
        ),
        conflict_resolution_rate=(
            sum(strict_success[index] for index in conflict_indices) / len(conflict_indices)
            if conflict_indices
            else None
        ),
    )


def selective_risk_coverage(
    records: Iterable[ConfidenceScoredEpisode],
) -> tuple[RiskCoveragePoint, ...]:
    """Build a tie-safe curve from explicit confidence on submitted answers.

    Coverage uses every supplied episode as its denominator. Abstentions and
    non-terminal/no-answer episodes are therefore not silently recast as
    predictions. A point is emitted only after all answers sharing a confidence
    value have been included.
    """

    items = tuple(records)
    if not items:
        raise ValueError("at least one confidence-scored episode is required")
    episode_ids = [item.state.episode_id for item in items]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")
    if any(not isfinite(item.confidence) or not 0.0 <= item.confidence <= 1.0 for item in items):
        raise ValueError("confidence must be finite and in [0, 1]")

    verifier = DeterministicVerifier()
    answered = sorted(
        (item for item in items if item.state.final_claims),
        key=lambda item: item.confidence,
        reverse=True,
    )
    points: list[RiskCoveragePoint] = []
    retained = 0
    errors = 0
    index = 0
    while index < len(answered):
        threshold = answered[index].confidence
        while index < len(answered) and answered[index].confidence == threshold:
            score = verifier.score(answered[index].state)
            retained += 1
            errors += not verifier.is_success(score)
            index += 1
        points.append(
            RiskCoveragePoint(
                confidence_threshold=threshold,
                coverage=retained / len(items),
                risk=errors / retained,
                retained_answers=retained,
                total_episodes=len(items),
            )
        )
    return tuple(points)
