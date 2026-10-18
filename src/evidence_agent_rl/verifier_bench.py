"""Replay-based evaluation for deterministic or learned trajectory verifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .harness import EvidenceHarness
from .models import Action, Document, StrategyMemory, TaskSpec
from .verifier import DeterministicVerifier, RewardBreakdown


REVIEW_RANK = {"draft": 0, "single_review": 1, "double_review": 2}
OWNER_DELEGATED_REVIEW_STATUS = "owner_delegated_ai_adjudication"
OWNER_DELEGATED_LABEL_SOURCE = "owner_delegated_ai"


@dataclass(frozen=True)
class VerifierBenchCase:
    case_id: str
    task_id: str
    gold_success: bool
    label_source: str
    review_status: str
    failure_mode: str
    actions: tuple[Action, ...]
    notes: str = ""


@dataclass(frozen=True)
class VerifierBenchPrediction:
    case_id: str
    gold_success: bool
    predicted_success: bool
    success_probability: float
    failure_mode: str
    score: RewardBreakdown


@dataclass(frozen=True)
class VerifierBenchReport:
    cases: int
    accuracy: float
    false_success_rate: float | None
    false_failure_rate: float | None
    expected_calibration_error: float
    cohens_kappa: float | None


def load_verifier_bench(
    path: str | Path,
    *,
    minimum_review: str = "draft",
    require_human: bool = False,
    allow_owner_delegated_ai: bool = False,
) -> tuple[VerifierBenchCase, ...]:
    if minimum_review not in REVIEW_RANK:
        raise ValueError(f"invalid minimum review status: {minimum_review}")
    cases: list[VerifierBenchCase] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: case must be an object")
            case_id = str(raw.get("case_id", "")).strip()
            if not case_id or case_id in seen:
                raise ValueError(f"{path}:{line_number}: missing or duplicate case_id")
            seen.add(case_id)
            review_status = str(raw.get("review_status", "draft"))
            owner_delegated = review_status == OWNER_DELEGATED_REVIEW_STATUS
            if owner_delegated and not allow_owner_delegated_ai:
                raise ValueError(f"{case_id}: owner-delegated AI labels require explicit opt-in")
            if review_status not in REVIEW_RANK and not owner_delegated:
                raise ValueError(f"{case_id}: invalid review status")
            review_rank = REVIEW_RANK["double_review"] if owner_delegated else REVIEW_RANK[review_status]
            if review_rank < REVIEW_RANK[minimum_review]:
                raise ValueError(f"{case_id}: below required review status {minimum_review}")
            label_source = str(raw.get("label_source", ""))
            allowed_sources = {"synthetic_canary", "human"}
            if allow_owner_delegated_ai:
                allowed_sources.add(OWNER_DELEGATED_LABEL_SOURCE)
            if label_source not in allowed_sources:
                raise ValueError(f"{case_id}: invalid label_source")
            if owner_delegated != (label_source == OWNER_DELEGATED_LABEL_SOURCE):
                raise ValueError(f"{case_id}: owner-delegated provenance fields disagree")
            if require_human and (
                label_source != "human" or review_status != "double_review"
            ):
                raise ValueError(
                    f"{case_id}: headline VerifierBench requires double-reviewed human labels"
                )
            if not isinstance(raw.get("gold_success"), bool):
                raise ValueError(f"{case_id}: gold_success must be a boolean")
            raw_actions = raw.get("actions")
            if not isinstance(raw_actions, list):
                raise ValueError(f"{case_id}: actions must be an array")
            parsed_actions: list[Action] = []
            for action_index, action in enumerate(raw_actions):
                if not isinstance(action, dict) or not isinstance(action.get("name"), str):
                    raise ValueError(f"{case_id}: action {action_index} is invalid")
                arguments = action.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ValueError(f"{case_id}: action {action_index} arguments must be an object")
                parsed_actions.append(Action(action["name"], dict(arguments)))
            actions = tuple(parsed_actions)
            cases.append(
                VerifierBenchCase(
                    case_id=case_id,
                    task_id=str(raw["task_id"]),
                    gold_success=raw["gold_success"],
                    label_source=label_source,
                    review_status=review_status,
                    failure_mode=str(raw.get("failure_mode", "unspecified")),
                    actions=actions,
                    notes=str(raw.get("notes", "")),
                )
            )
    if not cases:
        raise ValueError("VerifierBench must contain at least one case")
    return tuple(cases)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cohens_kappa(gold: list[bool], predicted: list[bool]) -> float | None:
    count = len(gold)
    observed = sum(left == right for left, right in zip(gold, predicted)) / count
    gold_positive = sum(gold) / count
    predicted_positive = sum(predicted) / count
    expected = (
        gold_positive * predicted_positive
        + (1 - gold_positive) * (1 - predicted_positive)
    )
    return (observed - expected) / (1 - expected) if expected != 1 else None


def _ece(probabilities: list[float], gold: list[bool], bins: int = 10) -> float:
    total = len(gold)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probabilities[index] for index in members) / len(members)
        accuracy = sum(gold[index] for index in members) / len(members)
        error += len(members) / total * abs(confidence - accuracy)
    return error


def evaluate_verifier_bench(
    cases: Iterable[VerifierBenchCase],
    tasks: Iterable[TaskSpec],
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    *,
    probability_fn: Callable[[RewardBreakdown], float] | None = None,
    max_budget: int = 8,
) -> tuple[VerifierBenchReport, tuple[VerifierBenchPrediction, ...]]:
    case_list = tuple(cases)
    task_index = {task.task_id: task for task in tasks}
    document_list = tuple(documents)
    memory_list = tuple(memories)
    verifier = DeterministicVerifier()
    predictions: list[VerifierBenchPrediction] = []
    for case in case_list:
        task = task_index.get(case.task_id)
        if task is None:
            raise ValueError(f"{case.case_id}: unknown task {case.task_id}")
        harness = EvidenceHarness(document_list, memory_list, max_budget=max_budget)
        # Frozen action handles are scoped to the original rollout identity.
        # Replaying under a synthetic prefix makes every checked citation look
        # cross-episode and turns valid trajectories into firewall failures.
        state = harness.replay(task, case.actions, episode_id=case.case_id)
        score = verifier.score(state)
        probability = (
            probability_fn(score)
            if probability_fn is not None
            else float(verifier.is_success(score))
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{case.case_id}: verifier probability is outside [0, 1]")
        predictions.append(
            VerifierBenchPrediction(
                case.case_id,
                case.gold_success,
                probability >= 0.5,
                probability,
                case.failure_mode,
                score,
            )
        )
    gold = [prediction.gold_success for prediction in predictions]
    predicted = [prediction.predicted_success for prediction in predictions]
    false_successes = sum(p and not g for p, g in zip(predicted, gold))
    false_failures = sum(not p and g for p, g in zip(predicted, gold))
    negatives = sum(not item for item in gold)
    positives = sum(gold)
    report = VerifierBenchReport(
        cases=len(predictions),
        accuracy=sum(p == g for p, g in zip(predicted, gold)) / len(predictions),
        false_success_rate=_safe_rate(false_successes, negatives),
        false_failure_rate=_safe_rate(false_failures, positives),
        expected_calibration_error=_ece(
            [prediction.success_probability for prediction in predictions], gold
        ),
        cohens_kappa=_cohens_kappa(gold, predicted),
    )
    return report, tuple(predictions)
