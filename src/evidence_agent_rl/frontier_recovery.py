"""Leakage-safe contracts for Paper-v3 verifier-guided frontier recovery."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .models import TaskSpec
from .recovery import RecoveryCandidate
from .sft import SFTExample, validate_public_context, validate_sft_examples


PAPER_V3_TASK_TYPES = (
    "multi_hop",
    "conflict",
    "direct_evidence",
    "insufficient",
)
PAPER_V3_DEFAULT_QUOTAS = {
    "multi_hop": 16,
    "conflict": 8,
    "direct_evidence": 8,
}
PAPER_V3_SOURCE_QUOTAS = {
    "multi_hop": 64,
    "conflict": 64,
    "direct_evidence": 64,
}
PAPER_V3_REVIEW_CHECKS = (
    "action_sequence_correct",
    "memory_boundary_correct",
    "evidence_chain_correct",
    "answer_correct",
    "train_split_safe",
    "feedback_boundary_correct",
    "student_context_feedback_free",
)

_FRONTIER_ACTIONS = (
    "search",
    "select_evidence",
    "answer",
    "abstain",
)
_FEEDBACK_TEXT = {
    "invalid_action": (
        "The previous decision was not executable. Continue with one valid structured "
        "evidence-acquisition action."
    ),
    "premature_terminal": (
        "The submitted terminal decision was not verified as correct. Revise it or "
        "continue evidence acquisition using only the public state."
    ),
    "incomplete": (
        "The trajectory ended before a verified terminal decision. Continue from the "
        "public state with an evidence-acquisition action."
    ),
    "tool_error": (
        "The previous action did not produce a usable observation. Recover with a valid "
        "evidence-acquisition action based only on the public state."
    ),
}


@dataclass(frozen=True)
class FrontierFeedback:
    """Generic training-only feedback; it never contains task-specific gold."""

    version: int
    feedback_kind: str
    instruction: str
    allowed_action_classes: tuple[str, ...]
    training_only: bool
    student_context_visibility: str

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_action_classes"] = list(self.allowed_action_classes)
        return payload


@dataclass(frozen=True)
class RecoveryCreditContract:
    """Fixed CAPF-inspired attenuation contract for repaired trajectories."""

    version: int = 1
    method: str = "fixed_pre_feedback_attenuation_v1"
    pre_feedback_multiplier: float = 0.25
    feedback_credit: float = 0.0
    post_feedback_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.version != 1 or self.method != "fixed_pre_feedback_attenuation_v1":
            raise ValueError("unsupported recovery credit contract identity")
        values = (
            self.pre_feedback_multiplier,
            self.feedback_credit,
            self.post_feedback_multiplier,
        )
        if any(not isinstance(value, (int, float)) for value in values):
            raise TypeError("recovery credit multipliers must be numeric")
        if not 0.0 <= float(self.pre_feedback_multiplier) <= 1.0:
            raise ValueError("pre-feedback multiplier must be in [0, 1]")
        if float(self.feedback_credit) != 0.0:
            raise ValueError("verifier feedback must receive zero policy credit")
        if float(self.post_feedback_multiplier) != 1.0:
            raise ValueError("post-feedback recovery credit must remain unattenuated")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _feedback_kind(failure_category: str) -> str:
    normalized = failure_category.casefold()
    if "invalid" in normalized or "decode" in normalized or "json" in normalized:
        return "invalid_action"
    if "answer" in normalized or "abstention" in normalized:
        return "premature_terminal"
    if "limit" in normalized or "incomplete" in normalized:
        return "incomplete"
    return "tool_error"


def build_frontier_feedback(failure_category: str) -> FrontierFeedback:
    """Map a failure class to fixed feedback without reading hidden task fields."""

    kind = _feedback_kind(failure_category)
    feedback = FrontierFeedback(
        version=1,
        feedback_kind=kind,
        instruction=_FEEDBACK_TEXT[kind],
        allowed_action_classes=_FRONTIER_ACTIONS,
        training_only=True,
        student_context_visibility="stripped_before_student_supervision",
    )
    validate_frontier_feedback(feedback)
    return feedback


def validate_frontier_feedback(feedback: FrontierFeedback | Mapping[str, Any]) -> None:
    """Reject dynamic or answer-bearing feedback fields and non-frozen wording."""

    payload = feedback.to_json() if isinstance(feedback, FrontierFeedback) else dict(feedback)
    required = {
        "version",
        "feedback_kind",
        "instruction",
        "allowed_action_classes",
        "training_only",
        "student_context_visibility",
    }
    if set(payload) != required:
        raise ValueError("frontier feedback contains non-contract fields")
    kind = str(payload["feedback_kind"])
    if kind not in _FEEDBACK_TEXT or payload["instruction"] != _FEEDBACK_TEXT[kind]:
        raise ValueError("frontier feedback must use frozen generic wording")
    if payload["training_only"] is not True:
        raise ValueError("frontier feedback must be training-only")
    if payload["student_context_visibility"] != "stripped_before_student_supervision":
        raise ValueError("frontier feedback must be stripped from student supervision")
    if tuple(payload["allowed_action_classes"]) != _FRONTIER_ACTIONS:
        raise ValueError("frontier feedback action classes differ from the frozen contract")


def apply_recovery_credit_attenuation(
    pre_feedback_credits: Iterable[float],
    post_feedback_credits: Iterable[float],
    *,
    contract: RecoveryCreditContract | None = None,
) -> dict[str, Any]:
    """Apply the auditable fixed contract without inventing new reward labels."""

    frozen = contract or RecoveryCreditContract()
    before = tuple(float(value) for value in pre_feedback_credits)
    after = tuple(float(value) for value in post_feedback_credits)
    if not after:
        raise ValueError("recovery credit requires at least one post-feedback action")
    return {
        "contract": frozen.to_json(),
        "pre_feedback": [frozen.pre_feedback_multiplier * value for value in before],
        "feedback": frozen.feedback_credit,
        "post_feedback": [frozen.post_feedback_multiplier * value for value in after],
    }


def _stable_rank(seed: int, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode("utf-8")).hexdigest()


def select_frontier_source_tasks(
    tasks: Iterable[TaskSpec],
    *,
    quotas: Mapping[str, int] | None = None,
    seed: int = 42,
) -> tuple[TaskSpec, ...]:
    """Freeze a broad, reviewed train-only source pool before observing outcomes."""

    frozen_quotas = dict(quotas or PAPER_V3_SOURCE_QUOTAS)
    if not frozen_quotas or any(
        task_type not in PAPER_V3_TASK_TYPES or count <= 0
        for task_type, count in frozen_quotas.items()
    ):
        raise ValueError("frontier source quotas must be positive known task types")
    rows = tuple(tasks)
    if len({task.task_id for task in rows}) != len(rows):
        raise ValueError("frontier source tasks contain duplicate task_id")

    selected: list[TaskSpec] = []
    for task_type in PAPER_V3_TASK_TYPES:
        if task_type not in frozen_quotas:
            continue
        pool = [
            task
            for task in rows
            if task.split == "train"
            and task.review_status == "double_review"
            and task.task_type == task_type
            and task.mismatch_pair_id is None
        ]
        required = frozen_quotas[task_type]
        if len(pool) < required:
            raise ValueError(f"frontier source lacks {task_type}: {len(pool)} < {required}")

        paper_counts: Counter[str] = Counter()
        chosen: list[TaskSpec] = []
        while len(chosen) < required:
            pool.sort(
                key=lambda task: (
                    paper_counts["|".join(task.all_source_paper_ids)],
                    _stable_rank(seed, task.task_id),
                )
            )
            task = pool.pop(0)
            chosen.append(task)
            paper_counts["|".join(task.all_source_paper_ids)] += 1
        selected.extend(chosen)
    return tuple(selected)


def select_frontier_recovery_candidates(
    candidates: Iterable[RecoveryCandidate],
    tasks: Iterable[TaskSpec],
    *,
    quotas: Mapping[str, int] | None = None,
    seed: int = 42,
) -> tuple[RecoveryCandidate, ...]:
    """Freeze a task-type-stratified, source-diverse recovery canary."""

    frozen_quotas = dict(quotas or PAPER_V3_DEFAULT_QUOTAS)
    if not frozen_quotas or any(
        task_type not in PAPER_V3_TASK_TYPES or count <= 0
        for task_type, count in frozen_quotas.items()
    ):
        raise ValueError("frontier recovery quotas must be positive known task types")
    task_map = {task.task_id: task for task in tasks}
    rows = tuple(candidates)
    if len({candidate.candidate_id for candidate in rows}) != len(rows):
        raise ValueError("frontier recovery candidates contain duplicate candidate_id")

    selected: list[RecoveryCandidate] = []
    for task_type in PAPER_V3_TASK_TYPES:
        if task_type not in frozen_quotas:
            continue
        pool: list[RecoveryCandidate] = []
        for candidate in rows:
            task = task_map.get(candidate.source_task_id)
            if task is None or task.split != "train":
                raise ValueError("frontier recovery candidate references an unknown train task")
            if task.task_type == task_type:
                pool.append(candidate)
        required = frozen_quotas[task_type]
        if len(pool) < required:
            raise ValueError(
                f"frontier recovery lacks {task_type}: {len(pool)} < {required}"
            )

        paper_counts: Counter[str] = Counter()
        remaining = list(pool)
        chosen: list[RecoveryCandidate] = []
        while len(chosen) < required:
            remaining.sort(
                key=lambda candidate: (
                    paper_counts[
                        "|".join(candidate.source_paper_ids) or "<unknown-paper>"
                    ],
                    _stable_rank(seed, candidate.candidate_id),
                )
            )
            candidate = remaining.pop(0)
            chosen.append(candidate)
            paper_counts["|".join(candidate.source_paper_ids) or "<unknown-paper>"] += 1
        selected.extend(chosen)
    return tuple(selected)


def build_frontier_recovery_row(
    candidate: RecoveryCandidate,
    task: TaskSpec,
    *,
    credit_contract: RecoveryCreditContract | None = None,
) -> dict[str, Any]:
    """Attach audit metadata while keeping every student prompt feedback-free."""

    if task.task_id != candidate.source_task_id or task.split != "train":
        raise ValueError("frontier recovery row requires the candidate source train task")
    feedback = build_frontier_feedback(candidate.failure_category)
    for example in candidate.examples:
        context = json.loads(example.messages[1]["content"])
        validate_public_context(context)
        if "training_feedback" in context or "frontier_feedback" in context:
            raise ValueError("student recovery context contains privileged feedback")
    row = candidate.to_json()
    row.update(
        {
            "paper_v3_version": 1,
            "task_type": task.task_type,
            "training_feedback": feedback.to_json(),
            "credit_contract": (credit_contract or RecoveryCreditContract()).to_json(),
            "epistemic_status": (
                "train_only_frontier_recovery_candidate_requires_independent_review"
            ),
        }
    )
    return row


def validate_frontier_recovery_row(row: Mapping[str, Any]) -> None:
    """Validate a serialized Paper-v3 row before review or promotion."""

    if row.get("paper_v3_version") != 1:
        raise ValueError("frontier recovery row has an unsupported version")
    if row.get("task_type") not in PAPER_V3_TASK_TYPES:
        raise ValueError("frontier recovery row has an unknown task type")
    if row.get("review_status") != "draft_requires_independent_review":
        raise ValueError("frontier recovery row bypasses independent review")
    validate_frontier_feedback(dict(row.get("training_feedback") or {}))
    RecoveryCreditContract(**dict(row.get("credit_contract") or {}))
    examples = list(row.get("examples") or [])
    if not examples:
        raise ValueError("frontier recovery row has no student examples")
    parsed_examples: list[SFTExample] = []
    for example in examples:
        messages = list(example.get("messages") or [])
        if len(messages) != 3 or messages[1].get("role") != "user":
            raise ValueError("frontier recovery row has an invalid student message sequence")
        context = json.loads(messages[1]["content"])
        validate_public_context(context)
        if "training_feedback" in context or "frontier_feedback" in context:
            raise ValueError("student recovery context contains privileged feedback")
        parsed_examples.append(
            SFTExample(
                example_id=str(example["example_id"]),
                task_id=str(example["task_id"]),
                split=str(example["split"]),
                source_paper_ids=tuple(example["source_paper_ids"]),
                review_status=str(example["review_status"]),
                step_index=int(example["step_index"]),
                messages=tuple(messages),
            )
        )
    validate_sft_examples(parsed_examples)


def build_frontier_recovery_review_template(
    rows: Iterable[Mapping[str, Any]], *, reviewer_id: str
) -> tuple[dict[str, Any], ...]:
    """Create a neutral Paper-v3 review file with explicit feedback checks."""

    if not reviewer_id.strip():
        raise ValueError("frontier recovery reviewer_id cannot be empty")
    result: list[dict[str, Any]] = []
    for row in rows:
        validate_frontier_recovery_row(row)
        review = {
            "reviewer_id": reviewer_id,
            "reviewer_kind": "ai_model",
            "candidate_id": str(row["candidate_id"]),
            "verdict": "REPLACE_ME",
            **{check: None for check in PAPER_V3_REVIEW_CHECKS},
            "approved_example_ids": [],
            "notes": "",
        }
        result.append(review)
    return tuple(result)
