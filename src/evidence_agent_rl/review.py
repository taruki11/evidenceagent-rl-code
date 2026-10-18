"""Independent-review records and deterministic promotion gates."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .models import Document, Span, TaskSpec


@dataclass(frozen=True)
class ReviewRecord:
    reviewer_id: str
    task_id: str
    verdict: str
    question_correct: bool
    answer_key_correct: bool
    insufficiency_correct: bool
    memory_label_correct: bool | None
    approved_span_ids: tuple[str, ...]
    notes: str = ""


def load_reviews(path: str | Path) -> tuple[ReviewRecord, ...]:
    records: list[ReviewRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("verdict") == "pending":
                continue
            reviewer_id = str(raw.get("reviewer_id", "")).strip()
            if not reviewer_id or reviewer_id == "REPLACE_ME":
                raise ValueError(f"review line {line_number} has no real reviewer_id")
            verdict = str(raw.get("verdict", ""))
            if verdict not in {"accept", "revise", "reject"}:
                raise ValueError(f"review line {line_number} has invalid verdict")
            records.append(
                ReviewRecord(
                    reviewer_id=reviewer_id,
                    task_id=str(raw["task_id"]),
                    verdict=verdict,
                    question_correct=bool(raw.get("question_correct", False)),
                    answer_key_correct=bool(raw.get("answer_key_correct", False)),
                    insufficiency_correct=bool(raw.get("insufficiency_correct", False)),
                    memory_label_correct=raw.get("memory_label_correct"),
                    approved_span_ids=tuple(str(item) for item in raw.get("approved_span_ids", [])),
                    notes=str(raw.get("notes", "")),
                )
            )
    return tuple(records)


def required_span_ids(task: TaskSpec, documents: Iterable[Document]) -> frozenset[str]:
    if task.insufficient:
        return frozenset()
    required_facets = set(task.required_facets)
    return frozenset(
        span.span_id
        for document in documents
        if document.doi in task.allowed_dois
        for span in document.spans
        if set(span.facets) & required_facets
    )


def select_review_tasks(
    tasks: Iterable[TaskSpec],
    *,
    target: int = 40,
    mandatory_splits: tuple[str, ...] = ("dev", "test"),
) -> tuple[TaskSpec, ...]:
    """Select a deterministic review set without splitting mismatch pairs."""
    task_list = tuple(tasks)
    if target <= 0 or target > len(task_list):
        raise ValueError("review target must be within the available task count")
    pair_members: dict[str, tuple[TaskSpec, ...]] = {}
    for task in task_list:
        if task.mismatch_pair_id:
            pair_members.setdefault(task.mismatch_pair_id, tuple())
            pair_members[task.mismatch_pair_id] += (task,)

    selected_ids = {
        task.task_id for task in task_list if task.split in set(mandatory_splits)
    }
    for task in task_list:
        if task.task_id in selected_ids and task.mismatch_pair_id:
            selected_ids.update(
                member.task_id for member in pair_members[task.mismatch_pair_id]
            )
    if len(selected_ids) > target:
        raise ValueError("mandatory review tasks exceed the requested target")

    groups: dict[str, tuple[TaskSpec, ...]] = {}
    for task in task_list:
        key = task.mismatch_pair_id or f"task:{task.task_id}"
        groups.setdefault(key, tuple())
        groups[key] += (task,)
    type_priority = {"multi_hop": 0, "conflict": 1, "insufficient": 2, "direct_evidence": 3}
    candidates = [
        group
        for group in groups.values()
        if not any(member.task_id in selected_ids for member in group)
    ]
    candidates.sort(
        key=lambda group: (
            min(type_priority.get(member.task_type, 9) for member in group),
            -len({paper for member in group for paper in member.all_source_paper_ids}),
            min(member.task_id for member in group),
        )
    )
    for group in candidates:
        if len(selected_ids) == target:
            break
        if len(selected_ids) + len(group) <= target:
            selected_ids.update(member.task_id for member in group)
    if len(selected_ids) != target:
        raise ValueError(
            f"cannot reach review target {target} without splitting a task group"
        )
    return tuple(task for task in task_list if task.task_id in selected_ids)


def _is_complete_accept(
    review: ReviewRecord,
    task: TaskSpec,
    candidate_spans: dict[str, Span],
) -> bool:
    if review.verdict != "accept":
        return False
    if not (
        review.question_correct
        and review.answer_key_correct
        and review.insufficiency_correct
    ):
        return False
    if task.mismatch_pair_id and review.memory_label_correct is not True:
        return False
    approved_ids = set(review.approved_span_ids)
    if not approved_ids <= candidate_spans.keys():
        return False
    if task.insufficient:
        return not approved_ids
    approved_facets = {
        facet
        for span_id in approved_ids
        for facet in candidate_spans[span_id].facets
    }
    return set(task.required_facets) <= approved_facets


def apply_review_gate(
    tasks: Iterable[TaskSpec],
    documents: Iterable[Document],
    reviews: Iterable[ReviewRecord],
) -> tuple[tuple[TaskSpec, ...], tuple[Document, ...]]:
    task_list = tuple(tasks)
    document_list = tuple(documents)
    task_index = {task.task_id: task for task in task_list}
    seen: set[tuple[str, str]] = set()
    accepted_by_task: dict[str, set[str]] = {}
    accepted_by_span: dict[str, set[str]] = {}
    for review in reviews:
        task = task_index.get(review.task_id)
        if task is None:
            raise ValueError(f"review references unknown task: {review.task_id}")
        key = (review.reviewer_id, review.task_id)
        if key in seen:
            raise ValueError(f"duplicate review from {review.reviewer_id} for {review.task_id}")
        seen.add(key)
        candidate_ids = required_span_ids(task, document_list)
        candidate_spans = {
            span.span_id: span
            for document in document_list
            for span in document.spans
            if span.span_id in candidate_ids
        }
        if not _is_complete_accept(review, task, candidate_spans):
            continue
        accepted_by_task.setdefault(task.task_id, set()).add(review.reviewer_id)
        for span_id in review.approved_span_ids:
            if span_id in candidate_spans:
                accepted_by_span.setdefault(span_id, set()).add(review.reviewer_id)

    def status(reviewers: set[str]) -> str:
        if len(reviewers) >= 2:
            return "double_review"
        if len(reviewers) == 1:
            return "single_review"
        return "draft"

    promoted_tasks = tuple(
        replace(task, review_status=status(accepted_by_task.get(task.task_id, set())))
        for task in task_list
    )
    promoted_documents = tuple(
        replace(
            document,
            spans=tuple(
                replace(span, review_status=status(accepted_by_span.get(span.span_id, set())))
                for span in document.spans
            ),
        )
        for document in document_list
    )
    return promoted_tasks, promoted_documents


def task_to_json(task: TaskSpec) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "source_paper_id": task.source_paper_id,
        "source_paper_ids": list(task.source_paper_ids),
        "split": task.split,
        "task_type": task.task_type,
        "question": task.question,
        "corpus_scope": task.corpus_scope,
        "required_facets": list(task.required_facets),
        "allowed_dois": sorted(task.allowed_dois),
        "insufficient": task.insufficient,
        "mismatch_pair_id": task.mismatch_pair_id,
        "mismatch_role": task.mismatch_role,
        "review_status": task.review_status,
        "available_memory_ids": list(task.available_memory_ids or ()),
        "deterministic_answer_key": {
            facet: list(tokens) for facet, tokens in task.deterministic_answer_key.items()
        },
        "forbidden_answer_tokens": list(task.forbidden_answer_tokens),
        "required_query_terms": list(task.required_query_terms),
    }


def document_to_json(document: Document) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "source_paper_id": document.source_paper_id,
        "title": document.title,
        "doi": document.doi,
        "corpus_scope": document.corpus_scope,
        "document_type": document.document_type,
        "spans": [
            {
                "span_id": span.span_id,
                "text": span.text,
                "facets": list(span.facets),
                "page_number": span.page_number,
                "page_text_sha256": span.page_text_sha256,
                "review_status": span.review_status,
            }
            for span in document.spans
        ],
    }
