"""Provenance and review gates for curated evidence spans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Document, TaskSpec


REVIEW_STATUSES = {"draft", "single_review", "double_review"}
DATASET_QUALITY_GATES = {
    "auto_provenance",
    "single_review",
    "double_review",
    "phase3_t1_correction_traces",
    "crosssource_correction_traces",
}


@dataclass(frozen=True)
class PageRecord:
    document_id: str
    page_number: int
    text: str
    text_sha256: str


def canonicalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_page_records(path: str | Path) -> tuple[PageRecord, ...]:
    records: list[PageRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            records.append(
                PageRecord(
                    document_id=str(raw["document_id"]),
                    page_number=int(raw["page_number"]),
                    text=str(raw["text"]),
                    text_sha256=str(raw["text_sha256"]),
                )
            )
    return tuple(records)


def validate_curated_spans(
    documents: Iterable[Document], pages: Iterable[PageRecord]
) -> None:
    page_index = {(page.document_id, page.page_number): page for page in pages}
    for document in documents:
        for span in document.spans:
            if span.review_status not in REVIEW_STATUSES:
                raise ValueError(f"span {span.span_id} has invalid review_status")
            if span.page_number is None or not span.page_text_sha256:
                raise ValueError(f"span {span.span_id} has incomplete page provenance")
            page = page_index.get((document.document_id, span.page_number))
            if page is None:
                raise ValueError(f"span {span.span_id} points to an unknown page")
            if page.text_sha256 != span.page_text_sha256:
                raise ValueError(f"span {span.span_id} page hash does not match")
            if canonicalize_text(span.text) not in canonicalize_text(page.text):
                raise ValueError(f"span {span.span_id} is not a verbatim extracted page span")


def validate_review_gate(tasks: Iterable[TaskSpec], minimum: str) -> None:
    rank = {"draft": 0, "single_review": 1, "double_review": 2}
    if minimum not in rank:
        raise ValueError(f"invalid minimum review status: {minimum}")
    for task in tasks:
        if task.review_status not in rank:
            raise ValueError(f"task {task.task_id} has invalid review_status")
        if rank[task.review_status] < rank[minimum]:
            raise ValueError(
                f"task {task.task_id} is {task.review_status}, below required {minimum}"
            )


def validate_evidence_review_gate(
    tasks: Iterable[TaskSpec], documents: Iterable[Document], minimum: str
) -> None:
    """Require human-reviewed answer evidence, not only a reviewed task label."""

    rank = {"draft": 0, "single_review": 1, "double_review": 2}
    if minimum not in rank or minimum == "draft":
        raise ValueError("evidence review gate requires single_review or double_review")
    spans_by_doi: dict[str, tuple] = {}
    for document in documents:
        spans_by_doi[document.doi] = spans_by_doi.get(document.doi, ()) + document.spans
    for task in tasks:
        if task.insufficient:
            continue
        for facet in task.required_facets:
            answer_tokens = task.deterministic_answer_key.get(facet, ())
            eligible = [
                span
                for doi in task.allowed_dois
                for span in spans_by_doi.get(doi, ())
                if facet in span.facets
                and all(token.casefold() in span.text.casefold() for token in answer_tokens)
                and rank.get(span.review_status, -1) >= rank[minimum]
            ]
            if not eligible:
                raise ValueError(
                    f"task {task.task_id} facet {facet} lacks {minimum} answer evidence"
                )


def validate_task_evidence_contract(
    tasks: Iterable[TaskSpec], documents: Iterable[Document]
) -> None:
    document_list = tuple(documents)
    document_dois = {document.doi for document in document_list}
    document_sources = {document.source_paper_id for document in document_list}
    spans_by_doi: dict[str, tuple] = {}
    for document in document_list:
        spans_by_doi[document.doi] = spans_by_doi.get(document.doi, ()) + document.spans
    for task in tasks:
        missing_sources = set(task.all_source_paper_ids) - document_sources
        if missing_sources:
            raise ValueError(f"task {task.task_id} references unknown source papers")
        if task.insufficient:
            continue
        if not task.deterministic_answer_key:
            raise ValueError(f"task {task.task_id} has no deterministic answer key")
        missing_dois = set(task.allowed_dois) - document_dois
        if missing_dois:
            raise ValueError(f"task {task.task_id} references unknown DOI")
        for facet in task.required_facets:
            answer_tokens = task.deterministic_answer_key.get(facet)
            if not answer_tokens:
                raise ValueError(f"task {task.task_id} facet {facet} has no answer tokens")
            supporting_spans = [
                span
                for doi in task.allowed_dois
                for span in spans_by_doi.get(doi, ())
                if facet in span.facets
                and all(token.casefold() in span.text.casefold() for token in answer_tokens)
            ]
            if not supporting_spans:
                raise ValueError(
                    f"task {task.task_id} facet {facet} has no provenance-grounded answer span"
                )


def validate_dataset_quality_gate(
    tasks: Iterable[TaskSpec],
    documents: Iterable[Document],
    pages: Iterable[PageRecord],
    *,
    gate: str,
    split: str,
) -> None:
    """Separate automatic train provenance from human evaluation labels."""

    if gate not in DATASET_QUALITY_GATES:
        raise ValueError(f"invalid dataset quality gate: {gate}")
    task_list = tuple(tasks)
    document_list = tuple(documents)
    validate_curated_spans(document_list, pages)
    validate_task_evidence_contract(task_list, document_list)
    if gate in {"phase3_t1_correction_traces", "crosssource_correction_traces"}:
        # Machine-generated supervision, deliberately draft: T1's traces supervise ONE action, and
        # the cross-source traces supervise a five-step trajectory.  Neither is human gold, so the
        # human review gate below does not apply, and the exact asset is pinned by
        # data.correction_traces_manifest instead.  A separate name rather than T1's, because
        # registration types must say which experiment they belong to.
        if split != "train":
            raise ValueError(f"{gate} is permitted only for the train split")
        return
    if gate == "auto_provenance":
        if split != "train":
            raise ValueError("auto_provenance is permitted only for the train split")
        unreviewed_insufficient = [
            task.task_id
            for task in task_list
            if task.insufficient and task.review_status == "draft"
        ]
        if unreviewed_insufficient:
            raise ValueError(
                "automatic provenance cannot validate evidence absence for insufficient tasks: "
                f"{unreviewed_insufficient}"
            )
        return
    validate_review_gate(task_list, gate)
    validate_evidence_review_gate(task_list, document_list, gate)


def validate_mismatch_pairs(tasks: Iterable[TaskSpec]) -> None:
    pairs: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        if task.mismatch_pair_id:
            pairs.setdefault(task.mismatch_pair_id, []).append(task)
    invariant_fields = (
        "source_paper_id",
        "source_paper_ids",
        "split",
        "question",
        "corpus_scope",
        "required_facets",
        "allowed_dois",
        "insufficient",
        "deterministic_answer_key",
        "forbidden_answer_tokens",
        "required_query_terms",
        "task_type",
    )
    for pair_id, members in pairs.items():
        roles = {member.mismatch_role for member in members}
        if len(members) != 2 or roles != {"matched", "mismatched"}:
            raise ValueError(f"mismatch pair {pair_id} must have exactly matched/mismatched roles")
        first, second = members
        for field_name in invariant_fields:
            if getattr(first, field_name) != getattr(second, field_name):
                raise ValueError(f"mismatch pair {pair_id} changes {field_name}")
        if first.available_memory_ids == second.available_memory_ids:
            raise ValueError(f"mismatch pair {pair_id} does not change memory exposure")
