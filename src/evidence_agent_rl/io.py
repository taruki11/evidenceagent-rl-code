"""JSONL boundary for datasets and replayable trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import (
    Document,
    EpisodeState,
    GuidanceKind,
    SourceState,
    Span,
    StrategyMemory,
    TaskSpec,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
            records.append(record)
    return records


def _require(record: dict[str, Any], key: str) -> Any:
    if key not in record:
        raise ValueError(f"missing required field: {key}")
    return record[key]


def load_documents(path: str | Path) -> tuple[Document, ...]:
    documents: list[Document] = []
    for record in _read_jsonl(path):
        spans = tuple(
            Span(
                span_id=str(_require(span, "span_id")),
                text=str(_require(span, "text")),
                facets=tuple(str(item) for item in _require(span, "facets")),
                page_number=span.get("page_number"),
                page_text_sha256=str(span.get("page_text_sha256", "")),
                review_status=str(span.get("review_status", "draft")),
            )
            for span in _require(record, "spans")
        )
        documents.append(
            Document(
                document_id=str(_require(record, "document_id")),
                title=str(_require(record, "title")),
                doi=str(_require(record, "doi")),
                corpus_scope=str(_require(record, "corpus_scope")),
                document_type=str(_require(record, "document_type")),
                spans=spans,
                source_paper_id=str(_require(record, "source_paper_id")),
            )
        )
    return tuple(documents)


def load_tasks(path: str | Path) -> tuple[TaskSpec, ...]:
    tasks: list[TaskSpec] = []
    for record in _read_jsonl(path):
        split = str(_require(record, "split"))
        if split not in {"train", "verifierbench", "dev", "test", "ood"}:
            raise ValueError(f"invalid task split: {split}")
        tasks.append(
            TaskSpec(
                task_id=str(_require(record, "task_id")),
                question=str(_require(record, "question")),
                corpus_scope=str(_require(record, "corpus_scope")),
                required_facets=tuple(str(item) for item in _require(record, "required_facets")),
                allowed_dois=frozenset(str(item) for item in _require(record, "allowed_dois")),
                insufficient=bool(record.get("insufficient", False)),
                source_paper_id=str(_require(record, "source_paper_id")),
                split=split,
                mismatch_pair_id=record.get("mismatch_pair_id"),
                mismatch_role=record.get("mismatch_role"),
                review_status=str(record.get("review_status", "draft")),
                available_memory_ids=tuple(
                    str(item) for item in record.get("available_memory_ids", [])
                ),
                deterministic_answer_key={
                    str(facet): tuple(str(token) for token in tokens)
                    for facet, tokens in record.get("deterministic_answer_key", {}).items()
                },
                forbidden_answer_tokens=tuple(
                    str(token) for token in record.get("forbidden_answer_tokens", [])
                ),
                required_query_terms=tuple(
                    str(term) for term in record.get("required_query_terms", [])
                ),
                source_paper_ids=tuple(
                    str(item) for item in record.get("source_paper_ids", [])
                ),
                task_type=str(record.get("task_type", "direct_evidence")),
            )
        )
    validate_task_splits(tasks)
    return tuple(tasks)


def load_memories(path: str | Path) -> tuple[StrategyMemory, ...]:
    memories: list[StrategyMemory] = []
    for record in _read_jsonl(path):
        source = _require(record, "source_state")
        memories.append(
            StrategyMemory(
                memory_id=str(_require(record, "memory_id")),
                source_state=SourceState(
                    claim_facets=tuple(str(item) for item in _require(source, "claim_facets")),
                    corpus_scope=str(_require(source, "corpus_scope")),
                    retrieved_document_types=tuple(
                        str(item) for item in source.get("retrieved_document_types", [])
                    ),
                    unresolved_gaps=tuple(str(item) for item in source.get("unresolved_gaps", [])),
                    contradiction_state=str(source.get("contradiction_state", "none")),
                ),
                guidance_kind=GuidanceKind(str(_require(record, "guidance_kind"))),
                instruction=str(_require(record, "instruction")),
            )
        )
    return tuple(memories)


def validate_task_splits(tasks: Iterable[TaskSpec]) -> None:
    paper_splits: dict[str, str] = {}
    pair_splits: dict[str, str] = {}
    for task in tasks:
        source_ids = task.all_source_paper_ids
        if not source_ids and not task.insufficient:
            raise ValueError(f"task {task.task_id} has no source paper")
        for source_paper_id in source_ids:
            previous = paper_splits.setdefault(source_paper_id, task.split)
            if previous != task.split:
                raise ValueError(
                    f"source paper {source_paper_id} leaks across {previous} and {task.split}"
                )
        if task.mismatch_pair_id:
            pair_previous = pair_splits.setdefault(task.mismatch_pair_id, task.split)
            if pair_previous != task.split:
                raise ValueError(
                    f"mismatch pair {task.mismatch_pair_id} leaks across "
                    f"{pair_previous} and {task.split}"
                )


def dump_trajectory(state: EpisodeState) -> dict[str, Any]:
    return {
        "episode_id": state.episode_id,
        "task_id": state.task.task_id,
        "source_paper_id": state.task.source_paper_id,
        "source_paper_ids": list(state.task.all_source_paper_ids),
        "split": state.task.split,
        "task_type": state.task.task_type,
        "steps": [
            {
                "index": step.index,
                "action": {"name": step.action.name, "arguments": step.action.arguments},
                "observation": {
                    "ok": step.observation.ok,
                    "kind": step.observation.kind,
                    "data": step.observation.data,
                    "error": step.observation.error,
                },
                "remaining_budget": step.remaining_budget,
            }
            for step in state.trace
        ],
        "terminal": {
            "done": state.done,
            "abstention_reason": state.abstention_reason,
            "firewall_violations": state.firewall_violations,
        },
    }
