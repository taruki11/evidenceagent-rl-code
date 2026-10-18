"""Typed records shared by the harness, skills, and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GuidanceKind(StrEnum):
    QUERY = "QUERY_GUIDANCE"
    VERIFY = "VERIFY_GUIDANCE"
    STOP = "STOP_GUIDANCE"
    EMPTY = "<EMPTY>"


class MemoryMode(StrEnum):
    NONE = "none"
    RAW_REPLAY = "raw_replay"
    QUERY_GUIDANCE = "query_guidance"
    EVIDENCE_BOUND_RECONSTRUCTION = "evidence_bound_reconstruction"


@dataclass(frozen=True)
class SourceState:
    claim_facets: tuple[str, ...]
    corpus_scope: str
    retrieved_document_types: tuple[str, ...] = ()
    unresolved_gaps: tuple[str, ...] = ()
    contradiction_state: str = "none"


@dataclass(frozen=True)
class StrategyMemory:
    """Historical strategy only; it intentionally has no answer/citation fields."""

    memory_id: str
    source_state: SourceState
    guidance_kind: GuidanceKind
    instruction: str

    def __post_init__(self) -> None:
        if self.guidance_kind is GuidanceKind.EMPTY:
            raise ValueError("stored memories must contain a strategy, not <EMPTY>")
        if not self.instruction.strip():
            raise ValueError("memory instruction cannot be empty")


@dataclass(frozen=True)
class Span:
    span_id: str
    text: str
    facets: tuple[str, ...]
    page_number: int | None = None
    page_text_sha256: str = ""
    review_status: str = "draft"


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    doi: str
    corpus_scope: str
    document_type: str
    spans: tuple[Span, ...]
    source_paper_id: str = ""


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    corpus_scope: str
    required_facets: tuple[str, ...]
    allowed_dois: frozenset[str]
    insufficient: bool = False
    source_paper_id: str = ""
    split: str = ""
    mismatch_pair_id: str | None = None
    mismatch_role: str | None = None
    review_status: str = "draft"
    available_memory_ids: tuple[str, ...] | None = None
    deterministic_answer_key: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forbidden_answer_tokens: tuple[str, ...] = ()
    required_query_terms: tuple[str, ...] = ()
    source_paper_ids: tuple[str, ...] = ()
    task_type: str = "direct_evidence"

    @property
    def all_source_paper_ids(self) -> tuple[str, ...]:
        return self.source_paper_ids or ((self.source_paper_id,) if self.source_paper_id else ())


@dataclass
class EvidenceHandle:
    handle_id: str
    episode_id: str
    task_id: str
    document_id: str
    span_id: str
    doi: str
    text: str
    facets: tuple[str, ...]
    checked: bool = False


@dataclass(frozen=True)
class Guidance:
    kind: GuidanceKind
    instruction: str = ""
    memory_id: str | None = None


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_handles: tuple[str, ...]


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    ok: bool
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class TraceStep:
    index: int
    action: Action
    observation: Observation
    remaining_budget: int


@dataclass
class EpisodeState:
    episode_id: str
    task: TaskSpec
    remaining_budget: int
    query_history: list[str] = field(default_factory=list)
    candidate_pool: dict[str, int] = field(default_factory=dict)
    retrieved_documents: set[str] = field(default_factory=set)
    evidence: dict[str, EvidenceHandle] = field(default_factory=dict)
    verification_records: dict[str, bool] = field(default_factory=dict)
    inspected_memory_ids: set[str] = field(default_factory=set)
    guidance_history: list[Guidance] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    final_claims: tuple[Claim, ...] = ()
    abstention_reason: str | None = None
    done: bool = False
    firewall_violations: int = 0

    @property
    def covered_facets(self) -> frozenset[str]:
        return frozenset(
            facet
            for evidence in self.evidence.values()
            if evidence.checked
            for facet in evidence.facets
        )

    @property
    def unresolved_facets(self) -> frozenset[str]:
        return frozenset(self.task.required_facets) - self.covered_facets

    @property
    def coverage_graph(self) -> dict[str, tuple[str, ...]]:
        graph: dict[str, list[str]] = {facet: [] for facet in self.task.required_facets}
        for handle in self.evidence.values():
            if not handle.checked:
                continue
            for facet in handle.facets:
                if facet in graph:
                    graph[facet].append(handle.handle_id)
        return {facet: tuple(handles) for facet, handles in graph.items()}
