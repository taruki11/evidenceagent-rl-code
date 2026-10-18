"""Small structured skills used by :class:`EvidenceHarness`."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING, Any, Callable

from .hotpotqa_answer import ANSWER_PROTOCOL_ATOMIC, ANSWER_PROTOCOL_SHORT
from .models import Claim, Guidance, GuidanceKind, Observation

if TYPE_CHECKING:
    from .harness import EvidenceHarness


SkillHandler = Callable[["EvidenceHarness", dict[str, Any]], Observation]

_STOPWORDS = {
    "a", "an", "and", "are", "by", "does", "from", "how", "in", "is",
    "of", "on", "the", "to", "what", "with",
}

# ``span_scoring`` selects how span candidates are ranked inside ``_search``.
# ``"overlap"`` is the frozen historical behaviour and MUST stay byte for byte
# identical.  ``"idf_max"`` is the rarity-weighted variant used by the -005
# ranking intervention: a rare query term that appears in only a small fraction
# of spans carries far more weight than a common one, instead of both counting
# as a single match.
#
# ``"idf_max_bridge"`` is the -006 variant.  It adds ONE extra term to
# ``idf_max``: a tie-breaking bonus for carrying the query's own rare
# "bridge" term.  It exists because the -005 counterfactual left gold tied for
# the top score on three of the six clean canary tasks, and the frozen
# tie-break is alphabetical ``span_id`` -- so which co-holder won was decided by
# the alphabet rather than by anything about the text.  The bonus is scaled by
# ``_IDF_BRIDGE_WEIGHT`` (1e-3), which is small enough that it can never
# overturn a strict difference under ``idf_max``; it only separates exact ties.
# It introduces no token the model did not already receive: the bridge term is a
# member of ``task.required_query_terms`` and is already inside the frozen forced
# query byte for byte.
SPAN_SCORING_MODES = ("overlap", "idf_max", "idf_max_bridge")

#: Modes that use the IDF table.
IDF_SPAN_SCORING_MODES = ("idf_max", "idf_max_bridge")

#: Weight on the document-title term in the ``idf_max`` score.  Only IDF
#: *ratios* affect the ordering, so this is robust: sweeping it over
#: {0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0} gives an identical ranking.
_IDF_TITLE_WEIGHT = 0.1

#: Weight on the bridge-term tie-break in ``idf_max_bridge``.  The term is
#: ``~1e-3 * idf``, i.e. at most a few thousandths, while the observed strict
#: gaps between distinct ``idf_max`` scores on the frozen canary are >= 0.22.
#: Verified by test: the extra term never reverses a strict ordering.
_IDF_BRIDGE_WEIGHT = 0.001


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w.-]+", text.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _token_sets(text: str) -> frozenset[str]:
    """Like :func:`_tokens` but returns a hashable frozen set for caching."""
    return frozenset(_tokens(text))


def _idf_table(harness: "EvidenceHarness") -> tuple[dict[str, float], float]:
    """Document frequency over every span of the task's corpus scope.

    Cached on the harness instance because the corpus is immutable for the
    lifetime of an episode.  Returns ``(idf_by_term, n_spans)``.
    """
    cached = getattr(harness, "_idf_cache", None)
    if cached is not None:
        return cached
    document_frequency: dict[str, int] = {}
    span_count = 0
    for document in harness.documents.values():
        if document.corpus_scope != harness.state.task.corpus_scope:
            continue
        for span in document.spans:
            span_count += 1
            for term in _tokens(span.text):
                document_frequency[term] = document_frequency.get(term, 0) + 1
    table = {
        term: math.log((span_count + 1) / (count + 1))
        for term, count in document_frequency.items()
    }
    result = (table, float(span_count))
    harness._idf_cache = result
    return result


def _idf_span_score(
    terms: set[str],
    title_terms: set[str],
    span_terms: set[str],
    idf: dict[str, float],
    bridge_term: str | None = None,
) -> float:
    """``max`` over shared span terms plus a small title term.

    ``max`` replaces the unnormalised count so that span *length* confers no
    bonus: a long span cannot accumulate a high score merely by matching many
    common words.

    When ``bridge_term`` is given (``idf_max_bridge``), a span carrying that term
    gains ``_IDF_BRIDGE_WEIGHT * idf(bridge_term)``.  The term is deliberately
    tiny so it can only separate an exact tie.
    """
    shared = [idf.get(term, 0.0) for term in (terms & span_terms)]
    score = max(shared) if shared else 0.0
    score += _IDF_TITLE_WEIGHT * sum(
        idf.get(term, 0.0) for term in (terms & title_terms)
    )
    if bridge_term is not None and bridge_term in span_terms:
        score += _IDF_BRIDGE_WEIGHT * idf.get(bridge_term, 0.0)
    return round(score, 6)


def _bridge_term(harness: "EvidenceHarness") -> str | None:
    """The single rare discriminator declared by a sequential multi-hop task.

    ``task.required_query_terms`` holds the bridge term that links the two hops;
    on the frozen canary bank it is exactly one term.  Returning it here lets
    ``idf_max_bridge`` use a query signal that is already inside the frozen
    forced query, so no new token reaches the prompt.

    Only the first term is used: the scoring change is declared to be a single
    tie-break, and a longer list would silently become a different scorer.
    """
    terms = getattr(harness.state.task, "required_query_terms", ()) or ()
    for term in terms:
        folded = str(term).casefold()
        if folded:
            return folded
    return None


@dataclass(frozen=True)
class Skill:
    name: str
    cost: int
    handler: SkillHandler
    description: str
    argument_schema: dict[str, Any]

    def public_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "cost": self.cost,
            "parameters": self.argument_schema,
        }

    def validate_arguments(self, arguments: Any) -> str | None:
        return _validate_schema(arguments, self.argument_schema, "arguments")


def validate_arguments_against(schema: dict[str, Any], arguments: Any) -> str | None:
    """Public entry point for schema validation, so a caller holding its own schema can reuse the
    harness's validator instead of reimplementing it."""
    return _validate_schema(arguments, schema, "arguments")


def _validate_schema(value: Any, schema: dict[str, Any], path: str) -> str | None:
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"
        if "enum" in schema and value not in schema["enum"]:
            return f"{path} must be one of {schema['enum']}"
        return None
    if expected_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return f"{path} must contain at least {minimum} items"
        if isinstance(maximum, int) and len(value) > maximum:
            return f"{path} must contain at most {maximum} items"
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                error = _validate_schema(item, item_schema, f"{path}[{index}]")
                if error:
                    return error
        return None
    if expected_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            return f"{path} missing required fields: {sorted(missing)}"
        if schema.get("additionalProperties") is False:
            unexpected = set(value) - set(properties)
            if unexpected:
                return f"{path} has unexpected fields: {sorted(unexpected)}"
        for key, child in value.items():
            if key in properties:
                error = _validate_schema(child, properties[key], f"{path}.{key}")
                if error:
                    return error
        return None
    return None


def _object_schema(
    properties: dict[str, dict[str, Any]], required: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def build_skill_registry() -> dict[str, Skill]:
    return {
        "search": Skill(
            "search", 1, _search, "Search the current corpus scope.",
            _object_schema({"query": {"type": "string"}}, ("query",)),
        ),
        "retrieve": Skill(
            "retrieve", 1, _retrieve, "Retrieve metadata and span IDs for a candidate document.",
            _object_schema({"document_id": {"type": "string"}}, ("document_id",)),
        ),
        "select_evidence": Skill(
            "select_evidence",
            1,
            _select_evidence,
            "Select one search-surfaced span; the harness retrieves, inspects, and citation-checks it atomically.",
            _object_schema(
                {"document_id": {"type": "string"}, "span_id": {"type": "string"}},
                ("document_id", "span_id"),
            ),
        ),
        "inspect_span": Skill(
            "inspect_span", 1, _inspect_span, "Read a span and create a current-episode evidence handle.",
            _object_schema(
                {"document_id": {"type": "string"}, "span_id": {"type": "string"}},
                ("document_id", "span_id"),
            ),
        ),
        "check_citation": Skill(
            "check_citation", 1, _check_citation, "Validate a current evidence handle for citation use.",
            _object_schema({"handle_id": {"type": "string"}}, ("handle_id",)),
        ),
        "inspect_memory": Skill(
            "inspect_memory", 1, _inspect_memory, "Inspect historical strategy and its source state; never evidence.",
            _object_schema({"memory_id": {"type": "string"}}, ("memory_id",)),
        ),
        "reconstruct_memory": Skill(
            "reconstruct_memory", 0, _reconstruct_memory,
            "Submit policy-generated typed guidance or <EMPTY> after inspecting memory.",
            _object_schema(
                {
                    "memory_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["QUERY_GUIDANCE", "VERIFY_GUIDANCE", "STOP_GUIDANCE", "<EMPTY>"],
                    },
                    "instruction": {"type": "string"},
                },
                ("memory_id", "kind"),
            ),
        ),
        "revise_query": Skill(
            "revise_query", 1, _revise_query, "Record a deliberate search-query revision.",
            _object_schema({"query": {"type": "string"}}, ("query",)),
        ),
        "answer": Skill(
            "answer", 0, _answer,
            "Terminate with atomic claim text and every checked current evidence handle "
            "that supports that claim.",
            _object_schema(
                {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "evidence_handles": {
                                    "type": "array", "items": {"type": "string"}
                                },
                            },
                            "required": ["text", "evidence_handles"],
                            "additionalProperties": False,
                        },
                    }
                },
                ("claims",),
            ),
        ),
        "abstain": Skill(
            "abstain", 0, _abstain, "Terminate because current evidence is insufficient or irreconcilable.",
            _object_schema({"reason": {"type": "string"}}, ("reason",)),
        ),
    }


def _required(arguments: dict[str, Any], key: str) -> Any:
    if key not in arguments:
        raise ValueError(f"missing required argument: {key}")
    return arguments[key]


def _search(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    query = str(_required(arguments, "query")).strip()
    if not query:
        raise ValueError("query cannot be empty")
    harness.state.query_history.append(query)
    terms = _tokens(query)
    span_scoring = getattr(harness, "span_scoring", "overlap")
    idf: dict[str, float] | None = None
    bridge_term: str | None = None
    span_term_cache: dict[str, frozenset[str]] = {}
    if span_scoring in IDF_SPAN_SCORING_MODES:
        idf, _ = _idf_table(harness)
    if span_scoring == "idf_max_bridge":
        bridge_term = _bridge_term(harness)
        # The tie-break may only prefer a term the model actually asked for.  A
        # task whose declared bridge term is absent from this query must get a
        # pure no-op rather than a silent boost for an unqueried word.  On the
        # frozen canary bank the term is present on all six clean tasks and
        # absent on both perturbed ones, where the boost would otherwise reorder
        # the frontier for no queried reason.
        if bridge_term is not None and bridge_term not in terms:
            bridge_term = None
    ranked: list[tuple[int, str]] = []
    ranked_spans: list[tuple[int, str, str]] = []
    # Optional document scope for the follow-up search of a sequential multi-hop
    # episode.  ``None`` keeps the historical whole-corpus behaviour byte for byte;
    # scoring, stopwords, top-k and tie-breaks are untouched either way.
    scoped_document_id = harness.followup_search_document_id()
    scanned_document_ids: list[str] = []
    for document in harness.documents.values():
        if document.corpus_scope != harness.state.task.corpus_scope:
            continue
        if scoped_document_id is not None and document.document_id != scoped_document_id:
            continue
        scanned_document_ids.append(document.document_id)
        title_terms = _tokens(document.title)
        body_terms = _tokens(" ".join(span.text for span in document.spans))
        score = 3 * len(terms & title_terms) + len(terms & body_terms)
        if score:
            ranked.append((score, document.document_id))
        for span in document.spans:
            if idf is None:
                span_score = 3 * len(terms & _tokens(span.text)) + len(
                    terms & title_terms
                )
            else:
                span_key = f"{document.document_id}:{span.span_id}"
                span_terms = span_term_cache.get(span_key)
                if span_terms is None:
                    span_terms = _token_sets(span.text)
                    span_term_cache[span_key] = span_terms
                span_score = _idf_span_score(
                    terms, title_terms, span_terms, idf, bridge_term
                )
            if span_score:
                ranked_spans.append((span_score, document.document_id, span.span_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    ranked_spans.sort(key=lambda item: (-item[0], item[1], item[2]))
    for score, document_id in ranked:
        previous = harness.state.candidate_pool.get(document_id, 0)
        harness.state.candidate_pool[document_id] = max(previous, score)
    visible_spans = ranked_spans[:24]
    if getattr(harness, "require_distinct_support_documents", False) and hasattr(
        harness, "state"
    ):
        covered = {h.document_id for h in harness.state.evidence.values()}
        if len(covered) == 1:
            only = next(iter(covered))
            without = [item for item in visible_spans if item[1] != only]
            if without:
                visible_spans = without
    if harness.policy_context_version == "v6":
        harness.public_span_candidates = {
            (document_id, span_id) for _, document_id, span_id in visible_spans
        }
        harness.public_span_metadata = {
            (document_id, span_id): {
                "score": score,
                "title": harness.documents[document_id].title,
                "text_preview": _public_text_preview(
                    next(
                        span.text
                        for span in harness.documents[document_id].spans
                        if span.span_id == span_id
                    )
                ),
            }
            for score, document_id, span_id in visible_spans
        }
    else:
        harness.public_span_candidates.update(
            (document_id, span_id) for _, document_id, span_id in visible_spans
        )
    span_candidates = [
        {
            "document_id": document_id,
            "span_id": span_id,
            "score": score,
            **(
                harness.public_span_metadata[(document_id, span_id)]
                if harness.policy_context_version == "v6"
                else {}
            ),
        }
        for score, document_id, span_id in visible_spans
    ]
    data: dict[str, Any] = {
        "document_ids": [item[1] for item in ranked],
        "span_candidates": span_candidates,
    }
    if harness.document_scoped_followup_search:
        # Instrument for the REV3 gate ``docscope_applied``: record the documents this
        # search actually scanned, so "the scope filter did not take effect" can be
        # distinguished from "the policy picked the wrong paper".  Emitted only when the
        # feature is enabled, keeping the default payload unchanged.
        data["scoped_search"] = {
            "enabled": True,
            "scoped_document_id": scoped_document_id,
            "scanned_document_ids": sorted(scanned_document_ids),
        }
    return Observation(True, "search_results", data)


def _public_text_preview(text: str, *, max_characters: int = 192) -> str:
    """Return bounded public candidate text without creating evidence."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_characters:
        return normalized
    return normalized[: max_characters - 1].rstrip() + "…"


def _retrieve(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    document_id = str(_required(arguments, "document_id"))
    document = harness.documents.get(document_id)
    if document is None or document.corpus_scope != harness.state.task.corpus_scope:
        return Observation(False, "retrieve", error="document is unavailable in this corpus scope")
    if document_id not in harness.state.candidate_pool:
        return Observation(False, "retrieve", error="search must place the document in the candidate pool")
    harness.state.retrieved_documents.add(document_id)
    return Observation(
        True,
        "document",
        {
            "document_id": document.document_id,
            "title": document.title,
            "doi": document.doi,
            "document_type": document.document_type,
            "span_ids": [span.span_id for span in document.spans],
        },
    )


def _select_evidence(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    """Collapse retrieve/inspect/check without weakening the evidence firewall."""

    document_id = str(_required(arguments, "document_id"))
    span_id = str(_required(arguments, "span_id"))
    if (document_id, span_id) not in harness.public_span_candidates:
        return Observation(
            False,
            "select_evidence",
            error="select only a span surfaced by a current-episode search",
        )
    document = harness.documents.get(document_id)
    if document is None or document.corpus_scope != harness.state.task.corpus_scope:
        return Observation(False, "select_evidence", error="span is unavailable in this corpus scope")
    span = next((item for item in document.spans if item.span_id == span_id), None)
    if span is None:
        return Observation(False, "select_evidence", error="unknown span")

    if getattr(harness, "require_distinct_support_documents", False):
        covered = {h.document_id for h in harness.state.evidence.values()}
        if covered and covered == {document_id}:
            return Observation(
                False, "select_evidence",
                error=("the supporting evidence must come from two different documents: "
                       "select from another paragraph first"),
            )
    harness.state.retrieved_documents.add(document_id)
    handle = harness.make_evidence_handle(document, span)
    harness.state.evidence[handle.handle_id] = handle
    valid = handle.doi in harness.state.task.allowed_dois
    handle.checked = valid
    harness.state.verification_records[handle.handle_id] = valid
    return Observation(
        valid,
        "selected_evidence",
        {
            "handle_id": handle.handle_id,
            "document_id": document_id,
            "span_id": span_id,
            "text": span.text,
            "doi": document.doi,
            "citation_checked": valid,
        },
        None if valid else "DOI is not allowed for this task",
    )


def _inspect_span(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    document_id = str(_required(arguments, "document_id"))
    span_id = str(_required(arguments, "span_id"))
    if document_id not in harness.state.retrieved_documents:
        return Observation(False, "inspect_span", error="retrieve the document before inspecting spans")
    document = harness.documents[document_id]
    span = next((item for item in document.spans if item.span_id == span_id), None)
    if span is None:
        return Observation(False, "inspect_span", error="unknown span")
    handle = harness.make_evidence_handle(document, span)
    harness.state.evidence[handle.handle_id] = handle
    return Observation(
        True,
        "evidence",
        {"handle_id": handle.handle_id, "text": handle.text},
    )


def _check_citation(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    handle_id = str(_required(arguments, "handle_id"))
    handle = harness.state.evidence.get(handle_id)
    if handle is None:
        harness.state.firewall_violations += 1
        return Observation(False, "citation_check", error="evidence handle is not from this episode")
    valid = handle.doi in harness.state.task.allowed_dois
    handle.checked = valid
    harness.state.verification_records[handle_id] = valid
    return Observation(
        valid,
        "citation_check",
        {"handle_id": handle_id, "doi": handle.doi, "checked": valid},
        None if valid else "DOI is not allowed for this task",
    )


def _inspect_memory(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    memory_id = str(_required(arguments, "memory_id"))
    return harness.expose_memory(memory_id)


def _reconstruct_memory(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    memory_id = str(_required(arguments, "memory_id"))
    if memory_id not in harness.state.inspected_memory_ids:
        return Observation(False, "memory_guidance", error="inspect memory before reconstruction")
    if any(item.memory_id == memory_id for item in harness.state.guidance_history):
        return Observation(False, "memory_guidance", error="memory was already reconstructed")
    try:
        kind = GuidanceKind(str(_required(arguments, "kind")))
    except ValueError:
        return Observation(False, "memory_guidance", error="invalid guidance kind")
    instruction = str(arguments.get("instruction", "")).strip()
    if kind is GuidanceKind.EMPTY and instruction:
        return Observation(False, "memory_guidance", error="<EMPTY> cannot contain an instruction")
    if kind is not GuidanceKind.EMPTY and not instruction:
        return Observation(False, "memory_guidance", error="non-empty guidance requires an instruction")
    guidance = Guidance(kind, instruction, memory_id)
    harness.state.guidance_history.append(guidance)
    return Observation(
        True,
        "memory_guidance",
        {"kind": guidance.kind.value, "instruction": guidance.instruction},
    )


def _revise_query(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    query = str(_required(arguments, "query")).strip()
    if not query:
        raise ValueError("query cannot be empty")
    harness.state.query_history.append(query)
    return Observation(True, "query_revision", {"query": query})


def _answer(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    if getattr(harness, "answer_protocol", ANSWER_PROTOCOL_ATOMIC) == ANSWER_PROTOCOL_SHORT:
        return _answer_short_with_support(harness, arguments)
    claims = harness.parse_claims(_required(arguments, "claims"))
    if harness.policy_context_version in {"v5", "v6"}:
        cited_handles = [
            handle_id
            for claim in claims
            for handle_id in claim.evidence_handles
        ]
        if any(len(claim.evidence_handles) != 1 for claim in claims):
            harness.state.done = True
            harness.state.firewall_violations += 1
            return Observation(
                False,
                "answer",
                error="v5 atomic claims must bind exactly one evidence handle",
            )
        if len(set(cited_handles)) < harness.minimum_answer_evidence_handles():
            harness.state.done = True
            harness.state.firewall_violations += 1
            return Observation(
                False,
                "answer",
                error="v5 answer lacks distinct checked evidence required by the task",
            )
    error = harness.validate_answer_evidence(claims)
    if error:
        if harness.policy_context_version in {"v5", "v6"}:
            harness.state.done = True
        harness.state.firewall_violations += 1
        return Observation(False, "answer", error=error)
    harness.state.final_claims = claims
    harness.state.done = True
    return Observation(True, "answer", {"claim_count": len(claims)})


def _answer_short_with_support(
    harness: "EvidenceHarness", arguments: dict[str, Any]
) -> Observation:
    """HotpotQA answer protocol: a short answer string plus the supporting handles it submits.

    The submitted set is used verbatim.  The evidence ledger is NOT auto-cited, the answer is NOT
    truncated or rewritten, and the number of citations is NOT derived from gold.  The current-
    episode firewall is preserved unchanged: every handle must exist in this episode, be bound to
    this task, and have been citation-checked.
    """
    raw_answer = _required(arguments, "answer")
    if not isinstance(raw_answer, str):
        harness.state.done = True
        return Observation(False, "answer", error="answer must be a string")
    raw_handles = _required(arguments, "supporting_handles")
    if not isinstance(raw_handles, list):
        harness.state.done = True
        return Observation(False, "answer", error="supporting_handles must be a list")
    handles: list[str] = []
    for item in raw_handles:
        if not isinstance(item, str) or not item:
            harness.state.done = True
            return Observation(
                False, "answer", error="supporting_handles must be non-empty strings"
            )
        if item not in handles:  # deduplicate, preserve submission order
            handles.append(item)
    claim = Claim(text=raw_answer.strip(), evidence_handles=tuple(handles))
    error = harness.validate_answer_evidence((claim,))
    if error:
        harness.state.done = True
        harness.state.firewall_violations += 1
        return Observation(False, "answer", error=error)
    if getattr(harness, "require_distinct_support_documents", False) and handles:
        documents = {
            harness.state.evidence[h].document_id
            for h in handles
            if h in harness.state.evidence
        }
        if len(documents) < 2:
            harness.state.done = True
            return Observation(
                False, "answer",
                error="the submitted support spans must come from two different documents",
            )
    harness.state.final_claims = (claim,)
    harness.state.done = True
    return Observation(
        True,
        "answer",
        {
            "claim_count": 1,
            "answer_chars": len(claim.text),
            "submitted_handles": len(handles),
            "protocol": ANSWER_PROTOCOL_SHORT,
        },
    )


def _abstain(harness: "EvidenceHarness", arguments: dict[str, Any]) -> Observation:
    reason = str(_required(arguments, "reason")).strip()
    if not reason:
        raise ValueError("abstention reason cannot be empty")
    harness.state.abstention_reason = reason
    harness.state.done = True
    return Observation(True, "abstention", {"reason": reason})
