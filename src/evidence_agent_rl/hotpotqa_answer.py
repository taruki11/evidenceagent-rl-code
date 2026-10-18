"""Thinnest HotpotQA answer adaptation: separate the short answer from the supporting set.

WHY THIS EXISTS
---------------
The Paper-v2/v6 answer action forces every conclusion into *atomic claims*, each bound to exactly
one evidence handle, and the answer text is the concatenation of those claim texts.  On HotpotQA
that conflates two different things -- the answer string and the citation set -- so a policy that
knows the answer but wants to support it with three sentences must either emit three claim texts
(producing a long, unscoreable answer) or under-cite.  The old diagnostic could therefore not tell
"the model cannot answer" apart from "the output protocol cannot express the answer".

This module adds a second, explicitly named protocol:

    atomic_claims               the historical behaviour; default; byte-for-byte unchanged
    short_answer_with_support   `answer` is the model's own short answer string and
                                `supporting_handles` is the set of checked handles it submits

Rules honoured here (all of them are load-bearing):

  * the supporting set is exactly what the model submitted -- the evidence ledger is never
    auto-cited, and nothing is inferred from gold;
  * the model may submit one handle, several, or none; the protocol never forces a multi-evidence
    conclusion into a single sentence and never rewrites the answer;
  * gold is used only to *choose actions* in an oracle, never to fill a prediction;
  * the prediction the scorer sees always comes from an actual action and the actual final state,
    through `extract_prediction` below -- the same function for model arms and for the oracle.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

ANSWER_PROTOCOL_ATOMIC = "atomic_claims"
ANSWER_PROTOCOL_SHORT = "short_answer_with_support"
ANSWER_PROTOCOLS: tuple[str, ...] = (ANSWER_PROTOCOL_ATOMIC, ANSWER_PROTOCOL_SHORT)


def short_answer_argument_schema(min_items: int = 1) -> dict[str, Any]:
    """The single definition of the short-answer action schema.

    Used both for the public skill catalog and for runtime argument validation, so the model's
    prompt and the harness's check can never drift apart.

    `min_items` is the task contract's declared minimum support count.  At the default of 1 no
    `minItems` is emitted at all, so the behaviour of every existing run is unchanged; at 2 or more it
    is rendered AND enforced, exactly as `minimum_answer_evidence_handles()` already did for the
    historical atomic-claims path.
    """
    handles: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    if min_items > 1:
        handles["minItems"] = min_items
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "supporting_handles": handles,
        },
        "required": ["answer", "supporting_handles"],
        "additionalProperties": False,
    }


def normalise_support_keys(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deduplicate (title, sent_id) pairs, preserving first-seen order. No gold, no filtering."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for title, sent_id in pairs:
        key = (str(title), str(sent_id))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def to_official_sp(pairs: Iterable[tuple[str, str]]) -> list[list[Any]]:
    """Render (title, sent_id) pairs the way the official scorer expects: [title, int]."""
    rendered: list[list[Any]] = []
    for title, sent_id in normalise_support_keys(pairs):
        rendered.append([title, int(sent_id) if str(sent_id).lstrip("-").isdigit() else sent_id])
    return rendered


def extract_prediction(
    state: Any,
    *,
    protocol: str | None = None,
    title_by_document: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The ONE extraction shared by every model arm and by the oracle.

    Reads only the episode's own final state.  For `short_answer_with_support` the answer is the
    single submitted answer string and the support is exactly the submitted handle set.  For
    `atomic_claims` the historical reconstruction is used, so old numbers stay reproducible.

    `protocol` is the harness's declared answer protocol; when omitted it is read from the state
    (model arms pass it explicitly, since the protocol lives on the harness, not the episode).
    """
    protocol = protocol or getattr(state, "answer_protocol", ANSWER_PROTOCOL_ATOMIC)
    claims = tuple(getattr(state, "final_claims", ()) or ())
    evidence = getattr(state, "evidence", {}) or {}
    title_map = title_by_document or {}

    if protocol == ANSWER_PROTOCOL_SHORT:
        answer = claims[0].text if claims else ""
        submitted: list[str] = []
        for claim in claims:
            submitted.extend(str(handle_id) for handle_id in claim.evidence_handles)
    else:
        answer = " ".join(dict.fromkeys(claim.text for claim in claims)) if claims else ""
        submitted = [str(handle_id) for claim in claims for handle_id in claim.evidence_handles]

    keys: list[tuple[str, str]] = []
    unresolved: list[str] = []
    seen_handles: set[str] = set()
    for handle_id in submitted:
        if handle_id in seen_handles:
            continue
        seen_handles.add(handle_id)
        handle = evidence.get(handle_id)
        if handle is None:
            unresolved.append(handle_id)
            continue
        span_id = str(handle.span_id)
        sent_id = span_id.split("|")[-1] if "|" in span_id else span_id
        title = title_map.get(str(handle.document_id), str(handle.document_id))
        keys.append((title, sent_id))

    support = normalise_support_keys(keys)
    return {
        "protocol": protocol,
        "answer": answer,
        "support_keys": support,
        "support_official": to_official_sp(support),
        "cited_handle_ids": [h for h in seen_handles],
        "unresolved_handle_ids": unresolved,
        "answer_chars": len(answer),
        "n_support": len(support),
    }


def title_map_from_gold(gold_row: Mapping[str, Any]) -> dict[str, str]:
    """document_id -> the SOURCE paragraph title, as the official scorer expects it.

    Display titles carry a `#k` suffix when a question repeats a title; the official scorer must be
    given the source's own title, so this maps back through the recorded raw titles.
    """
    ids = list(gold_row.get("candidate_document_ids") or [])
    raw = list(gold_row.get("candidate_raw_titles") or [])
    return {str(d): str(t) for d, t in zip(ids, raw)}
