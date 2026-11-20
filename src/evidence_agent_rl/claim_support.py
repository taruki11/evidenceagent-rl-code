"""Deterministic, gold-free claim/evidence support checks.

Motivation
----------
`EvidenceHarness.validate_answer_evidence` only enforces the *firewall*: every cited handle
must exist in the current episode, be bound to the current task, and have been
citation-checked.  It never looks at whether the claim's own content is actually carried by
the span it cites.  Two failures therefore pass the firewall untouched:

* an **unsupported claim** (the sentence is not in the cited span), and
* a **citation swap** (claim A cites span B, claim C cites span A).

The V9 bundle audit (`docs/PAPER_V2_V9_BUNDLE_FAILURE_AUDIT_20260905.md`) recorded both, and
the docscope diagnostic recorded unsupported incidence of 100% on the frozen canary.  This
module supplies the missing, purely lexical check.  It is deliberately conservative:

* it reads **only** public data -- the submitted claim text and the text of the handles the
  claim itself cites.  It never reads ``required_facets``, ``deterministic_answer_key``,
  ``required_query_terms``, ``allowed_dois`` or any other hidden ``TaskSpec`` field, so the
  guard cannot be satisfied by consulting gold;
* it is a *hint generator*, not a scorer.  The harness uses it only to reject an answer with
  a structured repair hint while repair hints remain; the scorer is untouched.

Rules
-----
A claim is locally supported by its cited spans iff

1. every numeric token in the claim appears in the union of the cited span texts, and
2. the share of the claim's content tokens (content words plus numbers) that appear in that
   union is at least ``min_coverage``.

Numbers are checked as an all-or-nothing set because a swapped or hallucinated statistic is
the highest-cost failure in this benchmark.  Content coverage is a threshold because claim
sentences legitimately re-order and lightly paraphrase span text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "claim_support_gaps",
    "content_tokens",
    "numeric_tokens",
]

DEFAULT_MIN_COVERAGE = 0.6

_STOPWORDS = frozenset(
    """
    a an and are as at be been being both but by can could did do does for from
    had has have if in into is it its may might more most no not of on or our over
    such than that the their them then there these they this those to was we were
    what when which while will with would you your also about above after again
    against all am any because before below between down during each few further
    here how itself just once only other out own same she should so some such
    through too under until up very why
    """.split()
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def numeric_tokens(text: str) -> tuple[str, ...]:
    """Return the numeric tokens of ``text`` in first-seen order (deduplicated)."""

    seen: list[str] = []
    for match in _NUMBER_RE.findall(_normalise(text)):
        if match not in seen:
            seen.append(match)
    return tuple(seen)


def content_tokens(text: str) -> tuple[str, ...]:
    """Return the content-bearing tokens of ``text`` (deduplicated, order preserved).

    A token is content-bearing when it is a number, or an alphabetic token of at least four
    characters that is not a stopword.  Short function words are ignored so that a paraphrase
    is not punished for grammatical glue alone.
    """

    normalised = _normalise(text)
    seen: list[str] = []
    for number in numeric_tokens(text):
        seen.append(number)
    for word in _WORD_RE.findall(normalised):
        if len(word) < 4 or word in _STOPWORDS:
            continue
        if word not in seen:
            seen.append(word)
    return tuple(seen)


def _field(item: Any, name: str, default: Any = "") -> Any:
    """Read ``name`` from an attribute-style object or from a raw scoreboard dict."""

    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _span_token_set(texts: Iterable[str]) -> tuple[set[str], set[str]]:
    words: set[str] = set()
    numbers: set[str] = set()
    for text in texts:
        words.update(content_tokens(text))
        numbers.update(numeric_tokens(text))
    return words, numbers


def claim_support_gaps(
    claims: Iterable[Any],
    evidence: dict[str, Any],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[dict[str, Any], ...]:
    """Return one gap record per claim that its own cited spans do not locally support.

    ``claims`` are ``Claim``-like objects (``text`` + ``evidence_handles``); ``evidence`` maps
    a handle id to an ``EvidenceHandle``-like object carrying ``text``.  Handles that do not
    exist in ``evidence`` are skipped here -- the firewall check owns that diagnosis.
    """

    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be within [0, 1]")
    gaps: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        handle_ids = tuple(_field(claim, "evidence_handles", ()) or ())
        cited = [evidence[h] for h in handle_ids if h in evidence]
        if not cited:
            # Nothing local to compare against; the firewall / schema path reports this.
            continue
        claim_text = str(_field(claim, "text", "") or "")
        claim_words = set(content_tokens(claim_text))
        claim_numbers = set(numeric_tokens(claim_text))
        span_words, span_numbers = _span_token_set(
            str(_field(handle, "text", "") or "") for handle in cited
        )
        missing_numbers = sorted(claim_numbers - span_numbers)
        missing_words = sorted(claim_words - span_words)
        total = max(len(claim_words), 1)
        covered = len(claim_words) - len(missing_words)
        coverage = covered / total
        if not missing_numbers and coverage >= min_coverage:
            continue
        gaps.append(
            {
                "claim_index": index,
                "cited_handle_count": len(cited),
                "cited_document_ids": sorted(
                    {str(_field(handle, "document_id", "")) for handle in cited}
                ),
                "content_coverage": round(coverage, 3),
                "min_coverage": min_coverage,
                "missing_numbers": missing_numbers,
                "missing_content_tokens": missing_words[:12],
                "claim_chars": len(claim_text),
            }
        )
    return tuple(gaps)
