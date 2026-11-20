"""A support-count contract derived only from public runtime information.

WHY THIS EXISTS
---------------
The HotpotQA audit (`docs/HOTPOTQA_CONTRACT_MECHANISM_ANALYSIS_20261002.md`) found that the
historical 2/120 -> 25/120 effect is mechanised by one thing: *the declared minimum support
count*.  With a minimum of 1 the policy stops after a single binding (115/120 tasks in
`confirm-base`); with a minimum of 2 it binds two and the second gold sentence is covered
mechanically.  That historical minimum came from a task annotation (`task_type=multi_hop`),
which is a builder-side field, not something a deployed harness can read from the question.

This module derives the same requirement from information a harness legitimately has at
runtime and that contains no gold: the question text and the titles of the candidate
documents it was given.  If the question names at least two distinct candidate documents, the
answer is required to draw on at least two distinct documents; otherwise the fallback minimum
(1) applies.  The rule is deliberately coarse and monotone: it can only *add* a second
binding, never remove one, so it cannot damage an answer that the old harness got right.

Everything here is gold-free by construction: the only inputs are the question string and
document titles.  No answer key, no supporting-fact positions, no hidden ``TaskSpec`` fields.
"""
from __future__ import annotations

import re
from typing import Iterable

DEFAULT_REQUIRED_DOCUMENTS = 2
#: A title is treated as "named by the question" when one of its content tokens (>=4 chars,
#: non-stopword) also occurs in the question.  Only function words are filtered: the rule is a
#: coarse *lower bound* on how many candidate documents the question refers to, and all HotpotQA
#: questions genuinely need two, so over-matching can only ask for the second binding and never
#: removes one.
_TITLE_TOKEN_RE = re.compile(r"[^\W\d_]+")
_MIN_TOKEN_CHARS = 4
_STOPWORDS = frozenset(
    """a an the of in on at to for from by with as is are was were be been being this that
    these those it its their his her he she they them who whom which what when where why how
    did does do done and or not no than then there here about into over under between during
    after before same
    """.split()
)


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase content tokens used for the public match.  Deterministic, locale-free."""

    if not isinstance(text, str):
        return frozenset()
    return frozenset(
        token
        for token in _TITLE_TOKEN_RE.findall(text.lower())
        if len(token) >= _MIN_TOKEN_CHARS and token not in _STOPWORDS
    )


def matched_document_titles(question: str, document_titles: Iterable[str]) -> tuple[str, ...]:
    """Return candidate document titles that the question appears to name.

    Public inputs only (question + corpus titles).  Order follows the input order so a replay
    renders the same list every time; duplicates are dropped.
    """

    question_tokens = content_tokens(question)
    if not question_tokens:
        return ()
    seen: set[str] = set()
    matched: list[str] = []
    for title in document_titles:
        if not isinstance(title, str) or not title.strip():
            continue
        if title in seen:
            continue
        seen.add(title)
        if content_tokens(title) & question_tokens:
            matched.append(title)
    return tuple(matched)


def required_distinct_documents(
    question: str,
    document_titles: Iterable[str],
    *,
    default: int = 1,
    multi_document: int = DEFAULT_REQUIRED_DOCUMENTS,
) -> int:
    """Return the number of distinct documents the answer should draw on.

    ``multi_document`` (2) only when at least that many distinct candidate documents are named
    by the question; otherwise ``default`` (1, i.e. the historical behaviour).
    """

    if default < 1 or multi_document < 1:
        raise ValueError("default and multi_document must be >= 1")
    matched = matched_document_titles(question, document_titles)
    return multi_document if len(matched) >= multi_document else default
