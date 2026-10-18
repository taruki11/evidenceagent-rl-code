"""Deterministic, provenance-first proposals for automatically built train tasks."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from .curation import PageRecord, canonicalize_text
from .io import validate_task_splits
from .models import Document, Span, TaskSpec


GENERATOR_VERSION = "evidence_proposal_v1"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_NUMBER = re.compile(
    r"(?<![\w.-])(?:≈|~)?(?:\d+(?:\.\d+)?\s*×\s*10\s*\d+|"
    r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*±\s*\d+(?:\.\d+)?)?(?:%|×|x)?(?!\w)"
)
_WORD = re.compile(r"[A-Za-z][A-Za-z-]{3,}")
_CITATION = re.compile(r"\[\s*(?:\d+[,-]?\s*)+\]")
_SECTION_REFERENCE = re.compile(
    r"(?:Sections?|§)\s*\d+(?:\.\d+)?(?:\s*[,–-]\s*\d+(?:\.\d+)?)*",
    flags=re.IGNORECASE,
)
_RESULT_TERMS = {
    "achieve", "achieved", "accuracy", "baseline", "compared", "evaluation",
    "improve", "improved", "outperform", "outperformed", "report", "reported",
    "result", "results", "score", "success", "training", "tasks", "dataset",
}
_QUALITATIVE_CLAIM_TERMS = {
    "achieve", "achieved", "improve", "improved", "outperform", "outperformed",
    "reduce", "reduced", "reduces", "increase", "increased", "increases",
    "enable", "enabled", "enables", "demonstrate", "demonstrated", "demonstrates",
}
_DEFINITION_PATTERN = re.compile(
    r"\b(?:we\s+(?:introduce|propose|present|develop|define)|"
    r"is\s+defined\s+as|refers\s+to|consists\s+of)\b",
    flags=re.IGNORECASE,
)
_FINDING_PATTERN = re.compile(
    r"\b(?:we\s+(?:show|find|demonstrate|observe)|"
    r"our\s+(?:results|experiments|evaluation)\s+(?:show|shows|demonstrate|demonstrates))\b",
    flags=re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(
    r"\b(?:in\s+contrast|unlike|compared\s+(?:to|with)|whereas)\b",
    flags=re.IGNORECASE,
)
_MECHANISM_PATTERN = re.compile(
    r"\b(?:(?:our|the|this)\s+)?(?:method|approach|framework|system|agent|model|"
    r"pipeline|algorithm)\s+(?:uses|combines|maintains|tracks|selects|retrieves|"
    r"constructs|updates|optimizes|enables|operates|performs)\b",
    flags=re.IGNORECASE,
)
_STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "before",
    "being", "between", "both", "could", "does", "during", "each", "from", "have",
    "into", "more", "most", "other", "over", "paper", "reported", "result", "results",
    "show", "shows", "such", "than", "that", "their", "these", "they", "this", "those",
    "through", "under", "using", "were", "which", "while", "with", "would",
    "accordingly", "crucially", "markedly", "only",
    "introduce", "introduced", "propose", "proposed", "present", "presented",
    "develop", "developed", "define", "defined",
}


@dataclass(frozen=True)
class AutoTrainingCandidate:
    candidate_id: str
    generator_version: str
    source_paper_id: str
    document_id: str
    page_number: int
    page_text_sha256: str
    span_text: str
    facet: str
    question: str
    answer_tokens: tuple[str, ...]
    topic_terms: tuple[str, ...]
    task_rule: str
    quality_score: float
    split: str = "train"

    def to_json(self) -> dict[str, object]:
        raw = asdict(self)
        raw["answer_tokens"] = list(self.answer_tokens)
        raw["topic_terms"] = list(self.topic_terms)
        return raw


@dataclass(frozen=True)
class AutoGenerationReport:
    generator_version: str
    requested: int
    generated: int
    source_papers: int
    per_paper: dict[str, int]
    per_rule: dict[str, int]
    rejected_existing: int
    rejected_near_duplicate: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _SentenceProposal:
    source_paper_id: str
    document_id: str
    title: str
    page_number: int
    page_text_sha256: str
    text: str
    answer_tokens: tuple[str, ...]
    words: tuple[str, ...]
    result_term_count: int
    task_rule: str


def _meaningful_numbers(text: str) -> tuple[str, ...]:
    without_citations = _SECTION_REFERENCE.sub("", _CITATION.sub("", text))
    values: list[str] = []
    for match in _NUMBER.finditer(without_citations):
        if without_citations[match.end():].lstrip().startswith(":"):
            continue
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        plain = value.lstrip("≈~").replace(",", "")
        numeric = re.match(r"\d+(?:\.\d+)?", plain)
        if not numeric:
            continue
        number = float(numeric.group(0))
        is_year = value.isdigit() and 1900 <= number <= 2099
        distinctive = any(symbol in value for symbol in ("%", ".", ",", "×", "x", "±", "≈", "~"))
        if is_year or (not distinctive and number < 10):
            continue
        if value not in values:
            values.append(value)
    return tuple(values[:3])


def _content_words(text: str) -> tuple[str, ...]:
    topic_text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    return tuple(
        word.casefold().strip("-")
        for word in _WORD.findall(topic_text)
        if word.casefold().strip("-") not in _STOPWORDS
        and len(word.casefold().strip("-")) >= 5
    )


def _eligible_sentence(text: str) -> bool:
    if not 100 <= len(text) <= 450:
        return False
    lowered = text.casefold()
    if any(
        marker in lowered
        for marker in (
            "arxiv:", "http://", "https://", "references", "et al.",
            "doi:", "journal of", "proceedings of", "springer", "publisher", "isbn",
            "figure ", "table ",
        )
    ):
        return False
    if re.match(r"^\s*\d+(?:\.\d+)+\s+[A-Z]", text):
        return False
    if re.search(r"(?:^|\s)\d+\.\d+\s+(?:[A-Z][A-Za-z-]+\s+){1,4}", text):
        return False
    if re.match(r"^\s*\d+\s*\)?\s*:", text):
        return False
    if re.search(r"\b\d+\s*:\s*\d+\s*[–-]\s*\d+\b", text):
        return False
    if re.search(r"[a-z][.!?][A-Z]", text) or re.search(r"[A-Za-z]{25,}", text):
        return False
    letters = sum(character.isalpha() for character in text)
    if letters / len(text) < 0.55:
        return False
    return len(_content_words(text)) >= 6


def _masked_evidence_question(proposal: _SentenceProposal) -> str:
    """Render a source-grounded cloze question without exposing answer anchors."""

    masked = proposal.text
    for token in sorted(proposal.answer_tokens, key=lambda value: (-len(value), value)):
        masked = re.sub(re.escape(token), "[BLANK]", masked, flags=re.IGNORECASE)
    question = (
        "Complete the masked evidence statement using the current paper and cite "
        f"the evidence span: {masked}"
    )
    if any(token.casefold() in question.casefold() for token in proposal.answer_tokens):
        raise ValueError("answer anchor remains visible in generated question")
    return question


def _jaccard(left: str, right: str) -> float:
    left_words = set(_content_words(left))
    right_words = set(_content_words(right))
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 1.0


def generate_auto_training_candidates(
    pages: Iterable[PageRecord],
    *,
    corpus_manifest: dict[str, dict[str, object]],
    split_manifest: dict[str, str],
    target: int = 360,
    existing_span_texts: Iterable[str] = (),
    existing_questions: Iterable[str] = (),
    max_per_paper: int = 35,
    max_per_page: int = 4,
    split: str = "train",
) -> tuple[tuple[AutoTrainingCandidate, ...], AutoGenerationReport]:
    if target <= 0 or max_per_paper <= 0 or max_per_page <= 0:
        raise ValueError("target, max_per_paper, and max_per_page must be positive")
    if split not in {"train", "verifierbench", "dev", "test"}:
        raise ValueError("automatic proposal split must be train, verifierbench, dev, or test")
    page_list = tuple(pages)
    existing_spans = {canonicalize_text(text).casefold() for text in existing_span_texts}
    existing_question_set = {
        canonicalize_text(question).casefold() for question in existing_questions
    }
    proposals: list[_SentenceProposal] = []
    rejected_existing = 0
    reference_start: dict[str, int] = {}
    for page in page_list:
        source_paper_id = page.document_id.removeprefix("arxiv:")
        if split_manifest.get(source_paper_id) != split:
            continue
        if re.search(
            r"(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)",
            page.text,
            flags=re.IGNORECASE,
        ):
            reference_start.setdefault(source_paper_id, page.page_number)
    for page in page_list:
        source_paper_id = page.document_id.removeprefix("arxiv:")
        if split_manifest.get(source_paper_id) != split:
            continue
        if page.page_number >= reference_start.get(source_paper_id, 10**9):
            continue
        manifest = corpus_manifest.get(source_paper_id)
        if manifest is None:
            raise ValueError(f"page references unknown source paper: {source_paper_id}")
        title = str(manifest["title"])
        normalized_page = canonicalize_text(page.text)
        for sentence in _SENTENCE_SPLIT.split(normalized_page):
            sentence = sentence.strip()
            if not _eligible_sentence(sentence):
                continue
            if sentence.casefold() in existing_spans:
                rejected_existing += 1
                continue
            words = _content_words(sentence)
            answer_tokens = _meaningful_numbers(sentence)
            if any(
                len(re.findall(re.escape(token), sentence, flags=re.IGNORECASE)) != 1
                for token in answer_tokens
            ):
                continue
            result_term_count = sum(word in _RESULT_TERMS for word in words)
            if answer_tokens:
                if result_term_count == 0 and len(answer_tokens) < 2:
                    continue
                task_rule = "quantitative"
            elif _DEFINITION_PATTERN.search(sentence):
                task_rule = "definition"
            elif _FINDING_PATTERN.search(sentence):
                task_rule = "finding"
            elif _COMPARISON_PATTERN.search(sentence):
                task_rule = "comparison"
            elif _MECHANISM_PATTERN.search(sentence):
                task_rule = "mechanism"
            elif any(word in _QUALITATIVE_CLAIM_TERMS for word in words):
                task_rule = "finding"
            else:
                continue
            proposals.append(
                _SentenceProposal(
                    source_paper_id,
                    page.document_id,
                    title,
                    page.page_number,
                    page.text_sha256,
                    sentence,
                    answer_tokens,
                    words,
                    result_term_count,
                    task_rule,
                )
            )

    document_frequency: Counter[str] = Counter()
    for proposal in proposals:
        document_frequency.update(set(proposal.words))
    total_proposals = max(1, len(proposals))
    by_paper_page: dict[
        str, dict[int, list[tuple[float, _SentenceProposal, tuple[str, ...]]]]
    ] = defaultdict(lambda: defaultdict(list))
    for proposal in proposals:
        positions: dict[str, int] = {}
        for index, word in enumerate(proposal.words):
            positions.setdefault(word, index)
        ranked_terms = sorted(
            set(proposal.words),
            key=lambda word: (positions[word], word),
        )
        topic_terms = tuple(ranked_terms[:3])
        if len(topic_terms) < 3:
            continue
        rarity = sum(
            math.log((1 + total_proposals) / (1 + document_frequency[word]))
            for word in topic_terms
        )
        if proposal.task_rule == "quantitative":
            answer_tokens = proposal.answer_tokens
            distinctive_numbers = sum(
                any(symbol in value for symbol in ("%", ".", ",", "×", "x", "±", "≈"))
                for value in answer_tokens
            )
            score = (
                100
                + 4 * len(answer_tokens)
                + 2 * distinctive_numbers
                + proposal.result_term_count
                + rarity
            )
        else:
            rare_terms = sorted(
                {
                    word
                    for word in set(proposal.words) - set(topic_terms)
                    if word in proposal.text.casefold()
                },
                key=lambda word: (
                    -math.log((1 + total_proposals) / (1 + document_frequency[word])),
                    positions[word],
                    word,
                ),
            )
            answer_tokens = tuple(rare_terms[:2])
            if len(answer_tokens) < 2:
                continue
            qualitative_base = {
                "definition": 8,
                "finding": 7,
                "comparison": 7,
                "mechanism": 7,
            }[proposal.task_rule]
            score = qualitative_base + rarity + sum(
                math.log((1 + total_proposals) / (1 + document_frequency[word]))
                for word in answer_tokens
            )
        if any(
            len(re.findall(re.escape(token), proposal.text, flags=re.IGNORECASE)) != 1
            for token in answer_tokens
        ):
            continue
        ranked = (score, replace(proposal, answer_tokens=answer_tokens), topic_terms)
        by_paper_page[proposal.source_paper_id][proposal.page_number].append(ranked)

    queues: dict[str, list[tuple[float, _SentenceProposal, tuple[str, ...]]]] = {}
    for paper_id, page_candidates in by_paper_page.items():
        diversified = [
            item
            for page_number in sorted(page_candidates)
            for item in sorted(
                page_candidates[page_number],
                key=lambda candidate: (-candidate[0], candidate[1].text),
            )[:max_per_page]
        ]
        queues[paper_id] = sorted(
            diversified, key=lambda item: (-item[0], item[1].page_number, item[1].text)
        )[:max_per_paper]

    selected: list[AutoTrainingCandidate] = []
    selected_texts: list[str] = []
    selected_questions: set[str] = set(existing_question_set)
    rejected_near_duplicate = 0
    paper_ids = sorted(queues)
    depth = 0
    while len(selected) < target and any(depth < len(queues[paper]) for paper in paper_ids):
        for paper_id in paper_ids:
            if len(selected) >= target or depth >= len(queues[paper_id]):
                continue
            score, proposal, topic_terms = queues[paper_id][depth]
            if any(_jaccard(proposal.text, previous) >= 0.82 for previous in selected_texts):
                rejected_near_duplicate += 1
                continue
            try:
                question = _masked_evidence_question(proposal)
            except ValueError:
                rejected_near_duplicate += 1
                continue
            normalized_question = canonicalize_text(question).casefold()
            if normalized_question in selected_questions:
                rejected_near_duplicate += 1
                continue
            digest = hashlib.sha256(
                f"{paper_id}\0{proposal.page_number}\0{proposal.text}".encode("utf-8")
            ).hexdigest()[:16]
            candidate_id = f"auto-{paper_id}-{digest}"
            selected.append(
                AutoTrainingCandidate(
                    candidate_id=candidate_id,
                    generator_version=GENERATOR_VERSION,
                    source_paper_id=paper_id,
                    document_id=proposal.document_id,
                    page_number=proposal.page_number,
                    page_text_sha256=proposal.page_text_sha256,
                    span_text=proposal.text,
                    facet=f"auto-facet-{digest}",
                    question=question,
                    answer_tokens=proposal.answer_tokens,
                    topic_terms=topic_terms,
                    task_rule=proposal.task_rule,
                    quality_score=round(score, 6),
                    split=split,
                )
            )
            selected_texts.append(proposal.text)
            selected_questions.add(normalized_question)
        depth += 1

    validate_auto_training_candidates(
        selected,
        pages=page_list,
        corpus_manifest=corpus_manifest,
        split_manifest=split_manifest,
        max_per_page=max_per_page,
    )
    per_paper = dict(Counter(item.source_paper_id for item in selected))
    report = AutoGenerationReport(
        generator_version=GENERATOR_VERSION,
        requested=target,
        generated=len(selected),
        source_papers=len(per_paper),
        per_paper=dict(sorted(per_paper.items())),
        per_rule=dict(sorted(Counter(item.task_rule for item in selected).items())),
        rejected_existing=rejected_existing,
        rejected_near_duplicate=rejected_near_duplicate,
    )
    return tuple(selected), report


def validate_auto_training_candidates(
    candidates: Iterable[AutoTrainingCandidate],
    *,
    pages: Iterable[PageRecord],
    corpus_manifest: dict[str, dict[str, object]],
    split_manifest: dict[str, str],
    max_per_page: int = 4,
) -> None:
    items = tuple(candidates)
    if not items:
        raise ValueError("auto training candidates cannot be empty")
    if len({item.candidate_id for item in items}) != len(items):
        raise ValueError("auto training candidate IDs must be unique")
    if len({canonicalize_text(item.question).casefold() for item in items}) != len(items):
        raise ValueError("auto training questions must be unique")
    page_index = {(page.document_id, page.page_number): page for page in pages}
    page_counts: Counter[tuple[str, int]] = Counter()
    seen_texts: list[str] = []
    for item in items:
        if (
            item.generator_version != GENERATOR_VERSION
            or item.split not in {"train", "verifierbench", "dev", "test"}
            or item.task_rule
            not in {"quantitative", "definition", "finding", "comparison", "mechanism"}
        ):
            raise ValueError(f"{item.candidate_id}: invalid generator version or split")
        if split_manifest.get(item.source_paper_id) != item.split:
            raise ValueError(
                f"{item.candidate_id}: source paper is not assigned to {item.split}"
            )
        manifest = corpus_manifest.get(item.source_paper_id)
        if manifest is None or str(manifest["document_id"]) != item.document_id:
            raise ValueError(f"{item.candidate_id}: corpus identity mismatch")
        page_key = (item.document_id, item.page_number)
        page_counts[page_key] += 1
        if page_counts[page_key] > max_per_page:
            raise ValueError(f"{item.candidate_id}: page candidate limit exceeded")
        page = page_index.get(page_key)
        if page is None or page.text_sha256 != item.page_text_sha256:
            raise ValueError(f"{item.candidate_id}: page provenance mismatch")
        if canonicalize_text(item.span_text) not in canonicalize_text(page.text):
            raise ValueError(f"{item.candidate_id}: span is not verbatim on source page")
        if not item.answer_tokens or any(token.casefold() not in item.span_text.casefold() for token in item.answer_tokens):
            raise ValueError(f"{item.candidate_id}: answer tokens are not grounded in span")
        if any(token.casefold() in item.question.casefold() for token in item.answer_tokens):
            raise ValueError(f"{item.candidate_id}: question leaks an answer token")
        if any(_jaccard(item.span_text, previous) >= 0.82 for previous in seen_texts):
            raise ValueError(f"{item.candidate_id}: near-duplicate evidence sentence")
        seen_texts.append(item.span_text)


def materialize_auto_training_dataset(
    candidates: Iterable[AutoTrainingCandidate],
    *,
    documents: Iterable[Document],
    tasks: Iterable[TaskSpec],
) -> tuple[tuple[Document, ...], tuple[TaskSpec, ...]]:
    items = tuple(candidates)
    document_list = list(documents)
    task_list = list(tasks)
    document_index = {document.document_id: index for index, document in enumerate(document_list)}
    existing_task_ids = {task.task_id for task in task_list}
    existing_span_ids = {span.span_id for document in document_list for span in document.spans}
    for item in items:
        index = document_index.get(item.document_id)
        if index is None:
            raise ValueError(f"{item.candidate_id}: target document is missing")
        span_id = f"{item.candidate_id}-span"
        task_id = f"{item.candidate_id}-task"
        if span_id in existing_span_ids or task_id in existing_task_ids:
            raise ValueError(f"{item.candidate_id}: materialized ID already exists")
        document = document_list[index]
        if document.source_paper_id != item.source_paper_id:
            raise ValueError(f"{item.candidate_id}: target document source mismatch")
        document_list[index] = replace(
            document,
            spans=document.spans
            + (
                Span(
                    span_id,
                    item.span_text,
                    (item.facet,),
                    item.page_number,
                    item.page_text_sha256,
                    "draft",
                ),
            ),
        )
        task_list.append(
            TaskSpec(
                task_id=task_id,
                question=item.question,
                corpus_scope=document.corpus_scope,
                required_facets=(item.facet,),
                allowed_dois=frozenset({document.doi}),
                source_paper_id=item.source_paper_id,
                split=item.split,
                review_status="draft",
                available_memory_ids=(),
                deterministic_answer_key={item.facet: item.answer_tokens},
                task_type="direct_evidence",
            )
        )
        existing_span_ids.add(span_id)
        existing_task_ids.add(task_id)
    validate_task_splits(task_list)
    return tuple(document_list), tuple(task_list)
