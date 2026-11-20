"""CPU contract tests for ``public_support_contract``.

The contract is a single-variable, gold-free harness capability: the minimum number of distinct
documents an answer must draw on is derived at runtime from the question and the titles of the
candidate documents.  Two distinct documents named by the question => the answer must cite two
distinct documents; otherwise the historical minimum (task_type-driven) is unchanged.

Everything here is CPU-only: no GPU, no model, no network, no frozen bundle, no gold data.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evidence_agent_rl.harness import EvidenceHarness  # noqa: E402
from evidence_agent_rl.hotpotqa_answer import ANSWER_PROTOCOL_SHORT  # noqa: E402
from evidence_agent_rl.models import (  # noqa: E402
    Action,
    Document,
    MemoryMode,
    Span,
    TaskSpec,
)
from evidence_agent_rl.public_contract import (  # noqa: E402
    content_tokens,
    matched_document_titles,
    required_distinct_documents,
)

SCOPE = "hotpotqa-fixture"
DOC_A = "hotpotqa:A"
DOC_B = "hotpotqa:B"
DOC_C = "hotpotqa:C"

# Titles are chosen so the question can name exactly two of them ("R. Kelly", "Celine Dion")
# without a shared generic token dragging in the third.
DOCUMENTS = (
    Document(
        document_id=DOC_A,
        title="R. Kelly",
        doi="hotpotqa:A",
        corpus_scope=SCOPE,
        document_type="hotpotqa_paragraph",
        spans=(
            Span("R. Kelly|0", "R. Kelly released the album in 1998.", ()),
            Span("R. Kelly|1", "The album was produced by R. Kelly himself.", ()),
        ),
    ),
    Document(
        document_id=DOC_B,
        title="Celine Dion",
        doi="hotpotqa:B",
        corpus_scope=SCOPE,
        document_type="hotpotqa_paragraph",
        spans=(
            Span("Celine Dion|0", "Celine Dion produced These Are Special Times in 1998.", ()),
        ),
    ),
    Document(
        document_id=DOC_C,
        title="USS Monitor",
        doi="hotpotqa:C",
        corpus_scope=SCOPE,
        document_type="hotpotqa_paragraph",
        spans=(Span("USS Monitor|0", "USS Monitor was an ironclad warship.", ()),),
    ),
)

QUESTION_TWO = (
    "What album did R. Kelly release the same year as producing These Are Special Times "
    "for Celine Dion?"
)
QUESTION_ONE = "What album did R. Kelly release in 1998?"
QUESTION_NONE = "How many albums were released that decade in total?"

SEARCH_QUERY = "album released 1998 produced These Are Special Times R. Kelly Celine Dion"


def task(question: str = QUESTION_TWO, **overrides) -> TaskSpec:
    baseline = dict(
        task_id="fixture-public-contract",
        question=question,
        corpus_scope=SCOPE,
        task_type="direct_evidence",
        # Hidden gold-ish fields are present on purpose: the contract must never read them.
        required_facets=("f1",),
        deterministic_answer_key={"f1": ("album",)},
        allowed_dois=frozenset({"hotpotqa:A", "hotpotqa:B"}),
        required_query_terms=("album",),
        source_paper_id="A",
    )
    baseline.update(overrides)
    return TaskSpec(**baseline)


def harness(
    *,
    spec: TaskSpec | None = None,
    short_protocol: bool = True,
    **kwargs,
) -> EvidenceHarness:
    instance = EvidenceHarness(
        DOCUMENTS,
        max_budget=8,
        memory_mode=MemoryMode.NONE,
        policy_context_version="v6",
        answer_protocol=ANSWER_PROTOCOL_SHORT if short_protocol else "atomic_claims",
        **kwargs,
    )
    instance.reset(spec or task(), episode_id="fixture-public-contract-episode")
    return instance


def search(instance: EvidenceHarness) -> None:
    observation = instance.step(Action("search", {"query": SEARCH_QUERY}))
    assert observation.ok, observation.error


def select(instance: EvidenceHarness, document_id: str, span_id: str) -> str:
    observation = instance.step(
        Action("select_evidence", {"document_id": document_id, "span_id": span_id})
    )
    assert observation.ok, observation.error
    return next(
        handle.handle_id
        for handle in instance.state.evidence.values()
        if handle.document_id == document_id and handle.span_id == span_id
    )


def short_answer(instance: EvidenceHarness, text: str, handle_ids: tuple[str, ...]):
    return instance.step(
        Action("answer", {"answer": text, "supporting_handles": list(handle_ids)})
    )


class PrimitiveTest(unittest.TestCase):
    """The public match is deterministic, order-preserving and gold-free."""

    def test_content_tokens_drop_short_and_stopword_glue(self) -> None:
        tokens = content_tokens("What album did the R. Kelly release in 1998?")
        self.assertIn("album", tokens)
        self.assertIn("kelly", tokens)
        self.assertNotIn("r", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("what", tokens)

    def test_two_named_titles_are_detected(self) -> None:
        titles = ("R. Kelly", "Celine Dion", "USS Monitor")
        self.assertEqual(
            matched_document_titles(QUESTION_TWO, titles),
            ("R. Kelly", "Celine Dion"),
        )

    def test_one_named_title_is_detected(self) -> None:
        titles = ("R. Kelly", "Celine Dion", "USS Monitor")
        self.assertEqual(matched_document_titles(QUESTION_ONE, titles), ("R. Kelly",))

    def test_question_naming_nothing_detects_nothing(self) -> None:
        titles = ("R. Kelly", "Celine Dion", "USS Monitor")
        self.assertEqual(matched_document_titles(QUESTION_NONE, titles), ())

    def test_required_count_is_two_only_when_two_titles_are_named(self) -> None:
        titles = ("R. Kelly", "Celine Dion", "USS Monitor")
        self.assertEqual(required_distinct_documents(QUESTION_TWO, titles), 2)
        self.assertEqual(required_distinct_documents(QUESTION_ONE, titles), 1)
        self.assertEqual(required_distinct_documents(QUESTION_NONE, titles), 1)

    def test_duplicate_titles_are_deduplicated_and_order_is_stable(self) -> None:
        titles = ("Celine Dion", "R. Kelly", "Celine Dion")
        self.assertEqual(
            matched_document_titles(QUESTION_TWO, titles),
            ("Celine Dion", "R. Kelly"),
        )


class DefaultOffTest(unittest.TestCase):
    """Not requesting the contract leaves every observable behaviour at the frozen default."""

    def test_required_count_stays_one(self) -> None:
        instance = harness()
        self.assertEqual(instance.public_support_contract, False)
        self.assertEqual(instance.public_contract_required_documents, 1)
        self.assertEqual(instance.public_contract_matched_document_titles, ())
        self.assertEqual(instance.required_support_document_count(), 1)

    def test_policy_state_has_no_contract_block(self) -> None:
        instance = harness()
        rendered = instance.render_policy_state()
        self.assertNotIn("public_support_contract", rendered)

    def test_single_handle_answer_is_still_accepted(self) -> None:
        instance = harness()
        search(instance)
        handle = select(instance, DOC_A, "R. Kelly|0")
        observation = short_answer(instance, "It was released in 1998", (handle,))
        self.assertTrue(observation.ok, observation.error)
        self.assertTrue(instance.state.done)


class ContractOnTest(unittest.TestCase):
    """With the contract on, the answer must draw on the publicly named documents."""

    def test_requirement_is_two_when_question_names_two_documents(self) -> None:
        instance = harness(public_support_contract=True)
        self.assertEqual(instance.public_contract_required_documents, 2)
        self.assertEqual(
            instance.public_contract_matched_document_titles,
            ("R. Kelly", "Celine Dion"),
        )
        self.assertEqual(instance.required_support_document_count(), 2)
        self.assertEqual(instance.minimum_answer_evidence_handles(), 2)

    def test_policy_state_block_is_public_and_explains_the_rule(self) -> None:
        instance = harness(public_support_contract=True)
        block = instance.render_policy_state()["public_support_contract"]
        self.assertTrue(block["enabled"])
        self.assertEqual(block["required_distinct_documents"], 2)
        self.assertEqual(
            block["matched_document_titles"], ["R. Kelly", "Celine Dion"]
        )
        # No hidden TaskSpec field may appear in the rendered block.
        for key in ("required_facets", "deterministic_answer_key", "allowed_dois"):
            self.assertNotIn(key, block)

    def test_single_document_answer_is_rejected(self) -> None:
        instance = harness(public_support_contract=True)
        search(instance)
        handle = select(instance, DOC_A, "R. Kelly|0")
        observation = short_answer(instance, "It was released in 1998", (handle,))
        self.assertFalse(observation.ok)
        # The public minimum is rendered into the answer schema as ``minItems: 2``, so a
        # one-handle submission is refused at argument validation before the document check.
        self.assertIn("at least 2 items", observation.error)
        self.assertTrue(instance.state.done)

    def test_two_documents_are_accepted(self) -> None:
        instance = harness(public_support_contract=True)
        search(instance)
        first = select(instance, DOC_A, "R. Kelly|0")
        second = select(instance, DOC_B, "Celine Dion|0")
        observation = short_answer(
            instance, "It was released in 1998", (first, second)
        )
        self.assertTrue(observation.ok, observation.error)
        self.assertTrue(instance.state.done)

    def test_two_spans_from_one_document_are_rejected(self) -> None:
        instance = harness(public_support_contract=True)
        search(instance)
        first = select(instance, DOC_A, "R. Kelly|0")
        second = select(instance, DOC_A, "R. Kelly|1")
        observation = short_answer(instance, "It was released in 1998", (first, second))
        self.assertFalse(observation.ok)
        self.assertIn("two different documents", observation.error)
        self.assertTrue(instance.state.done)

    def test_question_naming_one_document_falls_back_to_one(self) -> None:
        instance = harness(spec=task(QUESTION_ONE), public_support_contract=True)
        self.assertEqual(instance.public_contract_required_documents, 1)
        self.assertEqual(instance.public_contract_matched_document_titles, ("R. Kelly",))
        search(instance)
        handle = select(instance, DOC_A, "R. Kelly|0")
        observation = short_answer(instance, "It was released in 1998", (handle,))
        self.assertTrue(observation.ok, observation.error)

    def test_requirement_does_not_depend_on_task_type(self) -> None:
        # The historical minimum came from a builder-side task annotation; the public contract
        # must reach the same requirement on a direct_evidence task without that annotation.
        direct = harness(
            spec=task(task_type="direct_evidence"), public_support_contract=True
        )
        self.assertEqual(direct.public_contract_required_documents, 2)

    def test_atomic_protocol_also_enforces_the_two_document_minimum(self) -> None:
        instance = harness(short_protocol=False, public_support_contract=True)
        search(instance)
        handle = select(instance, DOC_A, "R. Kelly|0")
        observation = instance.step(
            Action(
                "answer",
                {"claims": [{"text": "It was released in 1998", "evidence_handles": [handle]}]},
            )
        )
        self.assertFalse(observation.ok)
        # Under v6 the answer action is not even offered while the public minimum is unmet, so
        # the refusal surfaces as an unavailable action rather than a post-hoc document check.
        self.assertTrue(instance.state.done)

    def test_titles_outside_the_task_scope_are_ignored(self) -> None:
        # Retrieval is scoped to the task's corpus_scope (skills.py:84); the contract must use
        # the same scope, otherwise another question's document titles could raise the
        # requirement.  A document from a foreign scope whose title matches the question must
        # not count.
        foreign = Document(
            document_id="hotpotqa:FOREIGN",
            title="R. Kelly",
            doi="hotpotqa:FOREIGN",
            corpus_scope="some-other-question",
            document_type="hotpotqa_paragraph",
            spans=(Span("R. Kelly|0", "A foreign paragraph.", ()),),
        )
        instance = EvidenceHarness(
            DOCUMENTS + (foreign,),
            max_budget=8,
            memory_mode=MemoryMode.NONE,
            policy_context_version="v6",
            answer_protocol=ANSWER_PROTOCOL_SHORT,
            public_support_contract=True,
        )
        instance.reset(task(QUESTION_ONE), episode_id="scope-episode")
        # Only the in-scope "R. Kelly" document is named, so the requirement stays at one even
        # though two documents in the harness carry that title.
        self.assertEqual(instance.public_contract_required_documents, 1)
        self.assertEqual(instance.public_contract_matched_document_titles, ("R. Kelly",))

    def test_reset_recomputes_the_contract(self) -> None:
        instance = harness(public_support_contract=True)
        self.assertEqual(instance.public_contract_required_documents, 2)
        instance.reset(task(QUESTION_ONE), episode_id="second-episode")
        self.assertEqual(instance.public_contract_required_documents, 1)

    def test_declared_flag_and_contract_compose_monotonically(self) -> None:
        instance = harness(
            spec=task(QUESTION_ONE),
            public_support_contract=True,
            require_distinct_support_documents=True,
        )
        # The contract alone would allow one document; the declared flag still demands two.
        self.assertEqual(instance.required_support_document_count(), 2)


if __name__ == "__main__":
    unittest.main()
