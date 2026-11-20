"""CPU tests for the optional support-contract repair loop.

The default is the frozen hard reject: the first contract violation ends the episode.  With
``support_repair=True`` the first violations are returned as non-terminal observations with a
hint, so the policy can search/inspect a second document and answer again.  The allowance is
per-episode and resets with the harness.

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

SCOPE = "hotpotqa-fixture"
DOC_A = "hotpotqa:A"
DOC_B = "hotpotqa:B"
DOC_C = "hotpotqa:C"

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
SEARCH_QUERY = "album released 1998 produced These Are Special Times R. Kelly Celine Dion"


def task(question: str = QUESTION_TWO, **overrides) -> TaskSpec:
    baseline = dict(
        task_id="fixture-support-repair",
        question=question,
        corpus_scope=SCOPE,
        task_type="direct_evidence",
        required_facets=("f1",),
        deterministic_answer_key={"f1": ("album",)},
        allowed_dois=frozenset({"hotpotqa:A", "hotpotqa:B"}),
        required_query_terms=("album",),
        source_paper_id="A",
    )
    baseline.update(overrides)
    return TaskSpec(**baseline)


def harness(*, spec: TaskSpec | None = None, **kwargs) -> EvidenceHarness:
    instance = EvidenceHarness(
        DOCUMENTS,
        max_budget=8,
        memory_mode=MemoryMode.NONE,
        policy_context_version="v6",
        answer_protocol=ANSWER_PROTOCOL_SHORT,
        **kwargs,
    )
    instance.reset(spec or task(), episode_id="fixture-support-repair-episode")
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


def same_document_handles(instance: EvidenceHarness) -> tuple[str, str]:
    search(instance)
    return (
        select(instance, DOC_A, "R. Kelly|0"),
        select(instance, DOC_A, "R. Kelly|1"),
    )


class DefaultOffTest(unittest.TestCase):
    def test_default_is_off(self) -> None:
        instance = harness(public_support_contract=True)
        self.assertFalse(instance.support_repair)
        self.assertEqual(instance.support_repair_hints_left, 0)

    def test_hard_reject_still_terminates_and_keeps_the_frozen_error(self) -> None:
        instance = harness(public_support_contract=True)
        first, second = same_document_handles(instance)
        observation = short_answer(instance, "It was released in 1998", (first, second))
        self.assertFalse(observation.ok)
        self.assertTrue(instance.state.done)
        self.assertEqual(
            observation.error,
            "the submitted support spans must come from two different documents",
        )
        self.assertEqual(instance.state.firewall_violations, 0)

    def test_policy_state_has_no_repair_hint(self) -> None:
        instance = harness(public_support_contract=True)
        block = instance.render_policy_state()["public_support_contract"]
        self.assertNotIn("repair_hints_remaining", block)


class RepairOnTest(unittest.TestCase):
    def test_first_violation_is_non_terminal(self) -> None:
        instance = harness(public_support_contract=True, support_repair=True)
        first, second = same_document_handles(instance)
        budget_before = instance.state.remaining_budget
        observation = short_answer(instance, "It was released in 1998", (first, second))
        self.assertFalse(observation.ok)
        self.assertFalse(instance.state.done)
        self.assertEqual(instance.support_repair_hints_left, 1)
        self.assertIn("another document", observation.error)
        self.assertIn("1 repair attempt(s) left", observation.error)
        self.assertEqual(instance.state.remaining_budget, budget_before)
        self.assertEqual(instance.state.firewall_violations, 0)

    def test_repair_then_second_document_succeeds(self) -> None:
        instance = harness(public_support_contract=True, support_repair=True)
        first, second = same_document_handles(instance)
        rejected = short_answer(instance, "It was released in 1998", (first, second))
        self.assertFalse(rejected.ok)
        other = select(instance, DOC_B, "Celine Dion|0")
        accepted = short_answer(instance, "It was released in 1998", (first, other))
        self.assertTrue(accepted.ok, accepted.error)
        self.assertTrue(instance.state.done)
        self.assertEqual(len(instance.state.final_claims), 1)

    def test_hints_exhaust_then_terminal(self) -> None:
        instance = harness(
            public_support_contract=True,
            support_repair=True,
            support_repair_max_hints=1,
        )
        first, second = same_document_handles(instance)
        first_reject = short_answer(instance, "It was released in 1998", (first, second))
        self.assertFalse(first_reject.ok)
        self.assertFalse(instance.state.done)
        self.assertEqual(instance.support_repair_hints_left, 0)
        terminal = short_answer(instance, "It was released in 1998", (first, second))
        self.assertFalse(terminal.ok)
        self.assertTrue(instance.state.done)
        self.assertEqual(
            terminal.error,
            "the submitted support spans must come from two different documents",
        )

    def test_reset_restores_the_allowance(self) -> None:
        instance = harness(public_support_contract=True, support_repair=True)
        first, second = same_document_handles(instance)
        short_answer(instance, "It was released in 1998", (first, second))
        self.assertEqual(instance.support_repair_hints_left, 1)
        instance.reset(task(), episode_id="second-episode")
        self.assertEqual(instance.support_repair_hints_left, 2)

    def test_policy_state_reports_remaining_hints(self) -> None:
        instance = harness(public_support_contract=True, support_repair=True)
        self.assertEqual(
            instance.render_policy_state()["public_support_contract"][
                "repair_hints_remaining"
            ],
            2,
        )
        first, second = same_document_handles(instance)
        short_answer(instance, "It was released in 1998", (first, second))
        self.assertEqual(
            instance.render_policy_state()["public_support_contract"][
                "repair_hints_remaining"
            ],
            1,
        )

class NoViolationTest(unittest.TestCase):
    def test_one_document_task_is_unaffected(self) -> None:
        instance = harness(
            spec=task(QUESTION_ONE),
            public_support_contract=True,
            support_repair=True,
        )
        search(instance)
        handle = select(instance, DOC_A, "R. Kelly|0")
        observation = short_answer(instance, "It was released in 1998", (handle,))
        self.assertTrue(observation.ok, observation.error)
        self.assertEqual(instance.support_repair_hints_left, 2)

    def test_repair_without_any_requirement_is_inert(self) -> None:
        instance = harness(public_support_contract=False, support_repair=True)
        search(instance)
        handle = select(instance, DOC_A, "R. Kelly|0")
        observation = short_answer(instance, "It was released in 1998", (handle,))
        self.assertTrue(observation.ok, observation.error)


class ValidationTest(unittest.TestCase):
    def test_support_repair_must_be_boolean(self) -> None:
        with self.assertRaises(TypeError):
            EvidenceHarness(DOCUMENTS, support_repair=1)

    def test_max_hints_must_be_a_non_negative_integer(self) -> None:
        with self.assertRaises(TypeError):
            EvidenceHarness(DOCUMENTS, support_repair_max_hints=True)
        with self.assertRaises(ValueError):
            EvidenceHarness(DOCUMENTS, support_repair_max_hints=-1)


if __name__ == "__main__":
    unittest.main()
