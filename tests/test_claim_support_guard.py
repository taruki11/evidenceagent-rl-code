"""CPU contract tests for ``claim_support_guard``.

The guard is a single-variable, gold-free intervention: at answer time the harness rejects a
claim whose own cited span does not carry the claim's text (all numbers must appear, content
coverage must reach the threshold), returning a structured repair hint instead.  It must not
spend budget, must not terminate the episode, and must be completely inert when not requested.

Everything here is CPU-only: no GPU, no model, no network, no frozen bundle.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evidence_agent_rl.claim_support import (  # noqa: E402
    DEFAULT_MIN_COVERAGE,
    claim_support_gaps,
    content_tokens,
    numeric_tokens,
)
from evidence_agent_rl.harness import EvidenceHarness  # noqa: E402
from evidence_agent_rl.models import (  # noqa: E402
    Action,
    Claim,
    Document,
    EvidenceHandle,
    MemoryMode,
    Span,
    TaskSpec,
)

SCOPE = "paper-v2-text-evidence"
PAPER = "arxiv:A"
DISTRACTOR = "arxiv:B"

DOCUMENTS = (
    Document(
        document_id=PAPER,
        title="Alpha Treatment Trial",
        doi="10.1000/A",
        corpus_scope=SCOPE,
        document_type="research",
        spans=(
            Span("A1", "The treatment reduced blood pressure by 42 percent in the trial", ("f1",)),
            Span("A2", "Sodium intake was measured at baseline", ("f2",)),
        ),
    ),
    Document(
        document_id=DISTRACTOR,
        title="Beta Treatment Trial",
        doi="10.1000/B",
        corpus_scope=SCOPE,
        document_type="research",
        spans=(
            Span("B1", "The treatment reduced blood pressure by 17 percent", ()),
        ),
    ),
)

QUESTION = "How much did the treatment reduce blood pressure?"
SUPPORTED_CLAIM = "The treatment reduced blood pressure by 42 percent in the trial"
# A claim the cited span does not carry: different number and different content words.
UNSUPPORTED_CLAIM = "The trial enrolled 1200 patients with hypertension"
# Same subject and same headline number, different assertion.  A purely lexical check cannot
# see this drift; the test below pins that limitation so nobody mistakes the guard for a scorer.
DRIFT_CLAIM = "The treatment reduced blood pressure by 42 percent at twelve weeks"

# Fields a guard hint must never leak: they are hidden TaskSpec fields, i.e. gold.
HIDDEN_TASK_FIELDS = frozenset(
    {
        "required_facets",
        "deterministic_answer_key",
        "allowed_dois",
        "required_query_terms",
        "source_paper_id",
        "source_paper_ids",
        "mismatch_pair_id",
        "mismatch_role",
    }
)


def task(**overrides) -> TaskSpec:
    baseline = dict(
        task_id="fixture-claim-support",
        question=QUESTION,
        corpus_scope=SCOPE,
        required_facets=("f1",),
        allowed_dois=frozenset({"10.1000/A"}),
        source_paper_id="A",
        deterministic_answer_key={"f1": ("42",)},
        required_query_terms=("blood",),
        task_type="direct_evidence",
    )
    baseline.update(overrides)
    return TaskSpec(**baseline)


def harness(*, spec: TaskSpec | None = None, budget: int = 8, **kwargs) -> EvidenceHarness:
    instance = EvidenceHarness(
        DOCUMENTS,
        max_budget=budget,
        memory_mode=MemoryMode.NONE,
        policy_context_version="v6",
        **kwargs,
    )
    instance.reset(spec or task(), episode_id="fixture-episode")
    return instance


def select_paper_span(instance: EvidenceHarness, document_id: str = PAPER, span_id: str = "A1"):
    instance.step(Action("search", {"query": "blood pressure treatment"}))
    observation = instance.step(
        Action("select_evidence", {"document_id": document_id, "span_id": span_id})
    )
    assert observation.ok, observation.error
    return next(
        handle
        for handle in instance.state.evidence.values()
        if handle.document_id == document_id and handle.span_id == span_id
    )


def answer(instance: EvidenceHarness, text: str, handle_ids: tuple[str, ...]):
    return instance.step(
        Action(
            "answer",
            {"claims": [{"text": text, "evidence_handles": list(handle_ids)}]},
        )
    )


class PrimitiveTest(unittest.TestCase):
    """The lexical primitives are deterministic and conservative."""

    def test_numeric_tokens_are_order_preserving_and_deduplicated(self) -> None:
        self.assertEqual(numeric_tokens("42% then 3.5 then 42%"), ("42%", "3.5"))

    def test_content_tokens_drop_stopwords_and_short_glue(self) -> None:
        tokens = content_tokens("the blood pressure of 42 percent")
        self.assertIn("blood", tokens)
        self.assertIn("pressure", tokens)
        self.assertIn("percent", tokens)
        self.assertIn("42", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("of", tokens)

    def _handles(self) -> dict[str, EvidenceHandle]:
        return {
            "h1": EvidenceHandle(
                handle_id="h1",
                episode_id="e",
                task_id="t",
                document_id=PAPER,
                span_id="A1",
                doi="10.1000/A",
                text=SUPPORTED_CLAIM,
                facets=("f1",),
                checked=True,
            ),
            "h2": EvidenceHandle(
                handle_id="h2",
                episode_id="e",
                task_id="t",
                document_id=PAPER,
                span_id="A2",
                doi="10.1000/A",
                text="Sodium intake was measured at baseline",
                facets=("f2",),
                checked=True,
            ),
        }

    def test_verbatim_claim_is_supported(self) -> None:
        gaps = claim_support_gaps(
            (Claim(text=SUPPORTED_CLAIM, evidence_handles=("h1",)),), self._handles()
        )
        self.assertEqual(gaps, ())

    def test_citation_swap_is_caught_by_content_coverage(self) -> None:
        gaps = claim_support_gaps(
            (
                Claim(
                    text="Sodium intake was measured at baseline",
                    evidence_handles=("h1",),
                ),
            ),
            self._handles(),
        )
        self.assertEqual(len(gaps), 1)
        gap = gaps[0]
        self.assertEqual(gap["claim_index"], 0)
        self.assertEqual(gap["cited_document_ids"], [PAPER])
        self.assertEqual(gap["content_coverage"], 0.0)

    def test_documented_limitation_same_subject_drift_is_not_caught(self) -> None:
        # Every number and almost every content word of DRIFT_CLAIM also occurs in the cited
        # span, so the lexical check passes it.  The guard is therefore a bounded repair-hint
        # generator, not a support scorer; it must never be used as a reward.
        gaps = claim_support_gaps(
            (Claim(text=DRIFT_CLAIM, evidence_handles=("h1",)),), self._handles()
        )
        self.assertEqual(gaps, ())

    def test_accepts_raw_scoreboard_dicts(self) -> None:
        # The replay audit feeds the same function the JSON dicts straight out of
        # final_scoreboard.json, so both object-style and dict-style handles must work.
        handles = {
            "h1": {
                "handle_id": "h1",
                "document_id": PAPER,
                "text": SUPPORTED_CLAIM,
            }
        }
        claims = [{"text": SUPPORTED_CLAIM, "evidence_handles": ["h1"]}]
        self.assertEqual(claim_support_gaps(claims, handles), ())

        flagged = claim_support_gaps(
            [{"text": UNSUPPORTED_CLAIM, "evidence_handles": ["h1"]}], handles
        )
        self.assertEqual(len(flagged), 1)
        self.assertIn("1200", flagged[0]["missing_numbers"])

    def test_zero_threshold_still_requires_number_containment(self) -> None:
        # Numbers are all-or-nothing by design: a hallucinated statistic is the costliest
        # failure here, so lowering the coverage threshold must not relax number containment.
        claims = (Claim(text=UNSUPPORTED_CLAIM, evidence_handles=("h1",)),)
        gaps = claim_support_gaps(claims, self._handles(), min_coverage=0.0)
        self.assertEqual(len(gaps), 1)
        self.assertIn("1200", gaps[0]["missing_numbers"])


class DefaultOffTest(unittest.TestCase):
    """Without the flag the harness is byte-for-byte the frozen behaviour."""

    def test_flag_defaults_to_off(self) -> None:
        self.assertFalse(harness().claim_support_guard)

    def test_unsupported_answer_is_accepted_when_off(self) -> None:
        instance = harness()
        handle = select_paper_span(instance)

        observation = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))

        self.assertTrue(observation.ok, observation.error)
        self.assertTrue(instance.state.done)
        self.assertNotEqual(observation.kind, "claim_support_gap")

    def test_policy_state_gains_no_key_when_off(self) -> None:
        instance = harness()
        rendered = instance.render_policy_state()

        self.assertNotIn("claim_support_guard", rendered)
        # same fixture, no flag -> identical dict
        self.assertEqual(rendered, harness().render_policy_state())


class GuardEnabledTest(unittest.TestCase):
    """With the flag on, an unsupported answer becomes a bounded repair hint."""

    def test_unsupported_answer_returns_structured_hint_without_charging(self) -> None:
        instance = harness(claim_support_guard=True, claim_support_max_hints=3)
        handle = select_paper_span(instance)
        budget_before = instance.state.remaining_budget
        steps_before = len(instance.state.trace)

        observation = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))

        self.assertFalse(observation.ok)
        self.assertEqual(observation.kind, "claim_support_gap")
        self.assertFalse(instance.state.done)
        self.assertEqual(instance.state.remaining_budget, budget_before)
        self.assertEqual(len(instance.state.trace), steps_before + 1)
        self.assertEqual(instance.claim_support_hints_left, 2)
        gap = observation.data["gaps"][0]
        self.assertEqual(gap["claim_index"], 0)
        self.assertIn("1200", gap["missing_numbers"])
        self.assertEqual(observation.data["hints_remaining"], 2)
        self.assertEqual(observation.data["min_coverage"], DEFAULT_MIN_COVERAGE)
        self.assertIn("cite the span", observation.error or "")

    def test_supported_answer_still_terminates_the_episode(self) -> None:
        instance = harness(claim_support_guard=True)
        handle = select_paper_span(instance)

        observation = answer(instance, SUPPORTED_CLAIM, (handle.handle_id,))

        self.assertTrue(observation.ok, observation.error)
        self.assertTrue(instance.state.done)

    def test_hint_allowance_is_bounded_then_the_answer_goes_through(self) -> None:
        instance = harness(claim_support_guard=True, claim_support_max_hints=1)
        handle = select_paper_span(instance)

        first = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))
        self.assertEqual(first.kind, "claim_support_gap")
        second = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))

        self.assertTrue(second.ok, second.error)
        self.assertTrue(instance.state.done)
        self.assertEqual(instance.claim_support_hints_left, 0)

    def test_zero_hints_disables_the_intervention(self) -> None:
        instance = harness(claim_support_guard=True, claim_support_max_hints=0)
        handle = select_paper_span(instance)

        observation = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))

        self.assertTrue(observation.ok, observation.error)
        self.assertTrue(instance.state.done)

    def test_hint_payload_leaks_no_hidden_task_field(self) -> None:
        instance = harness(claim_support_guard=True)
        handle = select_paper_span(instance)

        observation = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))

        self.assertFalse(HIDDEN_TASK_FIELDS & set(observation.data))
        self.assertFalse(HIDDEN_TASK_FIELDS & set(observation.data["gaps"][0]))

    def test_unknown_handle_is_left_to_the_firewall(self) -> None:
        instance = harness(claim_support_guard=True)
        select_paper_span(instance)

        observation = answer(instance, UNSUPPORTED_CLAIM, ("does-not-exist",))

        self.assertFalse(observation.ok)
        self.assertNotEqual(observation.kind, "claim_support_gap")
        self.assertGreaterEqual(instance.state.firewall_violations, 1)

    def test_allowance_is_restored_on_reset(self) -> None:
        instance = harness(claim_support_guard=True, claim_support_max_hints=2)
        handle = select_paper_span(instance)
        answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))
        self.assertEqual(instance.claim_support_hints_left, 1)

        instance.reset(task(), episode_id="second-episode")

        self.assertEqual(instance.claim_support_hints_left, 2)

    def test_policy_state_advertises_the_allowance(self) -> None:
        instance = harness(claim_support_guard=True, claim_support_max_hints=2,
                           claim_support_min_coverage=0.5)

        rendered = instance.render_policy_state()["claim_support_guard"]

        self.assertEqual(
            rendered,
            {"enabled": True, "hints_remaining": 2, "min_coverage": 0.5},
        )

    def test_short_answer_protocol_gets_the_same_guard(self) -> None:
        from evidence_agent_rl.hotpotqa_answer import ANSWER_PROTOCOL_SHORT

        instance = harness(
            claim_support_guard=True,
            answer_protocol=ANSWER_PROTOCOL_SHORT,
        )
        instance.step(Action("search", {"query": "blood pressure treatment"}))
        instance.step(
            Action("select_evidence", {"document_id": PAPER, "span_id": "A1"})
        )
        handle_id = next(iter(instance.state.evidence))

        bad = instance.step(
            Action(
                "answer",
                {"answer": UNSUPPORTED_CLAIM, "supporting_handles": [handle_id]},
            )
        )
        self.assertEqual(bad.kind, "claim_support_gap")

        good = instance.step(
            Action(
                "answer",
                {"answer": SUPPORTED_CLAIM, "supporting_handles": [handle_id]},
            )
        )
        self.assertTrue(good.ok, good.error)


class PromptVisibilityTest(unittest.TestCase):
    """v3-v6 strip ``observation.data``, so the actionable detail must reach ``error``."""

    def test_hint_detail_survives_v6_context_stripping(self) -> None:
        from evidence_agent_rl.react import build_policy_context

        instance = harness(claim_support_guard=True)
        handle = select_paper_span(instance)
        observation = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))
        self.assertEqual(observation.kind, "claim_support_gap")
        self.assertIn("1200", observation.error or "")
        self.assertIn("claim 1", observation.error or "")

        context = build_policy_context(instance, "follow the instructions")
        last = context["trace"][-1]["observation"]

        # the stripped context keeps the actionable error but drops the structured payload
        self.assertEqual(last["kind"], "claim_support_gap")
        self.assertIn("1200", last["error"] or "")
        self.assertNotIn("gaps", last)


class ContractInteractionTest(unittest.TestCase):
    """The guard must never preempt an answer the frozen contract rejects on its own."""

    def test_guard_does_not_fire_when_answer_is_unavailable(self) -> None:
        spec = task(task_type="multi_hop")
        instance = harness(spec=spec, claim_support_guard=True)
        handle = select_paper_span(instance)

        observation = answer(instance, UNSUPPORTED_CLAIM, (handle.handle_id,))

        # multi_hop needs two checked handles; the availability gate rejects first and the
        # v6 answer submission stays terminal, so the guard must stay silent.
        self.assertNotEqual(observation.kind, "claim_support_gap")
        self.assertFalse(observation.ok)
        self.assertTrue(instance.state.done)

    def test_multi_handle_claim_is_left_to_the_atomic_contract(self) -> None:
        instance = harness(claim_support_guard=True)
        instance.step(Action("search", {"query": "blood pressure treatment"}))
        for span_id in ("A1", "A2"):
            instance.step(
                Action("select_evidence", {"document_id": PAPER, "span_id": span_id})
            )
        handles = tuple(instance.state.evidence)

        observation = answer(instance, UNSUPPORTED_CLAIM, handles)

        self.assertNotEqual(observation.kind, "claim_support_gap")
        self.assertIn("exactly one evidence handle", observation.error or "")
        self.assertTrue(instance.state.done)


if __name__ == "__main__":
    unittest.main()
