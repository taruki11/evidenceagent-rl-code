"""Small offline checks of public evidence and budget boundaries."""

import json
import unittest

from evidence_agent_rl.harness import EvidenceHarness
from evidence_agent_rl.io import dump_trajectory
from evidence_agent_rl.models import Action, Document, Span, TaskSpec
from evidence_agent_rl.verifier import DeterministicVerifier


class HarnessContractTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(
            "sample",
            "Amber",
            "synthetic:amber",
            "demo",
            "synthetic",
            (Span("color", "Amber is orange.", ("color",)),),
        )
        self.task = TaskSpec(
            "color-task",
            "What color is amber?",
            "demo",
            ("color",),
            frozenset({"synthetic:amber"}),
            deterministic_answer_key={"color": ("orange",)},
        )
        self.harness = EvidenceHarness([self.document])
        self.harness.reset(self.task, episode_id="first")

    def inspect(self):
        self.harness.step(Action("search", {"query": "amber"}))
        self.harness.step(Action("retrieve", {"document_id": "sample"}))
        result = self.harness.step(
            Action("inspect_span", {"document_id": "sample", "span_id": "color"})
        )
        self.assertTrue(result.ok)
        return result.data["handle_id"]

    def test_unchecked_citation_cannot_support_answer(self):
        handle = self.inspect()
        result = self.harness.step(
            Action(
                "answer",
                {
                    "claims": [
                        {"text": "Amber is orange.", "evidence_handles": [handle]}
                    ]
                },
            )
        )
        self.assertFalse(result.ok)
        self.assertIn("citation-checked", result.error)
        self.assertEqual(self.harness.state.final_claims, ())

    def test_evidence_from_previous_episode_is_rejected(self):
        handle = self.inspect()
        self.harness.reset(self.task, episode_id="second")
        result = self.harness.step(Action("check_citation", {"handle_id": handle}))
        self.assertFalse(result.ok)
        self.assertEqual(self.harness.state.firewall_violations, 1)

    def test_budget_exhaustion_cannot_spend_below_zero(self):
        harness = EvidenceHarness([self.document], max_budget=1)
        harness.reset(self.task, episode_id="budget")
        harness.step(Action("search", {"query": "amber"}))
        result = harness.step(Action("retrieve", {"document_id": "sample"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, "budget_exhausted")
        self.assertEqual(harness.state.remaining_budget, 0)

    def test_invalid_arguments_do_not_consume_budget(self):
        before = self.harness.state.remaining_budget
        result = self.harness.step(
            Action("search", {"query": "amber", "unknown": True})
        )
        self.assertFalse(result.ok)
        self.assertEqual(self.harness.state.remaining_budget, before)

    def test_search_respects_corpus_scope(self):
        other = Document(
            "other",
            "Amber",
            "synthetic:other",
            "other-scope",
            "synthetic",
            self.document.spans,
        )
        harness = EvidenceHarness([self.document, other])
        harness.reset(self.task, episode_id="scope")
        result = harness.step(Action("search", {"query": "amber"}))
        self.assertEqual(result.data["document_ids"], ["sample"])

    def test_serialized_actions_replay_to_identical_state(self):
        handle = self.inspect()
        self.harness.step(Action("check_citation", {"handle_id": handle}))
        self.harness.step(
            Action(
                "answer",
                {
                    "claims": [
                        {"text": "Amber is orange.", "evidence_handles": [handle]}
                    ]
                },
            )
        )
        record = json.loads(json.dumps(dump_trajectory(self.harness.state)))
        actions = [Action(**step["action"]) for step in record["steps"]]
        replay = EvidenceHarness([self.document])
        state = replay.replay(self.task, actions, episode_id=record["episode_id"])
        self.assertEqual(state, self.harness.state)
        self.assertTrue(
            DeterministicVerifier.is_success(DeterministicVerifier().score(state))
        )


if __name__ == "__main__":
    unittest.main()
