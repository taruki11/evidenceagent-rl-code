"""Tests for the zero-GPU HotpotQA harness-vs-SFT failure-funnel audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_hv_failure_funnel as funnel  # noqa: E402


def _make_row(
    task_id: str,
    *,
    task_type: str = "bridge",
    empty_answer: bool = False,
    abstention_reason: str | None = None,
    observation_errors: list[str] | None = None,
    action_sequence: list[str] | None = None,
    gold_sp: list[list[object]] | None = None,
    cited_sp: list[list[object]] | None = None,
    pred_sp: list[list[object]] | None = None,
    em: float = 0.0,
    f1: float = 0.0,
    sp_em: float = 0.0,
    sp_f1: float = 0.0,
    joint_em: float = 0.0,
    joint_f1: float = 0.0,
    m1: bool = False,
    m2: bool = False,
    n_decisions: int = 4,
    repair_events: int = 0,
    decode_errors: int = 0,
    firewall_violations: int = 0,
) -> dict:
    return {
        "task_id": task_id,
        "type": task_type,
        "empty_answer": empty_answer,
        "abstention_reason": abstention_reason,
        "observation_errors": observation_errors or [],
        "action_sequence": action_sequence or ["search", "select_evidence", "answer"],
        "gold_sp": gold_sp if gold_sp is not None else [["DocA", 0], ["DocB", 1]],
        "cited_sp": cited_sp if cited_sp is not None else [],
        "pred_sp": pred_sp if pred_sp is not None else [],
        "official": {
            "em": em,
            "f1": f1,
            "sp_em": sp_em,
            "sp_f1": sp_f1,
            "joint_em": joint_em,
            "joint_f1": joint_f1,
        },
        "M1_full_support_reachable": m1,
        "M2_full_support_selected": m2,
        "n_decisions": n_decisions,
        "support_repair_events": repair_events,
        "decode_errors": decode_errors,
        "firewall_violations": firewall_violations,
    }


class FailureFunnelAuditTest(unittest.TestCase):
    def test_all_primary_buckets_and_sub_buckets(self) -> None:
        rows = [
            # 1. joint_em_success (2-sent gold)
            _make_row(
                "t1",
                cited_sp=[["DocA", 0], ["DocB", 1]],
                pred_sp=[["DocA", 0], ["DocB", 1]],
                em=1.0,
                f1=1.0,
                sp_em=1.0,
                sp_f1=1.0,
                joint_em=1.0,
                joint_f1=1.0,
                m1=True,
                m2=True,
            ),
            # 2a. empty_support_reject via abstention_reason (m1=True)
            _make_row(
                "t2",
                empty_answer=True,
                abstention_reason="The supporting evidence must come from two different paragraphs.",
                action_sequence=["search", "select_evidence", "abstain"],
                m1=True,
            ),
            # 2b. empty_support_reject via observation_errors[-1] (m1=False)
            _make_row(
                "t3",
                empty_answer=True,
                abstention_reason=None,
                observation_errors=[
                    "the supporting evidence must come from two different documents"
                ],
                action_sequence=["search", "select_evidence", "answer"],
                m1=False,
            ),
            # 3a. empty_other_decision_limit after prior support reject
            _make_row(
                "t4",
                empty_answer=True,
                observation_errors=[
                    "the submitted support spans must come from two different documents",
                    "policy decision limit exhausted",
                ],
                action_sequence=["search", "answer", "search"],
                m1=True,
                repair_events=1,
            ),
            # 3b. empty_other_firewall_reject
            _make_row(
                "t5",
                empty_answer=True,
                observation_errors=["foreign or historical evidence handle: hp-1"],
                firewall_violations=1,
            ),
            # 3c. empty_other_misc
            _make_row(
                "t6",
                empty_answer=True,
                abstention_reason="insufficient information",
            ),
            # 4a. sp_ok_ans_miss near-miss (2-sent gold)
            _make_row(
                "t7",
                task_type="comparison",
                cited_sp=[["DocA", 0], ["DocB", 1]],
                pred_sp=[["DocA", 0], ["DocB", 1]],
                em=0.0,
                f1=0.6667,
                sp_em=1.0,
                sp_f1=1.0,
                joint_em=0.0,
                joint_f1=0.6667,
                m1=True,
                m2=True,
            ),
            # 4b. sp_ok_ans_miss low-f1 (3-sent gold)
            _make_row(
                "t8",
                gold_sp=[["DocA", 0], ["DocA", 1], ["DocB", 0]],
                cited_sp=[["DocA", 0], ["DocA", 1], ["DocB", 0]],
                pred_sp=[["DocA", 0], ["DocA", 1], ["DocB", 0]],
                em=0.0,
                f1=0.2,
                sp_em=1.0,
                sp_f1=1.0,
                joint_em=0.0,
                joint_f1=0.2,
                m1=True,
                m2=True,
            ),
            # 5a. ans_ok_sp_miss both docs cited, sentence miss (3-sent gold)
            _make_row(
                "t9",
                gold_sp=[["DocA", 0], ["DocA", 1], ["DocB", 0]],
                cited_sp=[["DocA", 0], ["DocB", 0]],
                pred_sp=[["DocA", 0], ["DocB", 0]],
                em=1.0,
                f1=1.0,
                sp_em=0.0,
                sp_f1=0.8,
                joint_em=0.0,
                joint_f1=0.8,
                m1=True,
                m2=False,
            ),
            # 5b. ans_ok_sp_miss missing gold doc
            _make_row(
                "t10",
                cited_sp=[["DocA", 0], ["DocC", 2]],
                pred_sp=[["DocA", 0], ["DocC", 2]],
                em=1.0,
                f1=1.0,
                sp_em=0.0,
                sp_f1=0.5,
                m1=False,
                m2=False,
            ),
            # 6a. both_miss m1 unreachable
            _make_row("t11", m1=False, m2=False),
            # 6b. both_miss m1 reachable m2 unselected
            _make_row("t12", m1=True, m2=False),
            # 6c. both_miss m2 selected (3-sent gold)
            _make_row(
                "t13",
                gold_sp=[["DocA", 0], ["DocA", 1], ["DocB", 0]],
                cited_sp=[["DocA", 0], ["DocC", 0]],
                m1=True,
                m2=True,
            ),
        ]

        res = funnel.audit_arm(rows, pred_sha256="deadbeef")
        self.assertEqual(res["n"], 13)
        self.assertEqual(sum(res["buckets"].values()), 13)
        self.assertEqual(
            res["buckets"],
            {
                "joint_em_success": 1,
                "empty_support_reject": 2,
                "empty_other": 3,
                "sp_ok_ans_miss": 2,
                "ans_ok_sp_miss": 2,
                "both_miss": 3,
            },
        )
        self.assertEqual(res["cited_joint_em_count"], 1)
        self.assertEqual(res["empty_count"], 5)

        sub = res["sub_buckets"]
        self.assertEqual(sub["empty_support_reject_m1_reachable"], 1)
        self.assertEqual(sub["empty_other_m1_reachable"], 1)
        self.assertEqual(sub["empty_other_decision_limit"], 1)
        self.assertEqual(sub["empty_other_decision_limit_after_support_reject"], 1)
        self.assertEqual(sub["empty_other_firewall_reject"], 1)
        self.assertEqual(sub["empty_other_misc"], 1)
        self.assertEqual(sub["sp_ok_ans_near_miss"], 1)
        self.assertEqual(sub["sp_ok_ans_low_f1"], 1)
        self.assertEqual(sub["ans_ok_both_docs_cited_sent_miss"], 1)
        self.assertEqual(sub["ans_ok_missing_gold_doc"], 1)
        self.assertEqual(sub["ans_ok_sp_miss_gold_3plus_sents"], 1)
        self.assertEqual(sub["both_miss_m1_unreachable"], 1)
        self.assertEqual(sub["both_miss_m1_reachable_m2_unselected"], 1)
        self.assertEqual(sub["both_miss_m2_selected"], 1)

        # 2-sent vs 3+-sent split: t8, t9, t13 have 3 gold sents; remaining 10 have 2.
        self.assertEqual(res["sp_em_on_2sent_gold"]["total"], 10)
        self.assertEqual(res["sp_em_on_2sent_gold"]["hits"], 2)  # t1, t7
        self.assertEqual(res["sp_em_on_2sent_gold"]["joint_em_hits"], 1)  # t1
        self.assertEqual(res["sp_em_on_3plus_sent_gold"]["total"], 3)
        self.assertEqual(res["sp_em_on_3plus_sent_gold"]["hits"], 1)  # t8
        self.assertEqual(res["sp_em_on_3plus_sent_gold"]["joint_em_hits"], 0)

    def test_cli_excludes_smoke_and_aborted_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs"
            study_valid = runs_root / "hv-clean120-2x2-20261120"
            study_aborted = runs_root / "hv-clean120-2x2-20261120-aborted"

            (study_valid / "v8_blanket2-r1").mkdir(parents=True)
            (study_valid / "smoke").mkdir(parents=True)
            (study_valid / "arm-aborted").mkdir(parents=True)
            (study_aborted / "v8_blanket2-r1").mkdir(parents=True)

            row_ok = _make_row("ok-1", em=1.0, f1=1.0, sp_em=1.0, sp_f1=1.0, joint_em=1.0, joint_f1=1.0)
            row_bad = _make_row("bad-1", empty_answer=True, abstention_reason="two different documents")

            (study_valid / "v8_blanket2-r1" / "predictions.jsonl").write_text(
                json.dumps(row_ok) + "\n", encoding="utf-8"
            )
            (study_valid / "smoke" / "predictions.jsonl").write_text(
                json.dumps(row_bad) + "\n", encoding="utf-8"
            )
            (study_valid / "arm-aborted" / "predictions.jsonl").write_text(
                json.dumps(row_bad) + "\n", encoding="utf-8"
            )
            (study_aborted / "v8_blanket2-r1" / "predictions.jsonl").write_text(
                json.dumps(row_bad) + "\n", encoding="utf-8"
            )

            out_json = root / "out" / "funnel.json"
            out_md = root / "out" / "funnel.md"
            rc = funnel.main(
                [
                    "--runs-root",
                    str(runs_root),
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_json.is_file())
            self.assertTrue(out_md.is_file())

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(list(report["studies"].keys()), ["hv-clean120-2x2-20261120"])
            arms = report["studies"]["hv-clean120-2x2-20261120"]["arms"]
            self.assertEqual(list(arms.keys()), ["v8_blanket2-r1"])
            self.assertEqual(arms["v8_blanket2-r1"]["cited_joint_em_count"], 1)
            self.assertIn("hv-clean120-2x2-20261120", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
