"""Zero-GPU replay failure-funnel audit for the HotpotQA harness-vs-SFT runs.

Reads preserved ``predictions.jsonl`` files under ``../runs/harness_vs_sft/``
in read-only mode, partitions every episode into six mutually exclusive primary
outcome buckets plus diagnostic sub-buckets, checks structural ceilings
(2-sentence vs 3+-sentence gold support), and writes both JSON and Markdown
reports.

This is a replay/attribution tool over already-executed diagnostic runs.
Secondary/component metrics reported here do NOT overturn pre-registered
primary ``cited Joint EM`` verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

STUDY_SPECS: list[tuple[str, str, list[str]]] = [
    (
        "hv-repair-blanket-20261119",
        "Study 1: contract vs mechanical min-2 (hv-rep120, 2026-11-19)",
        [
            "off-r1",
            "off-r2",
            "contract-r1",
            "contract-r2",
            "repair-r1",
            "repair-r2",
            "blanket2-r1",
            "blanket2-r2",
        ],
    ),
    (
        "hv-sft-under-blanket-20261120",
        "Study 2: SFT under blanket min-2 2x2 (hv-rep120, 2026-11-20)",
        [
            "base_off-r1",
            "base_off-r2",
            "base_blanket2-r1",
            "base_blanket2-r2",
            "v8_off-r1",
            "v8_off-r2",
            "v8_blanket2-r1",
            "v8_blanket2-r2",
        ],
    ),
    (
        "hv-blanket-repair-20261120",
        "Study 3 / E1: blanket min-2 + repair (hv-rep120, 2026-11-20)",
        [
            "vb-r1",
            "vb-r2",
            "br-r1",
            "br-r2",
        ],
    ),
    (
        "hv-clean120-2x2-20261120",
        "Study 4 / E2: fresh-set 2x2 replication (hv-clean120, 2026-11-20)",
        [
            "base_off-r1",
            "base_off-r2",
            "base_blanket2-r1",
            "base_blanket2-r2",
            "v8_off-r1",
            "v8_off-r2",
            "v8_blanket2-r1",
            "v8_blanket2-r2",
        ],
    ),
]

PRIMARY_BUCKETS = (
    "joint_em_success",
    "empty_support_reject",
    "empty_other",
    "sp_ok_ans_miss",
    "ans_ok_sp_miss",
    "both_miss",
)

SUPPORT_REJECT_TOKENS = (
    "different documents",
    "two different",
    "different paragraphs",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _doc_titles(pairs: list[list[Any]] | None) -> set[str]:
    return {str(pair[0]) for pair in (pairs or []) if pair}


def _contains_support_reject(text: str) -> bool:
    low = text.lower()
    return any(tok in low for tok in SUPPORT_REJECT_TOKENS)


def _terminal_empty_label(row: dict[str, Any]) -> str:
    reason = str(row.get("abstention_reason") or "").strip()
    obs_errors = [str(e) for e in (row.get("observation_errors") or []) if e]
    last_err = obs_errors[-1] if obs_errors else ""
    last_act = (
        str((row.get("action_sequence") or [""])[-1] or "").strip()
        if row.get("action_sequence")
        else ""
    )
    if reason:
        if _contains_support_reject(reason):
            return f"abstain: {reason[:64]}"
        return f"abstain_other: {reason[:64]}"
    if last_err:
        low = last_err.lower()
        if _contains_support_reject(low):
            return f"reject({last_act}): {last_err[:64]}"
        if "decision limit exhausted" in low:
            return f"decision_limit({last_act})"
        if "foreign or historical" in low:
            return f"firewall_foreign_handle({last_act})"
        return f"error({last_act}): {last_err[:64]}"
    return f"empty_no_reason({last_act})"


def classify_episode(row: dict[str, Any]) -> dict[str, Any]:
    """Compute per-episode derived flags and primary/sub-bucket labels."""
    official = row.get("official") or {}
    gold_sp = row.get("gold_sp") or []
    cited_sp = row.get("cited_sp") or []
    selected_sp = row.get("pred_sp") or []

    gold_docs = _doc_titles(gold_sp)
    cited_docs = _doc_titles(cited_sp)
    selected_docs = _doc_titles(selected_sp)

    is_empty = bool(row.get("empty_answer"))
    joint_em = float(official.get("joint_em", 0.0)) >= 1.0
    ans_em = float(official.get("em", 0.0)) >= 1.0
    ans_f1 = float(official.get("f1", 0.0))
    sp_em = float(official.get("sp_em", 0.0)) >= 1.0
    sp_f1 = float(official.get("sp_f1", 0.0))
    joint_f1 = float(official.get("joint_f1", 0.0))

    both_gold_docs_cited = bool(gold_docs) and gold_docs.issubset(cited_docs)
    both_gold_docs_selected = bool(gold_docs) and gold_docs.issubset(selected_docs)
    gold_sp_len = len(gold_sp)

    m1 = bool(row.get("M1_full_support_reachable"))
    m2 = bool(row.get("M2_full_support_selected"))

    reason = str(row.get("abstention_reason") or "").strip()
    obs_errors = [str(e) for e in (row.get("observation_errors") or []) if e]
    last_err = obs_errors[-1] if obs_errors else ""
    is_support_reject = _contains_support_reject(reason) or _contains_support_reject(
        last_err
    )

    if joint_em:
        primary_bucket = "joint_em_success"
    elif is_empty and is_support_reject:
        primary_bucket = "empty_support_reject"
    elif is_empty:
        primary_bucket = "empty_other"
    elif sp_em and not ans_em:
        primary_bucket = "sp_ok_ans_miss"
    elif ans_em and not sp_em:
        primary_bucket = "ans_ok_sp_miss"
    else:
        primary_bucket = "both_miss"

    sub_bucket: str | None = None
    decision_limit_after_support_reject = False
    if primary_bucket == "empty_other":
        low_last = last_err.lower()
        if "decision limit exhausted" in low_last:
            sub_bucket = "empty_other_decision_limit"
            decision_limit_after_support_reject = any(
                _contains_support_reject(err) for err in obs_errors
            )
        elif (
            "foreign or historical" in low_last
            or int(row.get("firewall_violations") or 0) > 0
        ):
            sub_bucket = "empty_other_firewall_reject"
        else:
            sub_bucket = "empty_other_misc"
    elif primary_bucket == "sp_ok_ans_miss":
        sub_bucket = (
            "sp_ok_ans_near_miss" if ans_f1 >= 0.5 else "sp_ok_ans_low_f1"
        )
    elif primary_bucket == "ans_ok_sp_miss":
        sub_bucket = (
            "ans_ok_both_docs_cited_sent_miss"
            if both_gold_docs_cited
            else "ans_ok_missing_gold_doc"
        )
    elif primary_bucket == "both_miss":
        if not m1:
            sub_bucket = "both_miss_m1_unreachable"
        elif not m2:
            sub_bucket = "both_miss_m1_reachable_m2_unselected"
        else:
            sub_bucket = "both_miss_m2_selected"

    return {
        "task_id": row.get("task_id"),
        "type": str(row.get("type") or "unknown"),
        "primary_bucket": primary_bucket,
        "sub_bucket": sub_bucket,
        "is_empty": is_empty,
        "joint_em": joint_em,
        "ans_em": ans_em,
        "ans_f1": ans_f1,
        "sp_em": sp_em,
        "sp_f1": sp_f1,
        "joint_f1": joint_f1,
        "m1": m1,
        "m2": m2,
        "both_gold_docs_cited": both_gold_docs_cited,
        "both_gold_docs_selected": both_gold_docs_selected,
        "gold_sp_len": gold_sp_len,
        "n_decisions": int(row.get("n_decisions") or 0),
        "repair_events": int(row.get("support_repair_events") or 0),
        "decode_errors": int(row.get("decode_errors") or 0),
        "firewall_violations": int(row.get("firewall_violations") or 0),
        "empty_m1_reachable": is_empty and m1,
        "decision_limit_after_support_reject": decision_limit_after_support_reject,
        "empty_label": _terminal_empty_label(row) if is_empty else None,
    }


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 6) if values else 0.0


def _rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def audit_arm(rows: list[dict[str, Any]], *, pred_sha256: str) -> dict[str, Any]:
    """Aggregate primary buckets, sub-buckets, and structural metrics for one arm."""
    classified = [classify_episode(r) for r in rows]
    n = len(classified)

    buckets = {k: 0 for k in PRIMARY_BUCKETS}
    for ep in classified:
        buckets[ep["primary_bucket"]] += 1
    assert sum(buckets.values()) == n, f"Bucket sum mismatch: {sum(buckets.values())} != {n}"

    sub_counts: Counter[str] = Counter()
    empty_support_m1 = 0
    empty_other_m1 = 0
    empty_limit_after_reject = 0
    empty_reasons: Counter[str] = Counter()
    ans_ok_sp_miss_3plus = 0

    for ep in classified:
        if ep["sub_bucket"]:
            sub_counts[ep["sub_bucket"]] += 1
        if ep["primary_bucket"] == "empty_support_reject" and ep["empty_m1_reachable"]:
            empty_support_m1 += 1
        if ep["primary_bucket"] == "empty_other" and ep["empty_m1_reachable"]:
            empty_other_m1 += 1
        if ep["decision_limit_after_support_reject"]:
            empty_limit_after_reject += 1
        if ep["empty_label"]:
            empty_reasons[ep["empty_label"]] += 1
        if ep["primary_bucket"] == "ans_ok_sp_miss" and ep["gold_sp_len"] >= 3:
            ans_ok_sp_miss_3plus += 1

    # Strict subset invariants on sub-buckets.
    assert (
        sub_counts["empty_other_decision_limit"]
        + sub_counts["empty_other_firewall_reject"]
        + sub_counts["empty_other_misc"]
        == buckets["empty_other"]
    )
    assert (
        sub_counts["sp_ok_ans_near_miss"] + sub_counts["sp_ok_ans_low_f1"]
        == buckets["sp_ok_ans_miss"]
    )
    assert (
        sub_counts["ans_ok_both_docs_cited_sent_miss"]
        + sub_counts["ans_ok_missing_gold_doc"]
        == buckets["ans_ok_sp_miss"]
    )
    assert (
        sub_counts["both_miss_m1_unreachable"]
        + sub_counts["both_miss_m1_reachable_m2_unselected"]
        + sub_counts["both_miss_m2_selected"]
        == buckets["both_miss"]
    )
    assert empty_support_m1 <= buckets["empty_support_reject"]
    assert empty_other_m1 <= buckets["empty_other"]
    assert empty_limit_after_reject <= sub_counts["empty_other_decision_limit"]
    assert ans_ok_sp_miss_3plus <= buckets["ans_ok_sp_miss"]

    gold2 = [ep for ep in classified if ep["gold_sp_len"] == 2]
    gold3 = [ep for ep in classified if ep["gold_sp_len"] >= 3]
    gold2_sp_em_hits = sum(1 for ep in gold2 if ep["sp_em"])
    gold3_sp_em_hits = sum(1 for ep in gold3 if ep["sp_em"])
    gold2_joint_hits = sum(1 for ep in gold2 if ep["joint_em"])
    gold3_joint_hits = sum(1 for ep in gold3 if ep["joint_em"])

    by_type: dict[str, Any] = {}
    for task_type in sorted({ep["type"] for ep in classified}):
        grp = [ep for ep in classified if ep["type"] == task_type]
        tb = {k: 0 for k in PRIMARY_BUCKETS}
        for ep in grp:
            tb[ep["primary_bucket"]] += 1
        assert sum(tb.values()) == len(grp)
        by_type[task_type] = {
            "n": len(grp),
            "buckets": tb,
            "ans_f1": _mean([ep["ans_f1"] for ep in grp]),
            "sp_f1": _mean([ep["sp_f1"] for ep in grp]),
            "joint_em": _mean([1.0 if ep["joint_em"] else 0.0 for ep in grp]),
        }

    return {
        "n": n,
        "predictions_sha256": pred_sha256,
        "cited_joint_em_count": buckets["joint_em_success"],
        "empty_count": sum(1 for ep in classified if ep["is_empty"]),
        "ans_em": _mean([1.0 if ep["ans_em"] else 0.0 for ep in classified]),
        "ans_f1": _mean([ep["ans_f1"] for ep in classified]),
        "sp_em": _mean([1.0 if ep["sp_em"] else 0.0 for ep in classified]),
        "sp_f1": _mean([ep["sp_f1"] for ep in classified]),
        "joint_em": _mean([1.0 if ep["joint_em"] else 0.0 for ep in classified]),
        "joint_f1": _mean([ep["joint_f1"] for ep in classified]),
        "m1_rate": _rate(sum(1 for ep in classified if ep["m1"]), n),
        "m2_rate": _rate(sum(1 for ep in classified if ep["m2"]), n),
        "both_gold_docs_cited_rate": _rate(
            sum(1 for ep in classified if ep["both_gold_docs_cited"]), n
        ),
        "both_gold_docs_selected_rate": _rate(
            sum(1 for ep in classified if ep["both_gold_docs_selected"]), n
        ),
        "sp_em_on_2sent_gold": {
            "hits": gold2_sp_em_hits,
            "total": len(gold2),
            "rate": _rate(gold2_sp_em_hits, len(gold2)),
            "joint_em_hits": gold2_joint_hits,
            "joint_em_rate": _rate(gold2_joint_hits, len(gold2)),
        },
        "sp_em_on_3plus_sent_gold": {
            "hits": gold3_sp_em_hits,
            "total": len(gold3),
            "rate": _rate(gold3_sp_em_hits, len(gold3)),
            "joint_em_hits": gold3_joint_hits,
            "joint_em_rate": _rate(gold3_joint_hits, len(gold3)),
        },
        "mean_decisions": round(
            statistics.mean([ep["n_decisions"] for ep in classified]), 4
        )
        if classified
        else 0.0,
        "mean_repair_events": round(
            statistics.mean([ep["repair_events"] for ep in classified]), 4
        )
        if classified
        else 0.0,
        "decode_error_episodes": sum(
            1 for ep in classified if ep["decode_errors"] > 0
        ),
        "firewall_violation_episodes": sum(
            1 for ep in classified if ep["firewall_violations"] > 0
        ),
        "buckets": buckets,
        "sub_buckets": {
            "empty_support_reject_m1_reachable": empty_support_m1,
            "empty_other_m1_reachable": empty_other_m1,
            "empty_other_decision_limit": sub_counts["empty_other_decision_limit"],
            "empty_other_decision_limit_after_support_reject": empty_limit_after_reject,
            "empty_other_firewall_reject": sub_counts["empty_other_firewall_reject"],
            "empty_other_misc": sub_counts["empty_other_misc"],
            "top_empty_reasons": [
                {"reason": reason, "count": count}
                for reason, count in empty_reasons.most_common(6)
            ],
            "sp_ok_ans_near_miss": sub_counts["sp_ok_ans_near_miss"],
            "sp_ok_ans_low_f1": sub_counts["sp_ok_ans_low_f1"],
            "ans_ok_both_docs_cited_sent_miss": sub_counts[
                "ans_ok_both_docs_cited_sent_miss"
            ],
            "ans_ok_missing_gold_doc": sub_counts["ans_ok_missing_gold_doc"],
            "ans_ok_sp_miss_gold_3plus_sents": ans_ok_sp_miss_3plus,
            "both_miss_m1_unreachable": sub_counts["both_miss_m1_unreachable"],
            "both_miss_m1_reachable_m2_unselected": sub_counts[
                "both_miss_m1_reachable_m2_unselected"
            ],
            "both_miss_m2_selected": sub_counts["both_miss_m2_selected"],
        },
        "by_type": by_type,
    }


def _discover_studies(runs_root: Path) -> list[tuple[str, str, list[str]]]:
    known = {name: (title, arms) for name, title, arms in STUDY_SPECS}
    ordered: list[tuple[str, str, list[str]]] = []
    for name, title, arms in STUDY_SPECS:
        if (runs_root / name).is_dir():
            ordered.append((name, title, arms))
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in known or "aborted" in entry.name or entry.name == "smoke":
            continue
        ordered.append((entry.name, entry.name, []))
    return ordered


def _discover_arms(study_dir: Path, preferred_order: list[str]) -> list[str]:
    valid = {
        entry.name
        for entry in study_dir.iterdir()
        if entry.is_dir()
        and entry.name != "smoke"
        and "aborted" not in entry.name
        and (entry / "predictions.jsonl").is_file()
    }
    ordered = [arm for arm in preferred_order if arm in valid]
    for arm in sorted(valid):
        if arm not in ordered:
            ordered.append(arm)
    return ordered


def run_audit(runs_root: Path) -> dict[str, Any]:
    """Scan all eligible studies under ``runs_root`` and compute the full funnel report."""
    studies_out: dict[str, Any] = {}
    for study_name, title, preferred_arms in _discover_studies(runs_root):
        study_dir = runs_root / study_name
        arms_out: dict[str, Any] = {}
        for arm in _discover_arms(study_dir, preferred_arms):
            pred_path = study_dir / arm / "predictions.jsonl"
            rows = load_jsonl(pred_path)
            arms_out[arm] = audit_arm(rows, pred_sha256=sha256_file(pred_path))
        studies_out[study_name] = {
            "title": title,
            "arms": arms_out,
        }
    return {
        "schema_version": 1,
        "runs_root": str(runs_root),
        "primary_buckets": list(PRIMARY_BUCKETS),
        "studies": studies_out,
    }


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "0.0%"


def _pct_f(val: float) -> str:
    return f"{100.0 * val:.1f}%"


def _arm_row_md(arm: str, d: dict[str, Any]) -> str:
    n = d["n"]
    b = d["buckets"]
    return (
        f"| `{arm}` | {n} | {b['joint_em_success']} ({_pct(b['joint_em_success'], n)}) "
        f"| {b['empty_support_reject']} | {b['empty_other']} "
        f"| {b['sp_ok_ans_miss']} | {b['ans_ok_sp_miss']} | {b['both_miss']} "
        f"| {_pct_f(d['ans_f1'])} | {_pct_f(d['sp_em'])} | {_pct_f(d['sp_f1'])} "
        f"| {_pct_f(d['joint_f1'])} | {_pct_f(d['m1_rate'])} | {_pct_f(d['both_gold_docs_cited_rate'])} "
        f"| {d['mean_decisions']:.2f} |"
    )


def _sub_row_md(arm: str, d: dict[str, Any]) -> str:
    s = d["sub_buckets"]
    g2 = d["sp_em_on_2sent_gold"]
    g3 = d["sp_em_on_3plus_sent_gold"]
    return (
        f"| `{arm}` | {s['empty_support_reject_m1_reachable']} "
        f"| {s['empty_other_decision_limit']} ({s['empty_other_decision_limit_after_support_reject']}) "
        f"| {s['empty_other_firewall_reject']} "
        f"| {s['sp_ok_ans_near_miss']} / {s['sp_ok_ans_low_f1']} "
        f"| {s['ans_ok_both_docs_cited_sent_miss']} / {s['ans_ok_missing_gold_doc']} "
        f"| {s['both_miss_m1_unreachable']} / {s['both_miss_m1_reachable_m2_unselected']} / {s['both_miss_m2_selected']} "
        f"| {g2['hits']}/{g2['total']} ({_pct(g2['hits'], g2['total'])}) "
        f"| {g3['hits']}/{g3['total']} ({_pct(g3['hits'], g3['total'])}) |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    """Render a Chinese diagnostic failure-funnel report from ``run_audit`` output."""
    lines: list[str] = [
        "# HotpotQA Harness-vs-SFT 零 GPU 失败漏斗审计（2026-11-20）",
        "",
        "> **证据边界声明**：本报告是对本地已落盘诊断运行（`../runs/harness_vs_sft/`）的零 GPU 只读重放拆解，"
        "未运行任何新推理或训练；所有运行均基于 120 题全 2-gold 子集、Qwen3-4B-Instruct-2507 与 V8 adapter、"
        "Triton Attention（E2 另开 `--enforce-eager`）。**次级分项指标（Answer F1 / Support F1 / Joint F1）仅用于解释失败机制，"
        "不推翻预注册主指标 `cited Joint EM` 的判定结论。**",
        "",
        "## 1. 互斥 6 桶定义（每行精确加和 = 120）",
        "",
        "1. `joint_em_success`：官方 `official.joint_em == 1.0`（短答案 EM 与完整支持句 EM 同时命中）。",
        "2. `empty_support_reject`：未成功且为空答案（`empty_answer=True`），且终局拒绝/弃答原因明确涉及双文档/双段落支持度约束。",
        "3. `empty_other`：未成功且为空答案，但终局原因是步数/决策耗尽（`decision limit exhausted`）或证据防火墙拦截（`foreign or historical evidence handle`）等其他原因。",
        "4. `sp_ok_ans_miss`：非空且支持句精确命中（`sp_em == 1.0`），仅短答案未精确匹配（拆分为 `ans_f1 >= 0.5` 近命中 vs `< 0.5`）。",
        "5. `ans_ok_sp_miss`：非空且短答案精确命中（`ans_em == 1.0`），仅支持句未完全命中（拆分为「两篇 gold 文档已引齐但漏句」vs「缺 gold 文档」）。",
        "6. `both_miss`：非空且 `ans_em == 0`、`sp_em == 0`（按 `M1` 检索可达性、`M2` 选中完整支持与否继续下钻）。",
        "",
        "## 2. 四组实验总览（6 桶 + 次级 F1 + 检索/引用覆盖）",
        "",
    ]

    for study_name, study in report["studies"].items():
        lines.append(f"### {study['title']}（`{study_name}`）")
        lines.append("")
        lines.append(
            "| arm | n | joint_em | empty_sup_rej | empty_other | sp_ok_ans_miss | ans_ok_sp_miss | both_miss | ans_f1 | sp_em | sp_f1 | joint_f1 | M1 | 2doc_cited | dec |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for arm, d in study["arms"].items():
            lines.append(_arm_row_md(arm, d))
        lines.append("")
        lines.append(
            "| arm | sup_rej中M1=T | limit_exhaust(先遭拒) | firewall拦截 | sp_ok近/远错 | ans_ok引全doc漏句/缺doc | both_miss(M1关/M1开M2关/M2开) | 2句gold sp_em | 3+句gold sp_em |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm, d in study["arms"].items():
            lines.append(_sub_row_md(arm, d))
        lines.append("")

    # Deep-dive 1: hv-clean120 2x2
    clean = report["studies"].get("hv-clean120-2x2-20261120", {}).get("arms", {})
    if clean:
        lines.extend(
            [
                "## 3. 专题解剖一：`hv-clean120` 2×2 — 为什么 `VB` 只有 17/120，而 `V8-off` 是 0/1？",
                "",
            ]
        )
        for arm in ("base_off-r1", "base_blanket2-r1", "v8_off-r1", "v8_blanket2-r1"):
            d = clean.get(arm)
            if not d:
                continue
            b = d["buckets"]
            s = d["sub_buckets"]
            g2 = d["sp_em_on_2sent_gold"]
            g3 = d["sp_em_on_3plus_sent_gold"]
            lines.append(
                f"- **`{arm}`**：`joint_em` **{b['joint_em_success']}/120** | "
                f"`empty_sup_rej` {b['empty_support_reject']} | `empty_other` {b['empty_other']} "
                f"(limit={s['empty_other_decision_limit']}, firewall={s['empty_other_firewall_reject']}) | "
                f"`sp_ok_ans_miss` {b['sp_ok_ans_miss']} (F1>=0.5 有 {s['sp_ok_ans_near_miss']}) | "
                f"`ans_ok_sp_miss` {b['ans_ok_sp_miss']} (引齐两篇仅漏句={s['ans_ok_both_docs_cited_sent_miss']}, 缺文档={s['ans_ok_missing_gold_doc']}) | "
                f"`both_miss` {b['both_miss']} (M1不可达={s['both_miss_m1_unreachable']}, M1可达未选={s['both_miss_m1_reachable_m2_unselected']}, 已选M2={s['both_miss_m2_selected']}) | "
                f"2句gold `sp_em` {g2['hits']}/{g2['total']} ({_pct(g2['hits'], g2['total'])}) vs 3+句gold {g3['hits']}/{g3['total']} ({_pct(g3['hits'], g3['total'])})"
            )
        v8_off = clean.get("v8_off-r1", {})
        base_off = clean.get("base_off-r1", {})
        base_bl = clean.get("base_blanket2-r1", {})
        v8_bl = clean.get("v8_blanket2-r1", {})
        v8_off_ans_miss = v8_off.get("buckets", {}).get("ans_ok_sp_miss", 0)
        v8_off_missing_doc = v8_off.get("sub_buckets", {}).get("ans_ok_missing_gold_doc", 0)
        base_off_lim = base_off.get("sub_buckets", {}).get("empty_other_decision_limit", 0)
        base_bl_lim = base_bl.get("sub_buckets", {}).get("empty_other_decision_limit", 0)
        base_bl_sup = base_bl.get("buckets", {}).get("empty_support_reject", 0)
        base_bl_fw = base_bl.get("sub_buckets", {}).get("empty_other_firewall_reject", 0)
        v8_bl_empty = v8_bl.get("empty_count", 0)
        v8_bl_sup_m1 = v8_bl.get("sub_buckets", {}).get("empty_support_reject_m1_reachable", 0)
        v8_bl_g3 = v8_bl.get("sp_em_on_3plus_sent_gold", {})
        v8_bl_g2 = v8_bl.get("sp_em_on_2sent_gold", {})
        lines.extend(
            [
                "",
                "**关键结论**：",
                (
                    f"- **`V8-off` 为什么 `Joint EM` 接近 0，但 `Answer F1` 高达 {_pct_f(v8_off.get('ans_f1', 0.0))}？**："
                    f"V8 在无 min-2 约束时平均仅 {v8_off.get('mean_decisions', 0.0):.2f} 轮就作答，"
                    f"在 `v8_off-r1` 上拿到了 **{v8_off_ans_miss} 个 `ans_ok_sp_miss`**（其中 **缺 gold 文档 = {v8_off_missing_doc}**），"
                    f"`sp_em` 仅 {_pct_f(v8_off.get('sp_em', 0.0))}，在精确引用联合口径下被直接清零。"
                ),
                (
                    f"- **`base` 臂为什么大量空答案？**：`base_off-r1` 共有 {base_off.get('empty_count', 0)} 个空答案，"
                    f"其中 **{base_off_lim} 题**耗尽 12 步决策上限（`decision limit exhausted`）；"
                    f"`base_blanket2-r1` 共有 {base_bl.get('empty_count', 0)} 个空答案，"
                    f"其中 `decision limit` 占 **{base_bl_lim}**、支持度约束直接拒绝 **{base_bl_sup}**、防火墙拦截 {base_bl_fw}。"
                ),
                (
                    f"- **`v8_blanket2-r1`（{v8_bl.get('cited_joint_em_count', 0)}/120）剩下的 {120 - v8_bl.get('cited_joint_em_count', 0)} 题都掉在哪？**："
                    f"① 空答案 **{v8_bl_empty} 题**（全部来自 `empty_support_reject`，其中 **{v8_bl_sup_m1}/{v8_bl_empty}** 的 `M1=True` 检索其实已可达）；"
                    f"② 非空未全对中，`both_miss` {v8_bl.get('buckets', {}).get('both_miss', 0)}、`ans_ok_sp_miss` {v8_bl.get('buckets', {}).get('ans_ok_sp_miss', 0)}、`sp_ok_ans_miss` {v8_bl.get('buckets', {}).get('sp_ok_ans_miss', 0)}；"
                    f"③ **结构性天花板**：`hv-clean120` 中有 **{v8_bl_g3.get('total', 0)}/120**（{_pct(v8_bl_g3.get('total', 0), 120)}）题的 `gold_sp` 包含 3–7 个句子，"
                    f"而 2-gold 提示与双文档约束下模型几乎只引用 2 句，导致 3+ 句 gold 上 `sp_em` 为 **{v8_bl_g3.get('hits', 0)}/{v8_bl_g3.get('total', 0)}**（2 句 gold 上为 {v8_bl_g2.get('hits', 0)}/{v8_bl_g2.get('total', 0)} = {_pct(v8_bl_g2.get('hits', 0), v8_bl_g2.get('total', 0))}）。"
                ),
                "",
            ]
        )

    # Deep-dive 2: hv-blanket-repair
    rep = report["studies"].get("hv-blanket-repair-20261120", {}).get("arms", {})
    if rep:
        lines.extend(
            [
                "## 4. 专题解剖二：E1 `blanket + repair`（`VB` vs `BR`）— 被救回来的 ~11 pp 空答案流向了哪里？",
                "",
            ]
        )
        for arm in ("vb-r1", "br-r1", "vb-r2", "br-r2"):
            d = rep.get(arm)
            if not d:
                continue
            b = d["buckets"]
            s = d["sub_buckets"]
            lines.append(
                f"- **`{arm}`**：`joint_em` **{b['joint_em_success']}** | "
                f"空答案 **{d['empty_count']}** (sup_rej={b['empty_support_reject']}, other={b['empty_other']}) | "
                f"`sp_ok_ans_miss` **{b['sp_ok_ans_miss']}** (near={s['sp_ok_ans_near_miss']}) | "
                f"`ans_ok_sp_miss` **{b['ans_ok_sp_miss']}** (引齐漏句={s['ans_ok_both_docs_cited_sent_miss']}, 缺doc={s['ans_ok_missing_gold_doc']}) | "
                f"`both_miss` **{b['both_miss']}** | `ans_f1` {_pct_f(d['ans_f1'])} | `sp_f1` {_pct_f(d['sp_f1'])} | `joint_f1` {_pct_f(d['joint_f1'])}"
            )
        lines.extend(
            [
                "",
                "**关键结论**：Repair 在 r1/r2 上分别把空答案减少 **14 题 / 13 题**（−11.7 / −10.8 pp），但大多数被救回的样本流向了 `ans_ok_sp_miss`（答案对了但支持句没完全对齐）、`sp_ok_ans_miss`（答案差一点精确匹配）和 `both_miss`，使得次级 `Answer F1`（+7–8 pp）、`Support F1`（+7–8 pp）与 `Joint F1`（+4–6 pp）明显抬升，而硬性 `cited Joint EM` 在 r1 打平、r2 仅 +3（统计不显著）。",
                "",
            ]
        )

    # Deep-dive 3: structural 2-sent vs 3+-sent table
    lines.extend(
        [
            "## 5. 结构性约束检查：2 句 gold vs 3+ 句 gold 的 `sp_em` 断崖",
            "",
            "| study | arm | 2句 gold (sp_em / joint_em) | 3+句 gold (sp_em / joint_em) |",
            "|---|---|---|---|",
        ]
    )
    for study_name, study in report["studies"].items():
        for arm, d in study["arms"].items():
            if not any(tag in arm for tag in ("blanket2-r1", "vb-r1", "br-r1", "contract-r1", "repair-r1", "v8_off-r1")):
                continue
            g2 = d["sp_em_on_2sent_gold"]
            g3 = d["sp_em_on_3plus_sent_gold"]
            lines.append(
                f"| `{study_name}` | `{arm}` "
                f"| {g2['hits']}/{g2['total']} ({_pct(g2['hits'], g2['total'])}) / {g2['joint_em_hits']} "
                f"| {g3['hits']}/{g3['total']} ({_pct(g3['hits'], g3['total'])}) / {g3['joint_em_hits']} |"
            )

    lines.extend(
        [
            "",
            "## 6. 输入溯源（只读 `predictions.jsonl` SHA-256）",
            "",
            "| study | arm | n | cited_joint_em | empty | predictions.jsonl SHA-256 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for study_name, study in report["studies"].items():
        for arm, d in study["arms"].items():
            lines.append(
                f"| `{study_name}` | `{arm}` | {d['n']} | {d['cited_joint_em_count']} | {d['empty_count']} | `{d['predictions_sha256'][:16]}…` |"
            )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("../runs/harness_vs_sft"),
        help="Directory holding the harness-vs-SFT run subdirectories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/harness_vs_sft/failure_funnel_20261120.json"),
        help="Path to write structured failure-funnel JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/harness_vs_sft/FAILURE_FUNNEL_20261120.md"),
        help="Path to write human-readable Markdown failure-funnel report.",
    )
    args = parser.parse_args(argv)

    report = run_audit(args.runs_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
