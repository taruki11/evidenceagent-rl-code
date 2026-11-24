#!/usr/bin/env python3
"""Generate publication-quality, zero-dependency SVG figures for EvidenceAgent-RL Paper v3.

All numbers are taken verbatim from the frozen Phase 2-6C audit documents:
- Block A: docs/harness_vs_sft/RESULTS_HV_MIXED180_BLOCKA_20261120.md
- Phase 3B (3-class V1): docs/harness_vs_sft/RESULTS_HV_MIXED180_VERIFIER_20261121.md
- Phase 3C (3-class V2): docs/harness_vs_sft/RESULTS_HV_MIXED180_VERIFIER_V2_20261121.md
- Phase 3D (Binary dev): docs/harness_vs_sft/RESULTS_HV_MIXED180_VERIFIER_BINARY_20261122.md
- Phase 4 (Blind held-out v1): docs/harness_vs_sft/RESULTS_HV_MIXED180_HELDOUT_20261122.md
- Phase 5 (Cross-model Gemma): docs/harness_vs_sft/RESULTS_HV_MIXED180_HELDOUT_CROSSMODEL_20261123.md
- Phase 6 (Controller held-out v2): docs/harness_vs_sft/RESULTS_HV_MIXED180_HELDOUT_V2_CONTROLLER_20261123.md
- Phase 6C (Mechanism analysis): docs/harness_vs_sft/ANALYSIS_PHASE6_MECHANISM_20261124.md
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper_v3_draft" / "figures"

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">\n'
        f"<defs>\n"
        f"  <style>\n"
        f"    text {{ font-family: {FONT_STACK}; fill: #1f2937; }}\n"
        f"    .title {{ font-size: 16px; font-weight: 700; }}\n"
        f"    .subtitle {{ font-size: 12px; fill: #4b5563; }}\n"
        f"    .panel-title {{ font-size: 13.5px; font-weight: 700; fill: #111827; }}\n"
        f"    .axis-label {{ font-size: 11.5px; font-weight: 600; fill: #374151; }}\n"
        f"    .tick {{ font-size: 10.5px; fill: #4b5563; }}\n"
        f"    .val {{ font-size: 10px; font-weight: 600; fill: #111827; }}\n"
        f"    .legend {{ font-size: 11px; fill: #374151; }}\n"
        f"    .note {{ font-size: 10px; fill: #6b7280; }}\n"
        f"  </style>\n"
        f"</defs>\n"
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>\n'
    )


def generate_fig1_heldout_main() -> str:
    """Fig 1: Blind Held-Out (hv-mixed180-heldout, Rep 1 primary) 3-Regime Bars + Pareto Frontier."""
    w, h = 1040, 460
    parts = [svg_header(w, h, "Blind Held-Out Evaluation: Three-Regime Tradeoff and Pareto Frontier")]
    parts.append(
        '<text x="32" y="34" class="title">Figure 1. Blind Held-Out Evaluation (hv-mixed180-heldout, 180 Tasks, Rep 1 Primary)</text>'
    )
    parts.append(
        '<text x="32" y="54" class="subtitle">'
        "Frozen Qwen3-4B + V8 with zero-shot Binary Verifier (568M BF16, 16.2ms P95). "
        "Vh-bin breaks the three-regime triangle (+72% Macro Utility vs. C2, -33% Selective Risk)."
        "</text>"
    )

    # Left panel: Grouped bars for 3 regimes
    # Arms: A0, A1, A2, C2, Vh-bin, Va-bin
    arms = [
        ("A0 (off)", 36 / 60, 0 / 60, 0.0000, 0.2000, 0.9500, "#9ca3af"),
        ("A1 (min-2)", 0 / 60, 10 / 60, 0.3214, 0.1627, 1.2833, "#ef4444"),
        ("A2 (contract)", 37 / 60, 4 / 60, 0.0000, 0.2278, 0.5333, "#f59e0b"),
        ("C2 (+abstain)", 36 / 60, 4 / 60, 0.0625, 0.2431, 0.5000, "#8b5cf6"),
        ("Vh-bin (Ours)", 34 / 60, 6 / 60, 0.6034, 0.4234, 0.3500, "#059669"),
        ("Va-bin (Auto)", 33 / 60, 5 / 60, 0.5667, 0.4000, 0.3500, "#0284c7"),
    ]

    lx0, ly0, lw, lh = 62, 102, 470, 275
    parts.append(f'<text x="{lx0}" y="86" class="panel-title">(a) Per-Regime Breakdown (Cited Joint EM &amp; Abstention F1)</text>')

    # Y gridlines 0.0 to 0.70
    y_max = 0.70
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        yy = ly0 + lh - (tick / y_max) * lh
        stroke = "#d1d5db" if tick > 0 else "#374151"
        dash = ' stroke-dasharray="3,3"' if tick > 0 else ""
        parts.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx0 + lw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"{dash}/>')
        parts.append(f'<text x="{lx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly0 + lh}" stroke="#374151" stroke-width="1"/>')

    bar_colors = ["#2563eb", "#d97706", "#059669"]
    group_w = lw / len(arms)
    bw = 17
    for idx, (label, r1, r2, r0, _, _, _) in enumerate(arms):
        gx = lx0 + idx * group_w + 13
        vals = [r1, r2, r0]
        for j, v in enumerate(vals):
            bh = (v / y_max) * lh
            bx = gx + j * (bw + 2)
            by = ly0 + lh - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{bar_colors[j]}" rx="2"/>')
            if v >= 0.05:
                parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" class="val">{v:.2f}</text>')
        cx = gx + 1.5 * bw + 2
        font_w = "700" if "Vh-bin" in label else "400"
        parts.append(f'<text x="{cx:.1f}" y="{ly0 + lh + 18}" text-anchor="middle" class="tick" font-weight="{font_w}">{escape(label)}</text>')

    # Legend for panel (a)
    leg_items = [("1-gold Cited EM", bar_colors[0]), ("2-gold Cited EM", bar_colors[1]), ("0-gold Abstention F1", bar_colors[2])]
    for i, (lname, lcol) in enumerate(leg_items):
        xx = lx0 + i * 155
        parts.append(f'<rect x="{xx}" y="{ly0 + lh + 38}" width="11" height="11" fill="{lcol}" rx="2"/>')
        parts.append(f'<text x="{xx + 16}" y="{ly0 + lh + 47}" class="legend">{lname}</text>')

    # Right panel: Pareto scatter (Selective Risk [lower=better] vs Macro Utility [higher=better])
    rx0, ry0, rw, rh = 605, 102, 395, 275
    parts.append(f'<text x="{rx0}" y="86" class="panel-title">(b) Selective Risk vs. Macro-Regime Utility (Pareto)</text>')

    # Highlight "better" upper-left corner
    parts.append(
        f'<rect x="{rx0}" y="{ry0}" width="{rw * 0.42:.1f}" height="{rh * 0.42:.1f}" fill="#ecfdf5" opacity="0.75"/>'
    )
    parts.append(
        f'<text x="{rx0 + 10}" y="{ry0 + 18}" class="note" fill="#047857" font-weight="700">Better Region (High Utility, Low Risk)</text>'
    )

    x_min, x_max = 0.20, 1.40
    y2_min, y2_max = 0.10, 0.50
    for xt in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        xx = rx0 + ((xt - x_min) / (x_max - x_min)) * rw
        parts.append(f'<line x1="{xx:.1f}" y1="{ry0}" x2="{xx:.1f}" y2="{ry0 + rh}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{xx:.1f}" y="{ry0 + rh + 16}" text-anchor="middle" class="tick">{xt:.1f}</text>')
    for yt in [0.1, 0.2, 0.3, 0.4, 0.5]:
        yy = ry0 + rh - ((yt - y2_min) / (y2_max - y2_min)) * rh
        parts.append(f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx0 + rw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{rx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{yt:.2f}</text>')
    parts.append(f'<rect x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" fill="none" stroke="#374151" stroke-width="1"/>')

    # Axis titles
    parts.append(f'<text x="{rx0 + rw / 2:.1f}" y="{ry0 + rh + 36}" text-anchor="middle" class="axis-label">Selective Risk (Lower is Better)</text>')
    parts.append(
        f'<text x="{rx0 - 44}" y="{ry0 + rh / 2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 {rx0 - 44} {ry0 + rh / 2:.1f})">Macro-Regime Utility</text>'
    )

    # Plot points + smart label offsets
    offsets = {
        "A0 (off)": (0, -14),
        "A1 (min-2)": (-12, -14),
        "A2 (contract)": (10, 16),
        "C2 (+abstain)": (10, -12),
        "Vh-bin (Ours)": (14, -10),
        "Va-bin (Auto)": (14, 16),
    }
    for label, _, _, _, macro, risk, col in arms:
        px = rx0 + ((risk - x_min) / (x_max - x_min)) * rw
        py = ry0 + rh - ((macro - y2_min) / (y2_max - y2_min)) * rh
        r = 7.5 if "Vh-bin" in label else 5.5
        stroke = "#064e3b" if "Vh-bin" in label else "#ffffff"
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{col}" stroke="{stroke}" stroke-width="1.8"/>')
        dx, dy = offsets[label]
        anchor = "end" if dx < 0 else ("middle" if dx == 0 else "start")
        fw = "700" if "Vh-bin" in label else "500"
        parts.append(
            f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}" class="val" font-weight="{fw}">'
            f"{escape(label)} ({macro:.2f})</text>"
        )

    parts.append(
        f'<text x="32" y="{h - 14}" class="note">'
        "Source: RESULTS_HV_MIXED180_HELDOUT_20261122.md (Rep 2 reproduces identically: Vh-bin Macro 0.4236, Risk 0.3333, 1-gold 34/60, 0-gold Abst F1 0.6207)."
        "</text>"
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def generate_fig2_unidentifiability() -> str:
    """Fig 2: 3-Class Unidentifiability Deadlock (V1 vs V2) vs. Decoupled Binary Verifier."""
    w, h = 1040, 420
    parts = [svg_header(w, h, "Partial Observability Unidentifiability vs Decoupled Binary Sufficiency")]
    parts.append(
        '<text x="32" y="34" class="title">Figure 2. Why 3-Class Verification Deadlocks — and Why Binary Decoupling Succeeds (Dev, n=500)</text>'
    )
    parts.append(
        '<text x="32" y="54" class="subtitle">'
        "Under partial observability, NEED_MORE vs. UNANSWERABLE is unidentifiable from 1 span + titles; "
        "decoupling SUFFICIENT vs. INSUFFICIENT reaches 0.962 macro-F1."
        "</text>"
    )

    # Panel (a): 3-Class V1 vs V2 tradeoff bars
    lx0, ly0, lw, lh = 65, 102, 420, 240
    parts.append(f'<text x="{lx0}" y="86" class="panel-title">(a) 3-Class Deadlock: False-Abstain vs. Recall Collapse</text>')
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yy = ly0 + lh - tick * lh
        parts.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx0 + lw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{lx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly0 + lh}" stroke="#374151" stroke-width="1"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0 + lh}" x2="{lx0 + lw}" y2="{ly0 + lh}" stroke="#374151" stroke-width="1"/>')

    # Gate line at 0.70
    gy = ly0 + lh - 0.70 * lh
    parts.append(f'<line x1="{lx0}" y1="{gy:.1f}" x2="{lx0 + lw}" y2="{gy:.1f}" stroke="#dc2626" stroke-width="1.4" stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{lx0 + lw - 4}" y="{gy - 5:.1f}" text-anchor="end" class="note" fill="#dc2626" font-weight="700">Pre-reg Gate (0.70)</text>')

    groups = [
        ("V1 (tau_unans=0.15)", [("Dev Macro-F1", 0.8140, "#3b82f6"), ("UNANS F1", 0.7515, "#8b5cf6"), ("NEED->UNANS Err", 0.2700, "#ef4444")]),
        ("V2 (tau_unans=0.80)", [("Dev Macro-F1", 0.7824, "#3b82f6"), ("UNANS F1", 0.6265, "#8b5cf6"), ("NEED->UNANS Err", 0.1000, "#ef4444")]),
    ]
    bw = 38
    for g_idx, (glabel, items) in enumerate(groups):
        gx = lx0 + 40 + g_idx * 200
        for j, (_, val, col) in enumerate(items):
            bh = val * lh
            bx = gx + j * (bw + 8)
            by = ly0 + lh - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" rx="3"/>')
            parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle" class="val">{val:.3f}</text>')
        parts.append(f'<text x="{gx + 1.5 * bw + 8:.1f}" y="{ly0 + lh + 20}" text-anchor="middle" class="axis-label">{escape(glabel)}</text>')

    # Legend (a)
    for i, (lname, lcol) in enumerate([("Dev Macro-F1", "#3b82f6"), ("UNANSWERABLE F1", "#8b5cf6"), ("NEED_MORE->UNANS Error (d=1)", "#ef4444")]):
        xx = lx0 + i * 145
        parts.append(f'<rect x="{xx}" y="{ly0 + lh + 38}" width="11" height="11" fill="{lcol}" rx="2"/>')
        parts.append(f'<text x="{xx + 15}" y="{ly0 + lh + 47}" class="legend">{lname}</text>')

    # Panel (b): Binary Verifier performance across subsets
    rx0, ry0, rw, rh = 565, 102, 435, 240
    parts.append(f'<text x="{rx0}" y="86" class="panel-title">(b) Decoupled Binary Verifier (SUFFICIENT vs. INSUFFICIENT)</text>')
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yy = ry0 + rh - tick * rh
        parts.append(f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx0 + rw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{rx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry0 + rh}" stroke="#374151" stroke-width="1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0 + rh}" x2="{rx0 + rw}" y2="{ry0 + rh}" stroke="#374151" stroke-width="1"/>')

    gy2 = ry0 + rh - 0.85 * rh
    parts.append(f'<line x1="{rx0}" y1="{gy2:.1f}" x2="{rx0 + rw}" y2="{gy2:.1f}" stroke="#059669" stroke-width="1.4" stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{rx0 + rw - 4}" y="{gy2 - 5:.1f}" text-anchor="end" class="note" fill="#047857" font-weight="700">Binary Gate (0.85)</text>')

    bin_bars = [
        ("Overall\nMacro-F1", 0.9616, "#059669"),
        ("distinct=1\nMacro-F1", 0.9868, "#10b981"),
        ("distinct=2\nMacro-F1", 0.9299, "#14b8a6"),
        ("SUFFICIENT\nF1", 0.9459, "#0284c7"),
        ("INSUFFICIENT\nF1", 0.9773, "#2563eb"),
    ]
    bw2 = 52
    step = rw / len(bin_bars)
    for i, (blabel, bval, bcol) in enumerate(bin_bars):
        bx = rx0 + i * step + (step - bw2) / 2
        bh = bval * rh
        by = ry0 + rh - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw2}" height="{bh:.1f}" fill="{bcol}" rx="4"/>')
        parts.append(f'<text x="{bx + bw2 / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" class="val">{bval:.4f}</text>')
        lines = blabel.split("\n")
        for li, line in enumerate(lines):
            parts.append(f'<text x="{bx + bw2 / 2:.1f}" y="{ry0 + rh + 16 + li * 13}" text-anchor="middle" class="tick">{escape(line)}</text>')

    parts.append(
        f'<text x="{rx0}" y="{ry0 + rh + 48}" class="note" fill="#047857" font-weight="600">'
        "Calibration &amp; Speed: ECE = 0.0045 (<= 0.15) | BF16 P50 = 14.50 ms, P95 = 16.57 ms (<= 30 ms)"
        "</text>"
    )
    parts.append(
        f'<text x="32" y="{h - 14}" class="note">'
        "Source: verifier_phase3b_20261121/metrics.json, verifier_phase3c_20261121/metrics.json, RESULTS_HV_MIXED180_VERIFIER_BINARY_20261122.md."
        "</text>"
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def generate_fig3_progression() -> str:
    """Fig 3: Six-Phase Progression from Block A to Blind Held-Out v1 & v2 (Macro Utility & Selective Risk)."""
    w, h = 1040, 410
    parts = [svg_header(w, h, "Scientific Progression from Development to Dual-Model Blind Held-Out")]
    parts.append(
        '<text x="32" y="34" class="title">Figure 3. Controlled Scientific Progression: Static Baselines -&gt; 3-Class -&gt; Binary -&gt; Controller Held-Out v2</text>'
    )
    parts.append(
        '<text x="32" y="54" class="subtitle">'
        "Rep 1 / Rep 2 mean Macro-Regime Utility (higher is better) and Selective Risk (lower is better) across frozen phases."
        "</text>"
    )

    stages = [
        ("Block A\nA0 (off)", 0.2139, 1.0083),
        ("Block A\nA2 (contract)", 0.2528, 0.5333),
        ("Phase 3B\nVh (3-class)", 0.3106, 0.4333),
        ("Phase 3D\nVh-bin (Dev)", 0.4905, 0.2167),
        ("Phase 4\nHeld-out v1", 0.4235, 0.3417),
        ("Phase 6\nQwen4 v2", 0.4641, 0.2917),
        ("Phase 6\nGemma12B v2", 0.3520, 0.1333),
    ]

    x0, y0, pw, ph = 84, 92, 872, 230
    # Left Y axis: Macro Utility [0.0, 0.60]; Right Y axis: Selective Risk [0.0, 1.50]
    for i in range(7):
        frac = i / 6.0
        yy = y0 + ph - frac * ph
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{x0 - 10}" y="{yy + 4:.1f}" text-anchor="end" class="tick" fill="#059669">{frac * 0.60:.2f}</text>')
        parts.append(f'<text x="{x0 + pw + 10}" y="{yy + 4:.1f}" text-anchor="start" class="tick" fill="#dc2626">{frac * 1.50:.2f}</text>')
    parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#374151" stroke-width="1"/>')

    # Vertical separator before Held-out
    sep_x = x0 + (3.5 / (len(stages) - 1)) * pw
    parts.append(f'<line x1="{sep_x:.1f}" y1="{y0}" x2="{sep_x:.1f}" y2="{y0 + ph}" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{sep_x + 8:.1f}" y="{y0 + 16}" class="note" font-weight="700" fill="#111827">Blind Held-Out (v1 &amp; Cross-Model v2)</text>')
    parts.append(f'<text x="{sep_x - 8:.1f}" y="{y0 + 16}" text-anchor="end" class="note" font-weight="700" fill="#4b5563">Dev (hv-mixed180)</text>')

    pts_m, pts_r = [], []
    for idx, (slabel, macro, risk) in enumerate(stages):
        px = x0 + (idx / (len(stages) - 1)) * pw
        pm_y = y0 + ph - (macro / 0.60) * ph
        pr_y = y0 + ph - (risk / 1.50) * ph
        pts_m.append((px, pm_y, macro))
        pts_r.append((px, pr_y, risk))
        for li, line in enumerate(slabel.split("\n")):
            fw = "700" if ("Vh-bin" in slabel or "v2" in slabel or "Held-out" in slabel) else "400"
            parts.append(f'<text x="{px:.1f}" y="{y0 + ph + 18 + li * 13}" text-anchor="middle" class="tick" font-weight="{fw}">{escape(line)}</text>')

    parts.append(
        '<polyline fill="none" stroke="#059669" stroke-width="2.6" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts_m)
        + '"/>'
    )
    parts.append(
        '<polyline fill="none" stroke="#dc2626" stroke-width="2.4" stroke-dasharray="6,3" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts_r)
        + '"/>'
    )

    for px, py, val in pts_m:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#059669" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(f'<text x="{px:.1f}" y="{py - 9:.1f}" text-anchor="middle" class="val" fill="#047857">{val:.3f}</text>')
    for px, py, val in pts_r:
        parts.append(f'<rect x="{px - 4.2:.1f}" y="{py - 4.2:.1f}" width="8.4" height="8.4" fill="#dc2626" stroke="#ffffff" stroke-width="1.3"/>')
        parts.append(f'<text x="{px:.1f}" y="{py + 17:.1f}" text-anchor="middle" class="val" fill="#b91c1c">{val:.3f}</text>')

    # Axis labels
    parts.append(f'<text x="20" y="{y0 + ph / 2:.1f}" text-anchor="middle" class="axis-label" fill="#047857" transform="rotate(-90 20 {y0 + ph / 2:.1f})">Macro-Regime Utility (↑)</text>')
    parts.append(f'<text x="{w - 18}" y="{y0 + ph / 2:.1f}" text-anchor="middle" class="axis-label" fill="#b91c1c" transform="rotate(90 {w - 18} {y0 + ph / 2:.1f})">Selective Risk (↓)</text>')

    # Legend
    parts.append(f'<circle cx="{x0 + 180}" cy="{h - 24}" r="5" fill="#059669"/>')
    parts.append(f'<text x="{x0 + 192}" y="{h - 20}" class="legend">Macro-Regime Utility (Rep 1/2 mean, ↑)</text>')
    parts.append(f'<rect x="{x0 + 475}" y="{h - 28.5}" width="9" height="9" fill="#dc2626"/>')
    parts.append(f'<text x="{x0 + 490}" y="{h - 20}" class="legend">Selective Risk (Rep 1/2 mean, ↓)</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def generate_fig4_crossmodel_phase6() -> str:
    """Fig 4: Phase 6 Cross-Model Blind Held-Out v2 (Qwen4+V8 & Gemma-3-12B) + Abstention Ownership."""
    w, h = 1040, 440
    parts = [svg_header(w, h, "Phase 6 Cross-Model Transfer and Controller-Owned Abstention Breakdown")]
    parts.append(
        '<text x="32" y="34" class="title">Figure 4. Cross-Model Blind Held-Out v2 (Phase 6): Zero-Shot Transfer &amp; Abstention Ownership</text>'
    )
    parts.append(
        '<text x="32" y="54" class="subtitle">'
        "Same frozen 568M BF16 Binary Verifier + Controller-Owned Abstention passes C-H1/C-H2/C-H3 on both Qwen3-4B+V8 and Gemma-3-12B."
        "</text>"
    )

    # Panel (a): Macro Utility comparison across A0 / A1 / A2 / Vh-ctrl for both models (Rep 1/2 mean)
    lx0, ly0, lw, lh = 65, 102, 440, 250
    parts.append(f'<text x="{lx0}" y="86" class="panel-title">(a) Macro-Regime Utility on Fresh Held-Out v2 (Rep 1/2 Mean)</text>')
    y_max = 0.55
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        yy = ly0 + lh - (tick / y_max) * lh
        parts.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx0 + lw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{lx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly0 + lh}" stroke="#374151" stroke-width="1"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0 + lh}" x2="{lx0 + lw}" y2="{ly0 + lh}" stroke="#374151" stroke-width="1"/>')

    model_groups = [
        ("Qwen3-4B + V8 (4B SFT)", [0.2195, 0.1561, 0.2361, 0.4641]),
        ("Gemma-3-12B (Cross-Vendor, 0-SFT)", [0.1833, 0.0084, 0.1889, 0.3520]),
    ]
    arm_colors = ["#9ca3af", "#ef4444", "#f59e0b", "#059669"]
    arm_names = ["A0 (off)", "A1 (min-2)", "A2 (contract)", "Vh-ctrl (Ours)"]
    bw = 34
    for g_idx, (mlabel, vals) in enumerate(model_groups):
        gx = lx0 + 32 + g_idx * 215
        for j, v in enumerate(vals):
            bh = (v / y_max) * lh
            bx = gx + j * (bw + 8)
            by = ly0 + lh - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{arm_colors[j]}" rx="3"/>')
            parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle" class="val">{v:.3f}</text>')
        parts.append(f'<text x="{gx + 2 * bw + 12:.1f}" y="{ly0 + lh + 20}" text-anchor="middle" class="axis-label">{escape(mlabel)}</text>')

    for i, (aname, acol) in enumerate(zip(arm_names, arm_colors)):
        xx = lx0 + i * 110
        parts.append(f'<rect x="{xx}" y="{ly0 + lh + 38}" width="11" height="11" fill="{acol}" rx="2"/>')
        parts.append(f'<text x="{xx + 15}" y="{ly0 + lh + 47}" class="legend">{aname}</text>')

    # Panel (b): Stacked bar of 0-gold explicit abstention ownership (Rep 1)
    rx0, ry0, rw, rh = 575, 102, 425, 250
    parts.append(f'<text x="{rx0}" y="86" class="panel-title">(b) 0-Gold Explicit Abstention Trigger Source (Rep 1)</text>')
    y2_max = 50
    for tick in [0, 10, 20, 30, 40, 50]:
        yy = ry0 + rh - (tick / y2_max) * rh
        parts.append(f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx0 + rw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{rx0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry0 + rh}" stroke="#374151" stroke-width="1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0 + rh}" x2="{rx0 + rw}" y2="{ry0 + rh}" stroke="#374151" stroke-width="1"/>')

    # Data: (label, model_abstain, ctrl_budget0, ctrl_12step, ctrl_hint_ignored)
    stacks = [
        ("Qwen4 Vh-ctrl\n(41/60 explicit)", 30, 10, 0, 1),
        ("Gemma Vh-bin\n(Phase 5: 0/60)", 0, 0, 0, 0),
        ("Gemma Vh-ctrl\n(Phase 6: 36/60)", 0, 2, 15, 19),
    ]
    scols = ["#2563eb", "#0ea5e9", "#f59e0b", "#10b981"]
    snames = ["Model abstain", "Ctrl: budget=0", "Ctrl: 12-step cap", "Ctrl: hint ignored"]
    sbw = 76
    step = rw / len(stacks)
    for i, (slabel, a, b, c, d) in enumerate(stacks):
        bx = rx0 + i * step + (step - sbw) / 2
        cur_y = ry0 + rh
        total = a + b + c + d
        for val, col in zip([a, b, c, d], scols):
            if val <= 0:
                continue
            bh = (val / y2_max) * rh
            cur_y -= bh
            parts.append(f'<rect x="{bx:.1f}" y="{cur_y:.1f}" width="{sbw}" height="{bh:.1f}" fill="{col}" stroke="#ffffff" stroke-width="1"/>')
            if val >= 5:
                parts.append(f'<text x="{bx + sbw / 2:.1f}" y="{cur_y + bh / 2 + 4:.1f}" text-anchor="middle" class="val" fill="#ffffff">{val}</text>')
        parts.append(f'<text x="{bx + sbw / 2:.1f}" y="{cur_y - 6:.1f}" text-anchor="middle" class="val">{total}/60</text>')
        for li, line in enumerate(slabel.split("\n")):
            parts.append(f'<text x="{bx + sbw / 2:.1f}" y="{ry0 + rh + 17 + li * 13}" text-anchor="middle" class="tick">{escape(line)}</text>')

    for i, (sname, scol) in enumerate(zip(snames, scols)):
        xx = rx0 + (i % 2) * 205
        yy = ry0 + rh + 48 + (i // 2) * 16
        parts.append(f'<rect x="{xx}" y="{yy - 9}" width="11" height="11" fill="{scol}" rx="2"/>')
        parts.append(f'<text x="{xx + 15}" y="{yy}" class="legend">{sname}</text>')

    parts.append(
        f'<text x="32" y="{h - 12}" class="note">'
        "Source: RESULTS_HV_MIXED180_HELDOUT_V2_CONTROLLER_20261123.md &amp; ANALYSIS_PHASE6_MECHANISM_20261124.md. "
        "Gemma 0-gold abstention is 100% controller-owned (36/36 in both reps)."
        "</text>"
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "fig1_impossible_triangle_heldout.svg": generate_fig1_heldout_main(),
        "fig2_unidentifiability_vs_binary.svg": generate_fig2_unidentifiability(),
        "fig3_progression_dev_to_heldout.svg": generate_fig3_progression(),
        "fig4_crossmodel_controller_phase6.svg": generate_fig4_crossmodel_phase6(),
    }
    for name, content in files.items():
        path = OUT_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
