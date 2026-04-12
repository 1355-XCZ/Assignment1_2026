#!/usr/bin/env python
"""
plot_H1.py — Three publication-quality figures for H1.

  Fig 1  Causal Tracing Heatmap   (2×2: Conv₀ | Conv₁ / Self-Attn | FFN)
  Fig 2  Dual-Method Validation    (Ablation |ΔF1| share vs Causal-Tracing AIE share)
  Fig 3  Attention JSD Degradation (per-block: −Conv₁ vs −Conv₀)

Usage (from project root):
    python -m experiments.plot_H1
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths ────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_DIR, "results", "H1")
OUT_DIR = os.path.join(_DIR, "prism-uploads")
os.makedirs(OUT_DIR, exist_ok=True)

# ── global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── palette ──────────────────────────────────────────────────────
PAL = {
    "conv1": "#d32f2f", "conv0": "#1976d2",
    "attn":  "#2e7d32", "ffn":   "#ef6c00",
    "accent": "#7b1fa2",
}
COMP_ORDER = ["conv_0", "conv_1", "self_attn", "ffn"]
COMP_LABEL = {
    "conv_0": "Conv₀", "conv_1": "Conv₁",
    "self_attn": "Self-Attn", "ffn": "FFN",
}
COMP_CLR = {
    "conv_0": PAL["conv0"], "conv_1": PAL["conv1"],
    "self_attn": PAL["attn"], "ffn": PAL["ffn"],
}


def _load(name):
    p = os.path.join(RESULT_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  Figure 1 — Causal Tracing Heatmap  (2×2)
# ═══════════════════════════════════════════════════════════════════
def fig1_heatmap():
    data = _load("h1_results.json")
    if data is None:
        print("[Fig 1] SKIP — h1_results.json not found.  Run notebook first.")
        return

    res = data["results"]
    grids = {}
    for comp in COMP_ORDER:
        g = np.zeros((3, 7))
        for p in range(3):
            for b in range(7):
                g[p, b] = res.get(f"p{p}_b{b}_{comp}", {}).get("aie_span", 0.0)
        grids[comp] = g

    vmax = max(g.max() for g in grids.values())
    cmap = plt.get_cmap("YlOrRd").copy()
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8), dpi=150)
    fig.suptitle("Causal Tracing: AIE per Sub-layer",
                 fontsize=15, fontweight="bold", y=0.97)

    layout = [(0, 0, "conv_0"), (0, 1, "conv_1"),
              (1, 0, "self_attn"), (1, 1, "ffn")]

    for r, c, comp in layout:
        ax = axes[r][c]
        g = grids[comp]
        im = ax.imshow(g, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")

        # panel title
        total_aie = float(g.sum())
        ax.set_title(f"{COMP_LABEL[comp]}    (Sum AIE = {total_aie:.3f})",
                     fontsize=12, fontweight="bold", color=COMP_CLR[comp])

        # x ticks
        ax.set_xticks(range(7))
        ax.set_xticklabels([f"B{i}" for i in range(7)], fontsize=9)
        if r == 1:
            ax.set_xlabel("Block", fontsize=10, fontweight="bold")

        # y ticks — only left column
        ax.set_yticks(range(3))
        if c == 0:
            ax.set_yticklabels(
                ["Pass 0  (→M₁)", "Pass 1  (→M₂)", "Pass 2  (→M₃)"],
                fontsize=9)
        else:
            ax.set_yticklabels([])

        # cell annotations
        for pi in range(3):
            for bi in range(7):
                v = g[pi, bi]
                txt_color = "white" if v > vmax * 0.55 else "black"
                ax.text(bi, pi, f"{v:.3f}", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=txt_color)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.03, aspect=25)
    cbar.set_label("AIE  (span-probability recovery)", fontsize=10,
                   fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(hspace=0.38, wspace=0.12)
    path = os.path.join(OUT_DIR, "h1_fig1_causal_tracing_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 2 — Dual-Method Cross-Validation
# ═══════════════════════════════════════════════════════════════════
def fig2_dual_method():
    # ── report Table 9.1 / 9.2  (hardcoded fallback) ──
    ABL = {"conv_1": 32.54, "conv_0": 1.09, "self_attn": 4.81, "ffn": 3.32}
    CT  = {"conv_1": 0.3089, "conv_0": 0.0664, "self_attn": 0.0330, "ffn": 0.0246}

    abl_data = _load("h1_ablation_global.json")
    if abl_data:
        base = abl_data.get("baseline", {}).get("f1", 70.24)
        _map = {"conv_1": "skip_conv_1", "conv_0": "skip_conv_0",
                "self_attn": "skip_attn", "ffn": "skip_ffn"}
        for comp, key in _map.items():
            if key in abl_data:
                ABL[comp] = base - abl_data[key]["f1"]

    ct_data = _load("h1_results.json")
    if ct_data:
        res = ct_data["results"]
        for comp in COMP_ORDER:
            CT[comp] = sum(v["aie_span"] for k, v in res.items()
                          if k.endswith(comp))

    abl_tot = sum(ABL.values())
    ct_tot  = sum(CT.values())
    abl_pct = {c: ABL[c] / abl_tot * 100 for c in COMP_ORDER}
    ct_pct  = {c: CT[c]  / ct_tot  * 100 for c in COMP_ORDER}

    x = np.arange(len(COMP_ORDER))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    ax.bar(x - w / 2, [abl_pct[c] for c in COMP_ORDER], w,
           label="Ablation  (|ΔF1| share)",
           color=PAL["conv1"], alpha=0.82, edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, [ct_pct[c] for c in COMP_ORDER], w,
           label="Causal Tracing  (AIE share)",
           color=PAL["conv0"], alpha=0.82, edgecolor="white", linewidth=0.8)

    for i, comp in enumerate(COMP_ORDER):
        ax.text(i - w / 2, abl_pct[comp] + 1.0,
                f"|ΔF1|={ABL[comp]:.1f}",
                ha="center", fontsize=7.5, fontweight="bold", color="#b71c1c")
        ax.text(i + w / 2, ct_pct[comp] + 1.0,
                f"{ct_pct[comp]:.1f}%",
                ha="center", fontsize=7.5, fontweight="bold", color="#0d47a1")

    ax.set_ylabel("Share of Total  (%)", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Validation: Conv₁ Dominance Confirmed by Both Methods",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([COMP_LABEL[c] for c in COMP_ORDER],
                       fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    ax.set_ylim(0, max(max(abl_pct.values()), max(ct_pct.values())) * 1.15)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "h1_fig2_dual_method.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 3 — Attention JSD Degradation
# ═══════════════════════════════════════════════════════════════════
def fig3_attention_jsd():
    # ── report Table 9.6  (hardcoded fallback) ──
    JSD_C1 = [0.0331, 0.0870, 0.0749, 0.1222, 0.0946, 0.0888, 0.0898]
    JSD_C0 = [0.0041, 0.0110, 0.0108, 0.0118, 0.0118, 0.0110, 0.0112]

    attn = _load("h1_attn_degradation.json")
    if attn and "js_divergence" in attn:
        JSD_C1 = attn["js_divergence"]["skip_conv_1"]
        JSD_C0 = attn["js_divergence"]["skip_conv_0"]

    x = np.arange(7)
    w = 0.32
    ratios = [c1 / max(c0, 1e-12) for c1, c0 in zip(JSD_C1, JSD_C0)]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    ax.bar(x - w / 2, JSD_C1, w, label="−Conv₁  (treatment)",
           color=PAL["conv1"], alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, JSD_C0, w, label="−Conv₀  (control)",
           color=PAL["conv0"], alpha=0.85, edgecolor="white", linewidth=0.8)

    for i in range(7):
        ax.annotate(f"{ratios[i]:.1f}×",
                    xy=(i, JSD_C1[i]), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=PAL["accent"])

    avg_c1 = np.mean(JSD_C1)
    avg_c0 = np.mean(JSD_C0)
    ax.axhline(avg_c1, color=PAL["conv1"], ls="--", lw=1.0, alpha=0.45,
               label=f"−Conv₁ avg ({avg_c1:.4f})")
    ax.axhline(avg_c0, color=PAL["conv0"], ls="--", lw=1.0, alpha=0.45,
               label=f"−Conv₀ avg ({avg_c0:.4f})")

    avg_ratio = avg_c1 / max(avg_c0, 1e-12)
    box_txt = (f"Average ratio:  {avg_ratio:.1f}×\n"
               f"Range:  {min(ratios):.1f}–{max(ratios):.1f}×")
    ax.text(0.97, 0.94, box_txt, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f3e5f5",
                      ec=PAL["accent"], alpha=0.92, lw=1.2))

    ax.set_xlabel("Encoder Block", fontsize=12, fontweight="bold")
    ax.set_ylabel("JS Divergence  (clean vs ablated)", fontsize=12,
                  fontweight="bold")
    ax.set_title(
        "Attention Distortion: Removing Conv₁ Is 8× Worse Than Conv₀",
        fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {i}" for i in range(7)], fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92, ncol=2)
    ax.set_ylim(0, max(JSD_C1) * 1.25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "h1_fig3_attention_jsd.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Generating H1 Figures")
    print("=" * 60)
    fig1_heatmap()
    fig2_dual_method()
    fig3_attention_jsd()
    print("\nDone.")
