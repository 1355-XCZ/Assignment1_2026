"""
visualization.py — Plotting utilities for H1, H2, H3 experiment results.

Usage:
    python -m experiments.visualization --exp H1 --results experiments/results/H1/h1_results.json
    python -m experiments.visualization --exp H2 --results experiments/results/H2/h2_results.json
    python -m experiments.visualization --exp H3 --results experiments/results/H3/h3_results.json
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ═══════════════════════════════════════════════════════════════════════════
# H1 Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def plot_h1_heatmap(results, output_dir, metric="aie_span"):
    """
    Main causal tracing heatmap: 4 rows (components) × 21 cols (3 passes × 7 blocks).
    """
    components = ["conv_0", "conv_1", "self_attn", "ffn"]
    n_passes, n_blocks = 3, 7

    matrix = np.zeros((len(components), n_passes * n_blocks))

    for p in range(n_passes):
        for b in range(n_blocks):
            col = p * n_blocks + b
            for c_idx, comp in enumerate(components):
                key = f"p{p}_b{b}_{comp}"
                matrix[c_idx, col] = results.get(key, {}).get(metric, 0.0)

    fig, ax = plt.subplots(figsize=(18, 4))
    vmax = max(abs(matrix.max()), abs(matrix.min()), 1e-6)
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto",
                   vmin=-vmax * 0.1, vmax=vmax)

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(["Conv 0", "Conv 1", "Self-Attn", "FFN"])

    xlabels = []
    for p in range(n_passes):
        for b in range(n_blocks):
            xlabels.append(f"B{b}")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7)

    # Pass separators
    for p in range(1, n_passes):
        ax.axvline(x=p * n_blocks - 0.5, color="white", linewidth=2)

    # Pass labels
    for p in range(n_passes):
        ax.text(p * n_blocks + n_blocks / 2 - 0.5, -0.8,
                f"Pass {p + 1} → M{p + 1}",
                ha="center", fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label=f"AIE ({metric})", shrink=0.8)
    ax.set_title("H1: Component-Level Causal Tracing — QANet Model Encoder")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"h1_heatmap_{metric}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved h1_heatmap_{metric}.png")


def plot_h1_component_bars(results, output_dir):
    """Aggregate AIE by component type across all passes/blocks."""
    components = ["conv_0", "conv_1", "self_attn", "ffn"]
    labels = ["Conv 0", "Conv 1", "Self-Attn", "FFN"]
    agg = {c: [] for c in components}

    for key, val in results.items():
        for c in components:
            if key.endswith(c):
                agg[c].append(val["aie_span"])

    means = [np.mean(agg[c]) for c in components]
    stds = [np.std(agg[c]) / np.sqrt(len(agg[c])) * 1.96 for c in components]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2196F3", "#42A5F5", "#FF9800", "#4CAF50"]
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="black")
    ax.set_ylabel("Average IE (span)")
    ax.set_title("H1: Aggregate Causal Effect by Component Type")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h1_component_bars.png"), dpi=150)
    plt.close(fig)
    print("  Saved h1_component_bars.png")


def plot_h1_layer_trends(results, output_dir):
    """AIE trend across blocks 0-6, one line per component, faceted by pass."""
    components = ["conv", "self_attn", "ffn"]
    comp_labels = {"conv": "Conv (avg)", "self_attn": "Self-Attn", "ffn": "FFN"}
    colors = {"conv": "#2196F3", "self_attn": "#FF9800", "ffn": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for p in range(3):
        ax = axes[p]
        for comp in components:
            vals = []
            for b in range(7):
                if comp == "conv":
                    v0 = results.get(f"p{p}_b{b}_conv_0", {}).get("aie_span", 0)
                    v1 = results.get(f"p{p}_b{b}_conv_1", {}).get("aie_span", 0)
                    vals.append((v0 + v1) / 2)
                else:
                    vals.append(results.get(f"p{p}_b{b}_{comp}", {}).get("aie_span", 0))
            ax.plot(range(7), vals, marker="o", label=comp_labels[comp],
                    color=colors[comp], linewidth=2)

        ax.set_xlabel("Block")
        ax.set_title(f"Pass {p + 1} → M{p + 1}")
        ax.set_xticks(range(7))
        if p == 0:
            ax.set_ylabel("AIE (span)")
        ax.legend(fontsize=8)

    fig.suptitle("H1: Layer-wise AIE Trends", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h1_layer_trends.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved h1_layer_trends.png")


def plot_h1_start_vs_end(results, output_dir):
    """Heatmap of AIE_p1 - AIE_p2 (positive = more important for start)."""
    components = ["conv_0", "conv_1", "self_attn", "ffn"]
    n_passes, n_blocks = 3, 7

    matrix = np.zeros((len(components), n_passes * n_blocks))
    for p in range(n_passes):
        for b in range(n_blocks):
            col = p * n_blocks + b
            for c_idx, comp in enumerate(components):
                key = f"p{p}_b{b}_{comp}"
                v = results.get(key, {})
                matrix[c_idx, col] = v.get("aie_p1", 0) - v.get("aie_p2", 0)

    fig, ax = plt.subplots(figsize=(18, 4))
    vmax = max(abs(matrix.max()), abs(matrix.min()), 1e-6)
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(["Conv 0", "Conv 1", "Self-Attn", "FFN"])
    xlabels = [f"B{b}" for p in range(n_passes) for b in range(n_blocks)]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7)

    for p in range(1, n_passes):
        ax.axvline(x=p * n_blocks - 0.5, color="black", linewidth=2)

    plt.colorbar(im, ax=ax, label="AIE_p1 − AIE_p2 (red=start, blue=end)")
    ax.set_title("H1: Start vs End Position Bias")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h1_start_vs_end.png"), dpi=150)
    plt.close(fig)
    print("  Saved h1_start_vs_end.png")


# ═══════════════════════════════════════════════════════════════════════════
# H2 Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def plot_h2_total_effect(results, output_dir):
    """Bar chart comparing TE across corruption conditions."""
    conditions = ["context", "question", "both"]
    labels = ["Corrupt Context", "Corrupt Question", "Corrupt Both"]
    means = [results[c]["total_effect"]["mean"] for c in conditions]
    ci = [results[c]["total_effect"]["ci95"] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2196F3", "#FF9800", "#F44336"]
    ax.bar(labels, means, yerr=ci, color=colors, capsize=5, edgecolor="black")
    ax.set_ylabel("Total Effect (TE)")
    ax.set_title("H2: Corruption Condition → Total Effect on Span Prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h2_total_effect.png"), dpi=150)
    plt.close(fig)
    print("  Saved h2_total_effect.png")


def plot_h2_emb_enc_components(results, output_dir):
    """Side-by-side bars: Emb Encoder sub-layer AIE for C vs Q streams."""
    components = ["conv_0", "conv_1", "conv_2", "conv_3", "self_attn", "ffn"]
    comp_labels = ["Conv0", "Conv1", "Conv2", "Conv3", "Attn", "FFN"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, (cc, stream, title) in enumerate([
        ("context", "emb_enc_C", "Corrupt Context → Restore C Encoder"),
        ("question", "emb_enc_Q", "Corrupt Question → Restore Q Encoder"),
    ]):
        ax = axes[ax_idx]
        ie_vals = results[cc]["indirect_effects"]
        means = []
        cis = []
        for comp in components:
            key = f"{stream}_{comp}"
            if key in ie_vals:
                means.append(ie_vals[key]["aie"])
                cis.append(ie_vals[key]["ci95"])
            else:
                means.append(0.0)
                cis.append(0.0)

        colors = ["#42A5F5"] * 4 + ["#FF9800", "#4CAF50"]
        ax.bar(comp_labels, means, yerr=cis, color=colors, capsize=4, edgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("AIE" if ax_idx == 0 else "")

    fig.suptitle("H2: Embedding Encoder Sub-layer Causal Effects")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h2_emb_enc_components.png"), dpi=150)
    plt.close(fig)
    print("  Saved h2_emb_enc_components.png")


def plot_h2_pipeline_waterfall(results, output_dir):
    """Waterfall showing cumulative NIE through the pipeline (CORRUPT-both)."""
    both = results.get("both", {}).get("indirect_effects", {})
    if not both:
        print("  Skipping waterfall (no CORRUPT-both results)")
        return

    stages = [
        ("Emb Enc (C)", "emb_enc_C_output"),
        ("Emb Enc (Q)", "emb_enc_Q_output"),
        ("CQ Attention", "cq_att_output"),
        ("CQ Resizer", "cq_resized_output"),
    ]

    labels = [s[0] for s in stages]
    nies = []
    for _, key in stages:
        if key in both:
            nies.append(both[key]["nie"])
        else:
            nies.append(0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    ax.bar(labels, nies, color=colors, edgecolor="black")
    ax.set_ylabel("Normalized IE (fraction of TE recovered)")
    ax.set_title("H2: Pipeline Stage Recovery (CORRUPT-both)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Full recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h2_pipeline_waterfall.png"), dpi=150)
    plt.close(fig)
    print("  Saved h2_pipeline_waterfall.png")


# ═══════════════════════════════════════════════════════════════════════════
# H3 Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def plot_h3_phase_a_table(phase_a, output_dir):
    """Pointer config comparison as a bar chart."""
    configs = list(phase_a.keys())
    f1_vals = [phase_a[c]["f1"] for c in configs]
    em_vals = [phase_a[c]["em"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, f1_vals, width, label="F1", color="#2196F3",
                   edgecolor="black")
    bars2 = ax.bar(x + width / 2, em_vals, width, label="EM", color="#FF9800",
                   edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("H3 Phase A: Pointer Configuration Comparison")
    ax.legend()

    # Annotate baseline
    baseline_f1 = phase_a.get("original", {}).get("f1", 0)
    ax.axhline(y=baseline_f1, color="red", linestyle="--", alpha=0.5,
               label=f"Baseline F1={baseline_f1:.1f}")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h3_phase_a_bars.png"), dpi=150)
    plt.close(fig)
    print("  Saved h3_phase_a_bars.png")


def plot_h3_cosine_distributions(phase_b, output_dir):
    """Violin/box plots of pairwise cosine similarity."""
    global_cos = phase_b.get("global_cosine", {})
    pairs = ["M1_M2", "M1_M3", "M2_M3"]
    labels = ["M1 vs M2", "M1 vs M3", "M2 vs M3"]
    means = [global_cos.get(p, {}).get("mean", 0) for p in pairs]
    stds = [global_cos.get(p, {}).get("std", 0) for p in pairs]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="black")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("H3 Phase B: Global Representation Similarity")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h3_cosine_global.png"), dpi=150)
    plt.close(fig)
    print("  Saved h3_cosine_global.png")


def plot_h3_position_cosine(phase_b, output_dir):
    """M2 vs M3 similarity by position region."""
    pos_cos = phase_b.get("position_cosine_M2_M3", {})
    regions = ["answer_start", "answer_end", "answer_interior", "non_answer"]
    labels = ["Ans Start", "Ans End", "Interior", "Non-Answer"]
    means = [pos_cos.get(r, {}).get("mean", 0) for r in regions]
    cis = [pos_cos.get(r, {}).get("ci95", 0) for r in regions]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#F44336", "#FF5722", "#FFC107", "#9E9E9E"]
    ax.bar(labels, means, yerr=cis, color=colors, capsize=5, edgecolor="black")
    ax.set_ylabel("Cosine Similarity (M2 vs M3)")
    ax.set_title("H3 Phase B: M2-M3 Similarity by Token Position")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h3_position_cosine.png"), dpi=150)
    plt.close(fig)
    print("  Saved h3_position_cosine.png")


def plot_h3_cka_matrix(phase_b, output_dir):
    """3x3 CKA heatmap."""
    cka = phase_b.get("cka", {})
    if not cka:
        print("  Skipping CKA plot (no data)")
        return

    labels = ["M1", "M2", "M3"]
    matrix = np.ones((3, 3))
    matrix[0, 1] = matrix[1, 0] = cka.get("M1_M2", 0)
    matrix[0, 2] = matrix[2, 0] = cka.get("M1_M3", 0)
    matrix[1, 2] = matrix[2, 1] = cka.get("M2_M3", 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, color="white" if matrix[i, j] > 0.7 else "black")

    plt.colorbar(im, ax=ax, label="CKA", shrink=0.8)
    ax.set_title(f"H3: CKA Similarity ({cka.get('n_tokens', '?')} tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "h3_cka_matrix.png"), dpi=150)
    plt.close(fig)
    print("  Saved h3_cka_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not HAS_MPL:
        print("matplotlib is required. Install with: pip install matplotlib")
        return

    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument("--exp", type=str, required=True, choices=["H1", "H2", "H3", "all"])
    parser.add_argument("--results", type=str, required=True, help="Path to JSON results file")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    output_dir = args.output_dir or os.path.dirname(args.results)
    os.makedirs(output_dir, exist_ok=True)

    if args.exp in ("H1", "all"):
        results = data.get("results", data)
        print("Generating H1 plots...")
        plot_h1_heatmap(results, output_dir, metric="aie_span")
        plot_h1_component_bars(results, output_dir)
        plot_h1_layer_trends(results, output_dir)
        plot_h1_start_vs_end(results, output_dir)

    if args.exp in ("H2", "all"):
        results = data.get("results", data)
        print("Generating H2 plots...")
        plot_h2_total_effect(results, output_dir)
        plot_h2_emb_enc_components(results, output_dir)
        plot_h2_pipeline_waterfall(results, output_dir)

    if args.exp in ("H3", "all"):
        print("Generating H3 plots...")
        plot_h3_phase_a_table(data.get("phase_a", {}), output_dir)
        plot_h3_cosine_distributions(data.get("phase_b", {}), output_dir)
        plot_h3_position_cosine(data.get("phase_b", {}), output_dir)
        plot_h3_cka_matrix(data.get("phase_b", {}), output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
