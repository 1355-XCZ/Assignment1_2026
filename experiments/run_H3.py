"""
run_H3.py — H3: Pointer Asymmetry & Pass Role Differentiation

Phase A: Eval-time replacement experiments (swap / symmetrize M1/M2/M3)
Phase B: Representation similarity analysis (cosine, CKA, position-stratified)

Usage (from project root):
    python -m experiments.run_H3 --ckpt _model/model.pt
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Data import SQuADDataset, load_word_char_mats, load_train_dev_eval, make_loader
from Models import QANet
from Models.encoder import mask_logits
from EvaluateTools.eval_utils import convert_tokens, squad_evaluate
from experiments.tracer import qanet_forward

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"] if "config" in ckpt else ckpt.get("args", {})
    args = argparse.Namespace(**config)

    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args).to(DEVICE)
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, args


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Pointer replacement experiments
# ═══════════════════════════════════════════════════════════════════════════

POINTER_CONFIGS = {
    "original":  {"p1": ("M1", "M2"), "p2": ("M1", "M3")},
    "swap":      {"p1": ("M1", "M3"), "p2": ("M1", "M2")},
    "sym_M2":    {"p1": ("M1", "M2"), "p2": ("M1", "M2")},
    "sym_M3":    {"p1": ("M1", "M3"), "p2": ("M1", "M3")},
    "only_M1":   {"p1": ("M1", "M1"), "p2": ("M1", "M1")},
    "no_M1":     {"p1": ("M2", "M3"), "p2": ("M2", "M3")},
}


def pointer_forward(model, M1, M2, M3, cmask, config_name="original"):
    """Run Pointer with a specified M1/M2/M3 wiring configuration."""
    cfg = POINTER_CONFIGS[config_name]
    M_map = {"M1": M1, "M2": M2, "M3": M3}

    X1 = torch.cat([M_map[cfg["p1"][0]], M_map[cfg["p1"][1]]], dim=1)
    X2 = torch.cat([M_map[cfg["p2"][0]], M_map[cfg["p2"][1]]], dim=1)

    Y1 = torch.matmul(model.out.w1, X1)
    Y2 = torch.matmul(model.out.w2, X2)

    p1 = mask_logits(Y1, cmask)
    p2 = mask_logits(Y2, cmask)
    return p1, p2


@torch.no_grad()
def run_phase_a(model, dataset, eval_file, batch_size=32):
    """
    Evaluate all pointer configurations on the full dev set.

    Returns dict: config_name -> {f1, em}
    """
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    results = {cfg: {"answers": {}} for cfg in POINTER_CONFIGS}

    for Cwid, Ccid, Qwid, Qcid, y1, y2, ids in tqdm(loader, desc="Phase A"):
        Cwid = Cwid.to(DEVICE); Ccid = Ccid.to(DEVICE)
        Qwid = Qwid.to(DEVICE); Qcid = Qcid.to(DEVICE)
        cmask = (Cwid == 0)

        # Single forward to get M1, M2, M3
        _, _, _, intermediates = qanet_forward(
            model, Cwid, Ccid, Qwid, Qcid, collect=False
        )
        M1, M2, M3 = intermediates["M1"], intermediates["M2"], intermediates["M3"]

        for cfg_name in POINTER_CONFIGS:
            p1, p2 = pointer_forward(model, M1, M2, M3, cmask, cfg_name)

            # Decode best span
            p1_log = F.log_softmax(p1, dim=1)
            p2_log = F.log_softmax(p2, dim=1)
            outer = p1_log.unsqueeze(2) + p2_log.unsqueeze(1)
            mask = torch.triu(torch.ones(outer.size(-2), outer.size(-1),
                                         device=DEVICE, dtype=torch.bool))
            outer = outer.masked_fill(~mask, float('-inf'))
            flat = outer.view(outer.size(0), -1)
            idx = torch.argmax(flat, dim=1)
            L = p1.size(1)
            ymin = idx // L
            ymax = idx % L

            ans_dict, _ = convert_tokens(eval_file, ids.tolist(),
                                         ymin.tolist(), ymax.tolist())
            results[cfg_name]["answers"].update(ans_dict)

    # Compute F1/EM for each config
    phase_a_results = {}
    for cfg_name in POINTER_CONFIGS:
        metrics = squad_evaluate(eval_file, results[cfg_name]["answers"])
        phase_a_results[cfg_name] = {
            "f1": metrics["f1"],
            "em": metrics["exact_match"],
            "wiring": POINTER_CONFIGS[cfg_name],
        }
        print(f"  {cfg_name:>12s}: F1={metrics['f1']:.2f}  EM={metrics['exact_match']:.2f}")

    return phase_a_results


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Representation similarity analysis
# ═══════════════════════════════════════════════════════════════════════════

def cosine_sim_per_token(Ma, Mb, mask):
    """
    Token-level cosine similarity between two representations.
    Ma, Mb: [B, d_model, L]
    mask: [B, L] (True = PAD)
    Returns: [B, L] cosine similarities (PAD positions = 0)
    """
    Ma_t = Ma.transpose(1, 2)  # [B, L, d]
    Mb_t = Mb.transpose(1, 2)
    cos = F.cosine_similarity(Ma_t, Mb_t, dim=2)  # [B, L]
    cos = cos.masked_fill(mask, 0.0)
    return cos


def linear_cka(X, Y):
    """
    Linear CKA between two feature matrices.
    X, Y: [n_samples, n_features]
    """
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    hsic_xy = (XtY ** 2).sum()
    hsic_xx = (XtX ** 2).sum()
    hsic_yy = (YtY ** 2).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


@torch.no_grad()
def run_phase_b(model, dataset, num_samples=1000, batch_size=32, seed=42):
    """
    Collect M1, M2, M3 and compute pairwise similarity metrics.

    Returns dict with:
        global_cosine: pairwise avg cosine sim
        position_cosine: cosine sim stratified by token position relative to answer
        cka: pairwise CKA values
    """
    torch.manual_seed(seed)
    loader = make_loader(dataset, batch_size=batch_size, shuffle=True)

    # Accumulators for cosine similarity
    cos_pairs = {
        "M1_M2": [], "M1_M3": [], "M2_M3": [],
    }

    # Position-stratified accumulators for M2 vs M3
    pos_groups = {
        "answer_start": [], "answer_end": [],
        "answer_interior": [], "non_answer": [],
    }

    # For CKA: collect flattened token representations
    cka_features = {"M1": [], "M2": [], "M3": []}
    max_cka_tokens = 5000  # limit for memory

    samples_seen = 0
    for Cwid, Ccid, Qwid, Qcid, y1, y2, ids in tqdm(loader, desc="Phase B"):
        if samples_seen >= num_samples:
            break

        Cwid = Cwid.to(DEVICE); Ccid = Ccid.to(DEVICE)
        Qwid = Qwid.to(DEVICE); Qcid = Qcid.to(DEVICE)
        y1 = y1.to(DEVICE); y2 = y2.to(DEVICE)
        cmask = (Cwid == 0)

        _, _, _, intermediates = qanet_forward(
            model, Cwid, Ccid, Qwid, Qcid, collect=False
        )
        M1, M2, M3 = intermediates["M1"], intermediates["M2"], intermediates["M3"]

        # Global cosine similarity
        cos_12 = cosine_sim_per_token(M1, M2, cmask)  # [B, L]
        cos_13 = cosine_sim_per_token(M1, M3, cmask)
        cos_23 = cosine_sim_per_token(M2, M3, cmask)

        valid = ~cmask  # [B, L]
        for b_idx in range(Cwid.size(0)):
            v = valid[b_idx]
            if v.sum() == 0:
                continue

            cos_pairs["M1_M2"].append(cos_12[b_idx][v].mean().item())
            cos_pairs["M1_M3"].append(cos_13[b_idx][v].mean().item())
            cos_pairs["M2_M3"].append(cos_23[b_idx][v].mean().item())

            # Position-stratified M2 vs M3 similarity
            y1_val = y1[b_idx].item()
            y2_val = y2[b_idx].item()
            L = v.size(0)
            cos_23_b = cos_23[b_idx]

            start_region = set(range(max(0, y1_val - 1), min(L, y1_val + 2)))
            end_region = set(range(max(0, y2_val - 1), min(L, y2_val + 2)))
            interior = set(range(y1_val + 1, y2_val)) - start_region - end_region

            all_valid = set(i for i in range(L) if v[i].item())
            non_answer = all_valid - start_region - end_region - interior

            for region_name, indices in [
                ("answer_start", start_region & all_valid),
                ("answer_end", end_region & all_valid),
                ("answer_interior", interior & all_valid),
                ("non_answer", non_answer),
            ]:
                if indices:
                    idx_list = list(indices)
                    region_cos = cos_23_b[idx_list].mean().item()
                    pos_groups[region_name].append(region_cos)

            # Collect features for CKA (subsample tokens)
            total_cka = sum(len(v) for v in cka_features.values())
            if total_cka < max_cka_tokens:
                valid_idx = v.nonzero(as_tuple=True)[0]
                # Subsample if too many
                if len(valid_idx) > 20:
                    perm = torch.randperm(len(valid_idx))[:20]
                    valid_idx = valid_idx[perm]

                m1_feat = M1[b_idx, :, valid_idx].T.cpu()  # [n_tokens, d]
                m2_feat = M2[b_idx, :, valid_idx].T.cpu()
                m3_feat = M3[b_idx, :, valid_idx].T.cpu()
                cka_features["M1"].append(m1_feat)
                cka_features["M2"].append(m2_feat)
                cka_features["M3"].append(m3_feat)

        samples_seen += Cwid.size(0)

    # Compute CKA
    cka_results = {}
    if cka_features["M1"]:
        M1_all = torch.cat(cka_features["M1"], dim=0).numpy()
        M2_all = torch.cat(cka_features["M2"], dim=0).numpy()
        M3_all = torch.cat(cka_features["M3"], dim=0).numpy()

        # Center features
        M1_all -= M1_all.mean(axis=0)
        M2_all -= M2_all.mean(axis=0)
        M3_all -= M3_all.mean(axis=0)

        cka_results = {
            "M1_M2": linear_cka(M1_all, M2_all),
            "M1_M3": linear_cka(M1_all, M3_all),
            "M2_M3": linear_cka(M2_all, M3_all),
            "n_tokens": len(M1_all),
        }

    # Aggregate cosine results
    def _stats(arr):
        arr = np.array(arr)
        n = len(arr)
        ci = 1.96 / np.sqrt(n) if n > 1 else 0.0
        return {
            "mean": float(arr.mean()) if n > 0 else 0.0,
            "std": float(arr.std()) if n > 0 else 0.0,
            "ci95": float(arr.std() * ci) if n > 0 else 0.0,
            "n": n,
        }

    global_cosine = {k: _stats(v) for k, v in cos_pairs.items()}
    position_cosine = {k: _stats(v) for k, v in pos_groups.items()}

    return {
        "global_cosine": global_cosine,
        "position_cosine_M2_M3": position_cosine,
        "cka": cka_results,
        "num_samples": samples_seen,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="H3: Pointer Asymmetry Analysis")
    parser.add_argument("--ckpt", type=str, default="_model/model.pt")
    parser.add_argument("--dev_npz", type=str, default="_data/dev.npz")
    parser.add_argument("--dev_eval_json", type=str, default="_data/dev_eval.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples_b", type=int, default=1000,
                        help="Samples for Phase B representation analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments/results/H3")
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    model, model_args = load_model(args.ckpt)

    dataset = SQuADDataset(args.dev_npz)

    import ujson
    with open(args.dev_eval_json) as f:
        eval_file = ujson.load(f)
    print(f"Dataset: {len(dataset)} samples")

    # Phase A
    print("\n" + "=" * 60)
    print("Phase A: Pointer Replacement Experiments")
    print("=" * 60)
    phase_a = run_phase_a(model, dataset, eval_file, batch_size=args.batch_size)

    # Phase B
    print("\n" + "=" * 60)
    print("Phase B: Representation Similarity Analysis")
    print("=" * 60)
    phase_b = run_phase_b(model, dataset, num_samples=args.num_samples_b,
                          batch_size=args.batch_size, seed=args.seed)

    # Print Phase B summary
    print("\n--- Global Cosine Similarity ---")
    for pair, stats in phase_b["global_cosine"].items():
        print(f"  {pair}: {stats['mean']:.4f} ± {stats['ci95']:.4f}")

    print("\n--- M2 vs M3 Cosine by Position ---")
    for region, stats in phase_b["position_cosine_M2_M3"].items():
        print(f"  {region:>20s}: {stats['mean']:.4f} ± {stats['ci95']:.4f}")

    print("\n--- CKA ---")
    for pair in ["M1_M2", "M1_M3", "M2_M3"]:
        if pair in phase_b["cka"]:
            print(f"  CKA({pair}) = {phase_b['cka'][pair]:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "h3_results.json")
    with open(output_path, "w") as f:
        json.dump({"phase_a": phase_a, "phase_b": phase_b}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
