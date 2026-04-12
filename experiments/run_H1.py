"""
run_H1.py — H1: Component-Level Causal Tracing in QANet Model Encoder

Measures the Average Indirect Effect (AIE) of each sub-layer
(Conv_0, Conv_1, Self-Attention, FFN) across all 7 blocks × 3 passes
of the Model Encoder.

Usage (from project root):
    python -m experiments.run_H1 --ckpt _model/model.pt --num_samples 300
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import random
import numpy as np
import torch
from tqdm import tqdm

from Data import SQuADDataset, load_word_char_mats, load_train_dev_eval, make_loader
from Models import QANet
from experiments.tracer import (
    qanet_forward, compute_span_prob, compute_start_prob, compute_end_prob,
    compute_ie, compute_nie, build_model_enc_specs, RestoreSpec,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    """Load trained QANet from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"] if "config" in ckpt else ckpt.get("args", {})
    args = argparse.Namespace(**config)

    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args).to(DEVICE)
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, args


@torch.no_grad()
def run_h1(model, dataset, num_samples=300, noise_std_scale=3.0,
           noise_repeats=3, min_clean_prob=0.01, seed=42):
    """
    Run H1 causal tracing over Model Encoder.

    Returns
    -------
    results : dict mapping spec_key -> {aie_span, aie_p1, aie_p2, anie_span, ci95}
    meta    : dict with {num_samples_used, avg_te, ...}
    """
    random.seed(seed)
    torch.manual_seed(seed)

    loader = make_loader(dataset, batch_size=1, shuffle=True)
    specs = build_model_enc_specs()  # 3×7×4 = 84 specs
    spec_keys = [f"p{s.pass_idx}_b{s.block_idx}_{s.component}" for s in specs]

    # Accumulators: {key: list of per-sample IE values}
    ie_span_acc = {k: [] for k in spec_keys}
    ie_p1_acc = {k: [] for k in spec_keys}
    ie_p2_acc = {k: [] for k in spec_keys}
    nie_span_acc = {k: [] for k in spec_keys}
    te_list = []

    samples_used = 0
    for batch_idx, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(
        tqdm(loader, total=num_samples, desc="H1 Causal Tracing")
    ):
        if samples_used >= num_samples:
            break

        Cwid = Cwid.to(DEVICE); Ccid = Ccid.to(DEVICE)
        Qwid = Qwid.to(DEVICE); Qcid = Qcid.to(DEVICE)
        y1 = y1.to(DEVICE); y2 = y2.to(DEVICE)

        # 1. Clean run
        p1_c, p2_c, clean_acts, _ = qanet_forward(
            model, Cwid, Ccid, Qwid, Qcid, collect=True
        )
        prob_clean = compute_span_prob(p1_c, p2_c, y1, y2)

        if prob_clean.item() < min_clean_prob:
            continue

        # Average over noise repeats
        sample_ie_span = {k: [] for k in spec_keys}
        sample_ie_p1 = {k: [] for k in spec_keys}
        sample_ie_p2 = {k: [] for k in spec_keys}
        sample_te = []

        for rep in range(noise_repeats):
            noise_seed = seed + batch_idx * 100 + rep

            # 2. Corrupted run
            p1_x, p2_x, _, _ = qanet_forward(
                model, Cwid, Ccid, Qwid, Qcid,
                corrupt_target="context",
                noise_std_scale=noise_std_scale,
                noise_seed=noise_seed,
            )
            prob_corrupt = compute_span_prob(p1_x, p2_x, y1, y2)
            prob_p1_corrupt = compute_start_prob(p1_x, y1)
            prob_p2_corrupt = compute_end_prob(p2_x, y2)

            te = (prob_clean - prob_corrupt).item()
            sample_te.append(te)

            if abs(te) < 1e-6:
                continue

            # 3. Restoration runs for each spec
            for spec, key in zip(specs, spec_keys):
                p1_r, p2_r, _, _ = qanet_forward(
                    model, Cwid, Ccid, Qwid, Qcid,
                    corrupt_target="context",
                    noise_std_scale=noise_std_scale,
                    noise_seed=noise_seed,
                    clean_acts=clean_acts,
                    restore_spec=spec,
                )
                prob_r = compute_span_prob(p1_r, p2_r, y1, y2)
                ie = (prob_r - prob_corrupt).item()
                nie = ie / max(abs(te), 1e-8)
                ie_p1 = (compute_start_prob(p1_r, y1) - prob_p1_corrupt).item()
                ie_p2 = (compute_end_prob(p2_r, y2) - prob_p2_corrupt).item()

                sample_ie_span[key].append(ie)
                sample_ie_p1[key].append(ie_p1)
                sample_ie_p2[key].append(ie_p2)

        # Average over noise repeats for this sample
        avg_te = np.mean(sample_te) if sample_te else 0.0
        te_list.append(avg_te)

        for key in spec_keys:
            if sample_ie_span[key]:
                ie_span_acc[key].append(np.mean(sample_ie_span[key]))
                ie_p1_acc[key].append(np.mean(sample_ie_p1[key]))
                ie_p2_acc[key].append(np.mean(sample_ie_p2[key]))
                nie_val = np.mean(sample_ie_span[key]) / max(abs(avg_te), 1e-8)
                nie_span_acc[key].append(nie_val)

        samples_used += 1

    # Aggregate: mean and 95% CI
    results = {}
    for key in spec_keys:
        vals_span = np.array(ie_span_acc[key])
        vals_p1 = np.array(ie_p1_acc[key])
        vals_p2 = np.array(ie_p2_acc[key])
        vals_nie = np.array(nie_span_acc[key])
        n = len(vals_span)
        ci_factor = 1.96 / np.sqrt(n) if n > 1 else 0.0
        results[key] = {
            "aie_span": float(vals_span.mean()) if n > 0 else 0.0,
            "aie_p1": float(vals_p1.mean()) if n > 0 else 0.0,
            "aie_p2": float(vals_p2.mean()) if n > 0 else 0.0,
            "anie_span": float(vals_nie.mean()) if n > 0 else 0.0,
            "ci95_span": float(vals_span.std() * ci_factor) if n > 0 else 0.0,
            "n": n,
        }

    meta = {
        "num_samples_used": samples_used,
        "avg_te": float(np.mean(te_list)) if te_list else 0.0,
        "noise_std_scale": noise_std_scale,
        "noise_repeats": noise_repeats,
        "corrupt_target": "context",
    }

    return results, meta


def main():
    parser = argparse.ArgumentParser(description="H1: Component-Level Causal Tracing")
    parser.add_argument("--ckpt", type=str, default="_model/model.pt")
    parser.add_argument("--dev_npz", type=str, default="_data/dev.npz")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--noise_std_scale", type=float, default=3.0)
    parser.add_argument("--noise_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments/results/H1")
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    model, model_args = load_model(args.ckpt)
    print(f"Model loaded. Device: {DEVICE}")

    dataset = SQuADDataset(args.dev_npz)
    print(f"Dataset loaded: {len(dataset)} samples")

    results, meta = run_h1(
        model, dataset,
        num_samples=args.num_samples,
        noise_std_scale=args.noise_std_scale,
        noise_repeats=args.noise_repeats,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "h1_results.json")
    with open(output_path, "w") as f:
        json.dump({"results": results, "meta": meta}, f, indent=2)
    print(f"Results saved to {output_path}")
    print(f"Samples used: {meta['num_samples_used']}, Avg TE: {meta['avg_te']:.4f}")

    # Print summary table
    print("\n=== AIE Summary (top 10 by span) ===")
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["aie_span"], reverse=True)
    print(f"{'Spec':<25} {'AIE_span':>10} {'AIE_p1':>10} {'AIE_p2':>10} {'ANIE':>10}")
    print("-" * 70)
    for k in sorted_keys[:10]:
        r = results[k]
        print(f"{k:<25} {r['aie_span']:>10.4f} {r['aie_p1']:>10.4f} {r['aie_p2']:>10.4f} {r['anie_span']:>10.4f}")


if __name__ == "__main__":
    main()
