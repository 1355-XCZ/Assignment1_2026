"""
run_H2.py — H2: Context vs Question Dual-Stream Information Flow

Measures the causal contribution of Context vs Question encoding
and the CQ Attention fusion bottleneck.

Usage (from project root):
    python -m experiments.run_H2 --ckpt _model/model.pt --num_samples 300
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
    qanet_forward, compute_span_prob, compute_ie,
    build_emb_enc_specs, build_fusion_specs, RestoreSpec,
)

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


@torch.no_grad()
def run_h2(model, dataset, num_samples=300, noise_std_scale=3.0,
           noise_repeats=3, min_clean_prob=0.01, seed=42):
    """
    Run H2 dual-stream causal tracing.

    Three corruption conditions × multiple restoration targets.

    Returns
    -------
    results : dict structured by corruption condition
    meta    : experiment metadata
    """
    random.seed(seed)
    torch.manual_seed(seed)

    loader = make_loader(dataset, batch_size=1, shuffle=True)

    # Build all restoration specs
    emb_specs = build_emb_enc_specs()   # C and Q Embedding Encoder sub-layers
    fusion_specs = build_fusion_specs()  # CQ Attention, CQ Resizer

    # For CORRUPT-CQ, also test restoring both C and Q encodings simultaneously
    # This requires a special two-stage restoration (handled separately)

    corrupt_conditions = ["context", "question", "both"]

    # Accumulators
    te_acc = {cc: [] for cc in corrupt_conditions}
    ie_acc = {cc: {} for cc in corrupt_conditions}

    def _spec_key(spec):
        return f"{spec.stage}_{spec.component}"

    all_specs = emb_specs + fusion_specs
    all_keys = [_spec_key(s) for s in all_specs]

    for cc in corrupt_conditions:
        for k in all_keys:
            ie_acc[cc][k] = []

    # Additional: combined C+Q restoration under CORRUPT-both
    ie_acc["both"]["restore_C_and_Q"] = []

    samples_used = 0
    for batch_idx, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(
        tqdm(loader, total=num_samples, desc="H2 Dual-Stream Tracing")
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

        for cc in corrupt_conditions:
            sample_te = []
            sample_ie = {k: [] for k in all_keys}
            sample_ie_cq = []  # combined C+Q restore

            for rep in range(noise_repeats):
                noise_seed = seed + batch_idx * 100 + rep

                # 2. Corrupted run
                p1_x, p2_x, _, _ = qanet_forward(
                    model, Cwid, Ccid, Qwid, Qcid,
                    corrupt_target=cc,
                    noise_std_scale=noise_std_scale,
                    noise_seed=noise_seed,
                )
                prob_corrupt = compute_span_prob(p1_x, p2_x, y1, y2)
                te = (prob_clean - prob_corrupt).item()
                sample_te.append(te)

                if abs(te) < 1e-6:
                    continue

                # 3. Restoration runs
                for spec, key in zip(all_specs, all_keys):
                    # Skip restoring Q sub-layers when only C is corrupted (no effect)
                    if cc == "context" and spec.stage == "emb_enc_Q":
                        continue
                    # Skip restoring C sub-layers when only Q is corrupted
                    if cc == "question" and spec.stage == "emb_enc_C":
                        continue

                    p1_r, p2_r, _, _ = qanet_forward(
                        model, Cwid, Ccid, Qwid, Qcid,
                        corrupt_target=cc,
                        noise_std_scale=noise_std_scale,
                        noise_seed=noise_seed,
                        clean_acts=clean_acts,
                        restore_spec=spec,
                    )
                    prob_r = compute_span_prob(p1_r, p2_r, y1, y2)
                    ie = (prob_r - prob_corrupt).item()
                    sample_ie[key].append(ie)

                # 4. Combined C+Q restoration (only for "both" condition)
                #    Restore emb_enc_C output, then emb_enc_Q output
                #    This requires two sequential restorations — approximate by
                #    restoring the CQ Attention input (Ce_clean, Qe_clean)
                #    which is equivalent to restoring both encoder outputs.
                if cc == "both":
                    # Restoring CQ-ATT output effectively restores the fusion of
                    # clean Ce and Qe, which is a good proxy for "restore both"
                    spec_cq = RestoreSpec("cq_att")
                    p1_r, p2_r, _, _ = qanet_forward(
                        model, Cwid, Ccid, Qwid, Qcid,
                        corrupt_target=cc,
                        noise_std_scale=noise_std_scale,
                        noise_seed=noise_seed,
                        clean_acts=clean_acts,
                        restore_spec=spec_cq,
                    )
                    prob_r = compute_span_prob(p1_r, p2_r, y1, y2)
                    ie_cq_both = (prob_r - prob_corrupt).item()
                    sample_ie_cq.append(ie_cq_both)

            # Average over repeats
            avg_te = np.mean(sample_te) if sample_te else 0.0
            te_acc[cc].append(avg_te)

            for key in all_keys:
                if sample_ie[key]:
                    ie_acc[cc][key].append(np.mean(sample_ie[key]))

            if cc == "both" and sample_ie_cq:
                ie_acc["both"]["restore_C_and_Q"].append(np.mean(sample_ie_cq))

        samples_used += 1

    # Aggregate results
    results = {}
    for cc in corrupt_conditions:
        te_arr = np.array(te_acc[cc])
        n_te = len(te_arr)
        ci = 1.96 / np.sqrt(n_te) if n_te > 1 else 0.0

        results[cc] = {
            "total_effect": {
                "mean": float(te_arr.mean()) if n_te > 0 else 0.0,
                "ci95": float(te_arr.std() * ci) if n_te > 0 else 0.0,
                "n": n_te,
            },
            "indirect_effects": {},
        }

        for key in list(all_keys) + (["restore_C_and_Q"] if cc == "both" else []):
            vals = np.array(ie_acc[cc].get(key, []))
            n = len(vals)
            ci_val = 1.96 / np.sqrt(n) if n > 1 else 0.0
            if n > 0:
                results[cc]["indirect_effects"][key] = {
                    "aie": float(vals.mean()),
                    "ci95": float(vals.std() * ci_val),
                    "nie": float(vals.mean() / max(abs(te_arr.mean()), 1e-8)),
                    "n": n,
                }

    # Additivity test (for "both" condition)
    # Compare IE(restore_C) + IE(restore_Q) vs IE(restore_CQ_att)
    additivity = {}
    if "both" in results:
        ie_c_output = results["both"]["indirect_effects"].get("emb_enc_C_output", {}).get("aie", 0)
        ie_q_output = results["both"]["indirect_effects"].get("emb_enc_Q_output", {}).get("aie", 0)
        ie_cq_att = results["both"]["indirect_effects"].get("cq_att_output", {}).get("aie", 0)
        ie_combined = results["both"]["indirect_effects"].get("restore_C_and_Q", {}).get("aie", 0)
        additivity = {
            "ie_C_plus_ie_Q": ie_c_output + ie_q_output,
            "ie_cq_att": ie_cq_att,
            "ie_combined_proxy": ie_combined,
            "superadditive": ie_combined > (ie_c_output + ie_q_output),
        }

    meta = {
        "num_samples_used": samples_used,
        "noise_std_scale": noise_std_scale,
        "noise_repeats": noise_repeats,
        "additivity_test": additivity,
    }

    return results, meta


def main():
    parser = argparse.ArgumentParser(description="H2: Dual-Stream Information Flow")
    parser.add_argument("--ckpt", type=str, default="_model/model.pt")
    parser.add_argument("--dev_npz", type=str, default="_data/dev.npz")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--noise_std_scale", type=float, default=3.0)
    parser.add_argument("--noise_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments/results/H2")
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    model, model_args = load_model(args.ckpt)

    dataset = SQuADDataset(args.dev_npz)
    print(f"Dataset: {len(dataset)} samples")

    results, meta = run_h2(
        model, dataset,
        num_samples=args.num_samples,
        noise_std_scale=args.noise_std_scale,
        noise_repeats=args.noise_repeats,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "h2_results.json")
    with open(output_path, "w") as f:
        json.dump({"results": results, "meta": meta}, f, indent=2)
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n=== Total Effect by Corruption Condition ===")
    for cc in ["context", "question", "both"]:
        te = results[cc]["total_effect"]
        print(f"  {cc:>10s}: TE = {te['mean']:.4f} ± {te['ci95']:.4f}")

    print("\n=== CQ Attention Recovery Ratio ===")
    for cc in ["context", "question", "both"]:
        cq_ie = results[cc]["indirect_effects"].get("cq_att_output", {})
        te_mean = results[cc]["total_effect"]["mean"]
        if cq_ie and te_mean:
            ratio = cq_ie["aie"] / abs(te_mean)
            print(f"  {cc:>10s}: R_CQ = {ratio:.2%}")

    if meta["additivity_test"]:
        a = meta["additivity_test"]
        print(f"\n=== Additivity Test (CORRUPT-both) ===")
        print(f"  IE(C) + IE(Q) = {a['ie_C_plus_ie_Q']:.4f}")
        print(f"  IE(CQ_att)    = {a['ie_cq_att']:.4f}")
        print(f"  Superadditive = {a['superadditive']}")


if __name__ == "__main__":
    main()
