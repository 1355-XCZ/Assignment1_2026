"""
tracer.py — Core causal tracing utilities for QANet.

Adapts the Causal Tracing methodology from Meng et al. (NeurIPS 2022)
to the QANet architecture (Conv + Self-Attention + FFN encoder blocks).

The key idea: re-implement QANet's forward pass with explicit intervention
points so we can corrupt inputs and selectively restore sub-layer outputs.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal


# ---------------------------------------------------------------------------
# Intervention specification
# ---------------------------------------------------------------------------

@dataclass
class RestoreSpec:
    """Specifies which sub-layer output to restore with clean activations."""
    stage: Literal["emb_enc_C", "emb_enc_Q", "cq_att", "cq_resized", "model_enc"]
    pass_idx: int = 0          # 0/1/2 for Model Encoder passes
    block_idx: int = 0         # 0-6 for Model Encoder blocks, 0 for Emb Encoder
    component: str = "output"  # "conv_0", "conv_1", "self_attn", "ffn", "output"


@dataclass
class SkipSpec:
    """Specifies which components to zero-out (ablate) in the Model Encoder.

    global_skip: set of component types to skip everywhere, e.g. {"conv", "self_attn", "ffn"}.
        "conv" skips all conv_i layers.
    block_skip: optional (pass_idx, block_idx, component_type) to skip in ONE block only.
        component_type is "conv", "self_attn", or "ffn".
    mode: "zero" (default) replaces output with zeros.
          "noise" replaces output with norm-matched random noise (same L2 norm, random direction).
    """
    global_skip: Optional[set] = None
    block_skip: Optional[tuple] = None  # (pass_idx, block_idx, "conv"|"self_attn"|"ffn")
    mode: str = "zero"  # "zero", "noise", or "mean"
    mean_acts: Optional[dict] = None  # {(pass_idx, block_idx, comp): tensor} for mode="mean"


# ---------------------------------------------------------------------------
# Encoder block forward with collection / restoration
# ---------------------------------------------------------------------------

def _encoder_block_forward(block, x, mask, l, total_layers,
                           collect: bool = False,
                           clean_acts: Optional[dict] = None,
                           restore_component: Optional[str] = None,
                           skip_components: Optional[set] = None,
                           skip_mode: str = "zero",
                           skip_mean_acts: Optional[dict] = None,
                           block_id: Optional[tuple] = None):
    """
    Replicate EncoderBlock.forward() with optional activation
    collection, single-component restoration, and component ablation.

    Parameters
    ----------
    block : EncoderBlock
    x : [B, d_model, L]
    mask : [B, L]
    l : sublayer counter start
    total_layers : total sublayers (for stochastic depth; 0 in eval)
    collect : if True, record each sub-layer output into `collected`
    clean_acts : dict of clean sub-layer outputs (needed when restoring)
    restore_component : which component to restore (None = no restore)
    skip_components : set of component names to zero-out for ablation,
        e.g. {"conv"} zeros all conv_i, {"self_attn"}, {"ffn"}

    Returns
    -------
    out : [B, d_model, L]
    collected : dict of sub-layer outputs (empty if collect=False)
    """
    collected = {}
    _skip = skip_components or set()
    drop_scale = block.dropout / total_layers if total_layers > 0 else 0.0

    def _ablate(tensor, comp_name=None):
        if skip_mode == "mean" and skip_mean_acts and block_id:
            key = (*block_id, comp_name)
            if key in skip_mean_acts:
                mean_val = skip_mean_acts[key]
                L_cur = tensor.shape[-1]
                mv = mean_val[..., :L_cur] if mean_val.shape[-1] >= L_cur else F.pad(mean_val, (0, L_cur - mean_val.shape[-1]))
                if mv.dim() < tensor.dim():
                    mv = mv.unsqueeze(0).expand_as(tensor)
                return mv
        if skip_mode == "noise":
            noise = torch.randn_like(tensor)
            norm_orig = tensor.norm()
            norm_noise = noise.norm().clamp(min=1e-8)
            return noise * (norm_orig / norm_noise)
        return torch.zeros_like(tensor)

    out = block.pos(x)

    for i, conv in enumerate(block.convs):
        res = out
        out = block.normb(out) if i == 0 else block.norms[i - 1](out)
        if i % 2 == 0:
            out = F.dropout(out, p=block.dropout, training=block.training)
        out = conv(out)
        out = block.act(out)

        comp_name = f"conv_{i}"
        if collect:
            collected[comp_name] = out.detach().clone()
        if restore_component == comp_name and clean_acts is not None:
            out = clean_acts[comp_name]
        if "conv" in _skip or comp_name in _skip:
            out = _ablate(out, comp_name)

        out = block._layer_dropout(out, res, drop_scale * l)
        l += 1

    # Self-Attention
    res = out
    out = block.norms[block.conv_num - 1](out)
    out = F.dropout(out, p=block.dropout, training=block.training)
    out = block.self_att(out, mask)

    if collect:
        collected["self_attn"] = out.detach().clone()
    if restore_component == "self_attn" and clean_acts is not None:
        out = clean_acts["self_attn"]
    if "self_attn" in _skip:
        out = _ablate(out, "self_attn")

    out = block._layer_dropout(out, res, drop_scale * l)
    l += 1

    # FFN
    res = out
    out = block.norme(out)
    out = F.dropout(out, p=block.dropout, training=block.training)
    out = out.transpose(1, 2)
    out = block.fc1(out)
    out = block.act(out)
    out = block.fc2(out)
    out = out.transpose(1, 2)

    if collect:
        collected["ffn"] = out.detach().clone()
    if restore_component == "ffn" and clean_acts is not None:
        out = clean_acts["ffn"]
    if "ffn" in _skip:
        out = _ablate(out, "ffn")

    out = block._layer_dropout(out, res, drop_scale * l)

    if collect:
        collected["output"] = out.detach().clone()

    return out, collected


# ---------------------------------------------------------------------------
# Full QANet forward with arbitrary intervention
# ---------------------------------------------------------------------------

def qanet_forward(
    model,
    Cwid, Ccid, Qwid, Qcid,
    # Corruption
    corrupt_target: Optional[Literal["context", "question", "both"]] = None,
    noise_std_scale: float = 3.0,
    noise_seed: Optional[int] = None,
    corrupt_mask_c: Optional[torch.Tensor] = None,
    # Collection
    collect: bool = False,
    # Restoration
    clean_acts: Optional[dict] = None,
    restore_spec: Optional[RestoreSpec] = None,
    # Ablation
    skip_spec: Optional['SkipSpec'] = None,
    skip_cq_att: bool = False,
    zero_cq_quadrants: Optional[List[int]] = None,
):
    """
    Run QANet with optional corruption and/or restoration.

    Returns
    -------
    p1, p2 : logits  [B, L]
    activations : dict (only populated when collect=True)
    intermediates : dict with keys "M1", "M2", "M3" (always populated)
    """
    device = Cwid.device
    acts = {}  # only filled when collect=True

    cmask = (Cwid == 0)
    qmask = (Qwid == 0)

    # ── Embedding ──────────────────────────────────────────────────────
    Cw, Cc = model.word_emb(Cwid), model.char_emb(Ccid)
    Qw, Qc = model.word_emb(Qwid), model.char_emb(Qcid)

    C, Q = model.emb(Cc, Cw), model.emb(Qc, Qw)
    C = model.proj_conv(C)
    Q = model.proj_conv(Q)

    # ── Corruption ─────────────────────────────────────────────────────
    if corrupt_target is not None:
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
        if corrupt_target in ("context", "both"):
            noise_std = noise_std_scale * C.std().item()
            noise = torch.randn_like(C) * noise_std
            if corrupt_mask_c is not None:
                mask = corrupt_mask_c.unsqueeze(1).float()
                noise = noise * mask
            C = C + noise
        if corrupt_target in ("question", "both"):
            noise_std = noise_std_scale * Q.std().item()
            Q = Q + torch.randn_like(Q) * noise_std

    # ── Embedding Encoder ──────────────────────────────────────────────
    emb_total = model.emb_enc.conv_num + 2

    # Context through Embedding Encoder
    _collect_C = collect
    _restore_C = (restore_spec and restore_spec.stage == "emb_enc_C")
    _clean_C = clean_acts.get("emb_enc_C") if (clean_acts and _restore_C) else None
    _comp_C = restore_spec.component if _restore_C else None

    Ce, Ce_acts = _encoder_block_forward(
        model.emb_enc, C, cmask, l=1, total_layers=emb_total,
        collect=_collect_C,
        clean_acts=_clean_C,
        restore_component=_comp_C,
    )
    if _collect_C:
        acts["emb_enc_C"] = Ce_acts

    # Restore full Embedding Encoder C output
    if (_restore_C and restore_spec.component == "output"
            and clean_acts and "emb_enc_C" in clean_acts):
        Ce = clean_acts["emb_enc_C"]["output"]

    # Question through Embedding Encoder
    _collect_Q = collect
    _restore_Q = (restore_spec and restore_spec.stage == "emb_enc_Q")
    _clean_Q = clean_acts.get("emb_enc_Q") if (clean_acts and _restore_Q) else None
    _comp_Q = restore_spec.component if _restore_Q else None

    Qe, Qe_acts = _encoder_block_forward(
        model.emb_enc, Q, qmask, l=1, total_layers=emb_total,
        collect=_collect_Q,
        clean_acts=_clean_Q,
        restore_component=_comp_Q,
    )
    if _collect_Q:
        acts["emb_enc_Q"] = Qe_acts

    if (_restore_Q and restore_spec.component == "output"
            and clean_acts and "emb_enc_Q" in clean_acts):
        Qe = clean_acts["emb_enc_Q"]["output"]

    # ── CQ Attention ───────────────────────────────────────────────────
    X = model.cq_att(Ce, Qe, cmask, qmask)

    if collect:
        acts["cq_att"] = X.detach().clone()
    if skip_cq_att:
        X = torch.zeros_like(X)
    if zero_cq_quadrants:
        d = X.size(1) // 4
        for q in zero_cq_quadrants:
            X[:, q * d:(q + 1) * d, :] = 0
    if (restore_spec and restore_spec.stage == "cq_att"
            and clean_acts and "cq_att" in clean_acts):
        X = clean_acts["cq_att"]

    # ── CQ Resizer ─────────────────────────────────────────────────────
    M0 = model.cq_resizer(X)

    if collect:
        acts["cq_resized"] = M0.detach().clone()
    if (restore_spec and restore_spec.stage == "cq_resized"
            and clean_acts and "cq_resized" in clean_acts):
        M0 = clean_acts["cq_resized"]

    # ── Model Encoder (3 passes) ───────────────────────────────────────
    num_blks = len(model.model_enc_blks)
    sub_per_blk = model.model_enc_blks[0].conv_num + 2
    model_total = sub_per_blk * num_blks

    if collect:
        acts["model_enc"] = {}

    passes_out = []
    current = M0

    for pass_idx in range(3):
        if collect:
            acts["model_enc"][f"pass_{pass_idx}"] = {}

        for blk_idx, enc in enumerate(model.model_enc_blks):
            # Determine if this specific (pass, block) needs restoration
            _is_target = (
                restore_spec is not None
                and restore_spec.stage == "model_enc"
                and restore_spec.pass_idx == pass_idx
                and restore_spec.block_idx == blk_idx
            )
            _blk_clean = None
            _blk_comp = None
            if _is_target and clean_acts:
                key = f"pass_{pass_idx}"
                blk_key = f"block_{blk_idx}"
                _blk_clean = clean_acts.get("model_enc", {}).get(key, {}).get(blk_key)
                _blk_comp = restore_spec.component

            # Determine skip components for this block
            _blk_skip = None
            if skip_spec is not None:
                if skip_spec.global_skip:
                    _blk_skip = skip_spec.global_skip
                elif skip_spec.block_skip:
                    sp, sb, sc = skip_spec.block_skip
                    if sp == pass_idx and sb == blk_idx:
                        _blk_skip = {sc}

            _skip_mode = skip_spec.mode if (skip_spec is not None and _blk_skip) else "zero"
            _skip_means = skip_spec.mean_acts if (skip_spec is not None and skip_spec.mean_acts) else None

            current, blk_acts = _encoder_block_forward(
                enc, current, cmask,
                l=blk_idx * sub_per_blk + 1,
                total_layers=model_total,
                collect=collect,
                clean_acts=_blk_clean,
                restore_component=_blk_comp,
                skip_components=_blk_skip,
                skip_mode=_skip_mode,
                skip_mean_acts=_skip_means,
                block_id=(pass_idx, blk_idx),
            )

            if collect:
                acts["model_enc"][f"pass_{pass_idx}"][f"block_{blk_idx}"] = blk_acts

        passes_out.append(current.detach().clone() if collect else current)

        if collect:
            acts["model_enc"][f"pass_{pass_idx}"]["output"] = current.detach().clone()

        # M2 starts from M1, M3 starts from M2 (no clone needed for forward)
        # current continues to next pass

    M1, M2, M3 = passes_out[0], passes_out[1], passes_out[2]

    # ── Pointer ────────────────────────────────────────────────────────
    p1, p2 = model.out(M1, M2, M3, cmask)

    intermediates = {"M1": M1, "M2": M2, "M3": M3}
    return p1, p2, acts, intermediates


# ---------------------------------------------------------------------------
# Probability and IE utilities
# ---------------------------------------------------------------------------

def compute_span_prob(p1, p2, y1, y2):
    """
    Compute P(correct span) = softmax(p1)[y1] * softmax(p2)[y2].
    Returns tensor of shape [B].
    """
    prob_p1 = F.softmax(p1, dim=1).gather(1, y1.unsqueeze(1)).squeeze(1)
    prob_p2 = F.softmax(p2, dim=1).gather(1, y2.unsqueeze(1)).squeeze(1)
    return prob_p1 * prob_p2


def compute_start_prob(p1, y1):
    """P(correct start) = softmax(p1)[y1]. Returns [B]."""
    return F.softmax(p1, dim=1).gather(1, y1.unsqueeze(1)).squeeze(1)


def compute_end_prob(p2, y2):
    """P(correct end) = softmax(p2)[y2]. Returns [B]."""
    return F.softmax(p2, dim=1).gather(1, y2.unsqueeze(1)).squeeze(1)


def compute_ie(prob_restored, prob_corrupt):
    """Indirect Effect (per sample). Returns [B]."""
    return prob_restored - prob_corrupt


def compute_nie(prob_restored, prob_corrupt, prob_clean):
    """Normalized Indirect Effect. Returns [B]."""
    te = prob_clean - prob_corrupt
    ie = prob_restored - prob_corrupt
    return ie / te.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Convenience: run one full causal tracing measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def trace_single_sample(
    model, Cwid, Ccid, Qwid, Qcid, y1, y2,
    corrupt_target: str = "context",
    noise_std_scale: float = 3.0,
    noise_seed: int = 42,
):
    """
    Run full causal tracing for a single batch.

    Returns
    -------
    dict with keys:
        prob_clean   : [B]
        prob_corrupt : [B]
        te           : [B]
        results      : list of dicts, each with {spec, ie_span, ie_p1, ie_p2, nie_span}
        clean_acts   : the collected clean activations (for reuse)
    """
    device = Cwid.device

    # 1. Clean run — collect all activations
    p1_c, p2_c, clean_acts, _ = qanet_forward(
        model, Cwid, Ccid, Qwid, Qcid, collect=True,
    )
    prob_clean = compute_span_prob(p1_c, p2_c, y1, y2)
    prob_p1_clean = compute_start_prob(p1_c, y1)
    prob_p2_clean = compute_end_prob(p2_c, y2)

    # 2. Corrupted run
    p1_x, p2_x, _, _ = qanet_forward(
        model, Cwid, Ccid, Qwid, Qcid,
        corrupt_target=corrupt_target,
        noise_std_scale=noise_std_scale,
        noise_seed=noise_seed,
    )
    prob_corrupt = compute_span_prob(p1_x, p2_x, y1, y2)
    prob_p1_corrupt = compute_start_prob(p1_x, y1)
    prob_p2_corrupt = compute_end_prob(p2_x, y2)

    te = prob_clean - prob_corrupt

    # 3. Enumerate all restore specs and measure IE
    results = []

    def _measure(spec):
        p1_r, p2_r, _, _ = qanet_forward(
            model, Cwid, Ccid, Qwid, Qcid,
            corrupt_target=corrupt_target,
            noise_std_scale=noise_std_scale,
            noise_seed=noise_seed,
            clean_acts=clean_acts,
            restore_spec=spec,
        )
        prob_r = compute_span_prob(p1_r, p2_r, y1, y2)
        ie_span = compute_ie(prob_r, prob_corrupt)
        ie_p1 = compute_ie(compute_start_prob(p1_r, y1), prob_p1_corrupt)
        ie_p2 = compute_ie(compute_end_prob(p2_r, y2), prob_p2_corrupt)
        nie_span = compute_nie(prob_r, prob_corrupt, prob_clean)
        return {
            "spec": spec,
            "ie_span": ie_span.cpu(),
            "ie_p1": ie_p1.cpu(),
            "ie_p2": ie_p2.cpu(),
            "nie_span": nie_span.cpu(),
        }

    return prob_clean, prob_corrupt, te, _measure, clean_acts


def build_model_enc_specs():
    """Build list of RestoreSpecs for all Model Encoder sub-layers."""
    specs = []
    for p in range(3):
        for b in range(7):
            for comp in ["conv_0", "conv_1", "self_attn", "ffn"]:
                specs.append(RestoreSpec("model_enc", pass_idx=p, block_idx=b, component=comp))
    return specs


def build_emb_enc_specs():
    """Build list of RestoreSpecs for Embedding Encoder sub-layers (C and Q)."""
    specs = []
    for stage in ["emb_enc_C", "emb_enc_Q"]:
        for comp in ["conv_0", "conv_1", "conv_2", "conv_3", "self_attn", "ffn", "output"]:
            specs.append(RestoreSpec(stage, component=comp))
    return specs


def build_fusion_specs():
    """Build RestoreSpecs for CQ Attention and CQ Resizer."""
    return [
        RestoreSpec("cq_att"),
        RestoreSpec("cq_resized"),
    ]
