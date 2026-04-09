# H1: Component-Level Importance in QANet's Model Encoder

## Hypothesis

> In QANet's Model Encoder, the second convolution sub-layer (Conv_1) is the most causally important component for span prediction. This importance stems from its structured information content — specifically, general-purpose local feature extraction — rather than answer-position encoding or output magnitude.

## Background

QANet's Model Encoder consists of 7 stacked Encoder Blocks, repeated for 3 passes (producing M1, M2, M3). Each block contains 4 sub-layers in sequence:

```
Position Encoding → Conv_0 → Conv_1 → Self-Attention → FFN
```

Each sub-layer has a residual connection: `output = residual + sublayer(x)`.

Conv_0 and Conv_1 are architecturally identical (depthwise separable convolution, same kernel size, same dimensions). The QANet paper (Table 5) reported that removing all convolution layers and retraining caused −2.7 F1, while removing Self-Attention caused −1.3 F1. However, the paper did not distinguish between Conv_0 and Conv_1, nor investigate the mechanism behind this difference.

---

## Experiment 1: Global Ablation (Eval-time Component Removal)

**Method**: Zero out all instances of a component type across all 7 blocks × 3 passes. Evaluate F1/EM on the full dev set (10,465 samples).

**Mechanism**: The sub-layer output is replaced with zeros before the residual addition, so downstream layers receive only the residual (skip connection).

| Config | F1 | ΔF1 | EM | Layers Removed |
|---|---|---|---|---|
| baseline | 70.24 | — | 59.22 | 0 |
| skip_conv_1 | 37.70 | −32.54 | 23.59 | 21 |
| skip_attn | 65.44 | −4.81 | 55.18 | 21 |
| skip_ffn | 66.92 | −3.32 | 54.56 | 21 |
| skip_conv_0 | 69.16 | −1.09 | 57.85 | 21 |
| skip_conv (both) | 27.14 | −43.11 | 13.60 | 42 |

**Finding 1**: The importance ranking is **Conv_1 (−32.54) >> Attn (−4.81) > FFN (−3.32) > Conv_0 (−1.09)**. Conv_1 removal is catastrophic; Conv_0 removal barely matters.

**Finding 2**: Conv_0 + Conv_1 removal (−43.11) exceeds the sum of individual removals (−32.54 + −1.09 = −33.63), showing a super-additive interaction (+9.48). When Conv_0 is also absent, Conv_1's loss becomes even more devastating.

**Limitation**: Zero-out ablation removes both the component's information and its norm contribution to the residual stream. We cannot yet distinguish which factor drives the damage.

---

## Experiment 2: ROME-Style Causal Tracing

**Method**: Adapted from Meng et al. (2022). Three-step procedure per sample:
1. **Clean run**: Collect all sub-layer activations. Record P(correct span).
2. **Corrupted run**: Add Gaussian noise (σ = 3× input std) to context embeddings. Record degraded P(correct span).
3. **Corrupted-with-restoration**: Re-run corrupted input, but restore ONE sub-layer's clean activation. Measure recovery (Average Indirect Effect, AIE).

**Configuration**: 200 samples (20 preliminary), 3 noise repeats, 84 sub-layers traced (7 blocks × 3 passes × 4 components).

### Results

| Component | Mean AIE | Sum AIE | Share of Total |
|---|---|---|---|
| Conv_1 | 0.0210 | 0.4418 | 67.1% |
| Conv_0 | 0.0047 | 0.0986 | 15.0% |
| FFN | 0.0040 | 0.0844 | 12.8% |
| Self-Attn | 0.0016 | 0.0343 | 5.2% |

AIE by pass: Pass 0 (M1) = 0.3395 > Pass 1 (M2) = 0.2324 > Pass 2 (M3) = 0.0871.

AIE by block depth: Block 0 (0.2282) > Block 1 (0.1574) > ... > Block 6 (0.0371). Monotonically decreasing.

**Finding 3**: Causal tracing confirms Conv_1 dominance. Restoring Conv_1's clean activation recovers the most information. The effect is concentrated in shallow blocks and early passes.

---

## Experiment 3: Norm Diagnostic

**Motivation**: The correlation between sub-layer output L2 norm and AIE could indicate that causal tracing measures magnitude rather than importance.

**Method**: Compute per-sub-layer output L2 norms across 200 clean forward passes.

| Component | Mean Norm | Ratio vs Conv_0 |
|---|---|---|
| Conv_1 | 250.47 | 2.22x |
| Conv_0 | 112.77 | 1.00x |
| FFN | 103.50 | 0.92x |
| Self-Attn | 81.84 | 0.73x |

**Pearson r(norm, AIE) = 0.895** across 84 sub-layers. 80% of AIE variance is explained by output norm.

### Is this a confound?

**For Conv_1 vs Attn**: Both norm and AIE favor Conv_1. We cannot separate the two for this comparison.

**For Attn vs FFN vs Conv_0**: The norm ordering is Conv_0 (113) > FFN (104) > Attn (82), but the ablation damage ordering is Attn (−4.81) > FFN (−3.32) > Conv_0 (−1.09). **Norm and importance are inversely related** for these three components. This is strong evidence that norm does not drive the ranking among the non-Conv_1 components.

**Finding 4**: The ranking Attn > FFN > Conv_0 is robust to norm effects (anti-correlated). The magnitude of Conv_1's advantage over Attn is uncertain due to the shared confound.

---

## Experiment 4: Zero-Out vs Noise Replacement

**Motivation**: Directly separate information from magnitude.

**Method**: Compare two interventions:
- **Zero-out**: Replace output with zeros (removes information AND norm).
- **Noise replacement**: Replace output with random noise of identical L2 norm (removes information, PRESERVES norm).

**Logic**:
- If norm drives importance → noise (preserving norm) should be LESS damaging than zero.
- If information drives importance → noise should be SIMILARLY or MORE damaging than zero (noise also removes information, and adds interference).

| Component | Zero ΔF1 | Noise ΔF1 | Noise/Zero Ratio |
|---|---|---|---|
| Conv_1 | −29.27 | −41.29 | 1.41x |
| Conv_0 | −2.23 | −2.70 | 1.21x |
| Attn | −5.20 | −5.89 | 1.13x |
| FFN | −3.90 | −4.72 | 1.21x |

(Evaluated on 2,048 subsampled dev set.)

**Finding 5**: Noise is MORE damaging than zero-out for ALL components. This rejects the norm hypothesis. If norm were what mattered, preserving it (via noise) should help. Instead, norm-matched noise actively corrupts the residual stream because it injects structured garbage that interferes with downstream processing.

**Interpretation**: All four components produce structured, useful information. The damage from removal comes from losing this information, not from losing magnitude. For Conv_1, the noise/zero ratio is highest (1.41x) because its large norm means the injected garbage has the largest impact.

---

## Experiment 5: Linear Probe — What Information Does Conv_1 Encode?

**Motivation**: Given that Conv_1's information is critical (Experiment 4), what is this information? Specifically, does Conv_1 encode answer-position signals?

**Method**: Train a logistic regression classifier on each sub-layer's per-token output vectors. Label: 1 = token is inside the answer span, 0 = otherwise. Evaluate ROC-AUC (robust to class imbalance, ~2.3% positive rate). 500 samples, 80/20 train/test split, probed at 5 representative (pass, block) positions.

### Results: Average AUC by Component Type

| Component | Mean AUC (5 blocks) |
|---|---|
| Conv_1 | 0.901 |
| FFN | 0.898 |
| Self-Attn | 0.894 |
| Conv_0 | 0.876 |

### Results: AUC by Block Location (averaged over 4 components)

| Location | Mean AUC |
|---|---|
| p0_b0 | 0.822 |
| p0_b3 | 0.901 |
| p0_b6 | 0.907 |
| p1_b0 | 0.910 |
| p2_b0 | 0.922 |

**Finding 6**: All four components encode answer-position information to a similar degree. Conv_1's AUC advantage over Attn is only +0.007. This is in stark contrast to the ablation results where Conv_1 is 7x more damaging to remove.

**Finding 7**: Answer-position information accumulates with depth (AUC increases from 0.822 at p0_b0 to 0.922 at p2_b0), and is not concentrated in any single component.

**Interpretation**: Conv_1's importance does NOT come from encoding "where the answer is." All sub-layers encode this information comparably. Conv_1's critical role must lie elsewhere — in the quality and structure of its general-purpose local features.

---

## Experiment 6: Per-Block Ablation

**Method**: Remove conv (both conv_0 and conv_1) or self_attn in a single (pass, block), measure F1 on 2,048 subsampled dev set. 42 configurations total. Baseline F1 on subset = 68.01.

### Key Results

| Config | F1 | ΔF1 |
|---|---|---|
| p0_b0_conv | 66.73 | −1.28 |
| p0_b0_self_attn | 67.35 | −0.66 |
| All other configs | ~67.7–68.2 | −0.3 to +0.2 |

**Finding 8**: Individual block removal has minimal impact (max ΔF1 = −1.28). Only p0_b0_conv shows a clearly significant effect. Most blocks are individually redundant.

**Finding 9**: Collective removal is catastrophic (−43.11 for all conv), but individual removal is negligible. This reveals a **distributed, redundant architecture**: each block contributes a small increment that is individually dispensable but collectively essential. The 7-block × 3-pass design provides massive redundancy.

**Finding 10**: Average ΔF1 by pass: M1 (−0.23) >> M2 (−0.03) >> M3 (+0.01). Pass 2 (M3) blocks can be individually removed with zero or slightly positive effect, suggesting marginal computational value at the individual block level.

---

## Synthesis: What Is Conv_1's Role?

Combining all six experiments:

| Evidence | What it shows |
|---|---|
| Global ablation | Conv_1 removal is catastrophic (−32.54 F1) |
| Causal tracing | Conv_1 carries most restorable information (67% of AIE) |
| Norm diagnostic | Conv_1 has 2.2x larger output norm; r(norm, AIE)=0.895 |
| Noise replacement | Information matters, not norm (noise worse than zero) |
| Linear probe | Conv_1 does NOT specifically encode answer positions |
| Per-block ablation | Individual blocks are redundant; collective effect is critical |

**Conclusion**: Conv_1 functions as the **primary local feature amplifier** in QANet's Model Encoder. It produces high-norm, information-rich representations that serve as the substrate for downstream Self-Attention and FFN processing. Its critical role is not about detecting answer boundaries (all components do this comparably), but about providing the foundational local features without which the entire downstream pipeline degrades.

Conv_0, despite identical architecture, learns a minimal preprocessing role (low norm, low ablation impact). The asymmetry between Conv_0 and Conv_1 arises from training dynamics: the model converges to a configuration where one convolution layer does most of the work, while the other becomes near-redundant.

### Limitations

1. **Conv_1 vs Attn magnitude**: The exact ratio of Conv_1's importance over Attn is uncertain (2–7x range) due to shared norm correlation.
2. **Eval-time ablation vs retraining**: Our zero-out intervention is harsher than the paper's retrain-after-removal approach, inflating absolute damage numbers.
3. **Probe scope**: The linear probe tests only answer-position encoding. Conv_1 may encode other types of information (syntactic patterns, entity boundaries, local context) that a binary answer/non-answer probe cannot detect.
4. **Causal tracing sample size**: 20 preliminary samples have wide confidence intervals. A full 200-sample run would strengthen the estimates.

### Novel Contributions

1. **Conv_0 vs Conv_1 dissociation**: First demonstration that two architecturally identical stacked convolutions learn drastically different roles (−1.09 vs −32.54 F1 on removal).
2. **Noise replacement control**: Establishes that component importance reflects information content, not output magnitude, resolving the norm confound.
3. **Linear probe negative result**: Shows Conv_1's importance is NOT about answer-position encoding, ruling out the most obvious explanation and pointing toward general-purpose feature extraction.
4. **Distributed redundancy**: Quantifies the gap between individual block dispensability (max −1.28 F1) and collective necessity (−43.11 F1), characterizing QANet's fault-tolerance architecture.
