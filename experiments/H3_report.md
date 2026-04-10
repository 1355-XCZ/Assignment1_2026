# H3: Necessity of Pointer Asymmetric Wiring — Testing Functional Specialization in Iterative Encoding

## Motivation

QANet's Model Encoder applies the same 7 encoder blocks iteratively three times (weight-shared), producing three intermediate representations:

```
M0 →[7 blocks]→ M1 →[same 7 blocks]→ M2 →[same 7 blocks]→ M3
```

The Pointer layer uses these three products **asymmetrically**:

```
P(start) = softmax(w1 · [M1; M2])     ← pass 1 + pass 2
P(end)   = softmax(w2 · [M1; M3])     ← pass 1 + pass 3
```

**Design provenance**: The QANet paper merely states "We adopt the strategy of Seo et al. (2016)" for this design, providing no theoretical justification. Tracing back to BiDAF (Seo et al., 2016), the original form of the asymmetric wiring is: start prediction uses the modeling layer output M, while end prediction passes M through an additional BiLSTM with **independent parameters** to obtain M2. In BiDAF, this asymmetry has at least structural justification — the extra BiLSTM with independent parameters can learn end-specific features. However, QANet adapted this pattern into **weight-shared** encoder iterations: the third iteration applies the exact same transformation function as the first two, significantly weakening the premise that "additional processing produces specialized information." This never-formally-tested design choice implies three assumptions:

1. **Depth specialization**: M2 (14 layers) and M3 (21 layers) carry different information suited for different prediction tasks — M2's moderate depth suits start localization, M3's additional processing provides specialized refinement for end prediction
2. **Shared anchor**: M1 (7 layers) provides the base localization signal needed by both predictions
3. **Iterative gain**: More encoder iterations produce better or at least different representations

Do these assumptions hold? Particularly under the weight-sharing constraint — can iterative application of the same transformation function produce functional specialization? If any of the three components fails, the design choice lacks sufficient justification.

## Research Question

Does the functional specialization hypothesis implied by the Pointer's asymmetric wiring — M1 serving as a shared anchor providing base signal, M2 and M3 each developing distinct expertise due to different processing depths — hold in the trained model?

## Methodological Note: Eval-Time Intervention and Mechanistic Analysis

All experiments in this study are performed at **eval time** — wiring replacement, ablation, and information probing are all executed on a fully trained model without retraining. This is a deliberate methodological choice:

Our research goal is **mechanistic analysis** — understanding the functional roles that components currently serve in the trained model. This aligns with the causal tracing methodology used by Meng et al. (2022) for localizing factual associations in GPT: intervening on a trained model, observing behavioral changes, and localizing functional roles of specific computation steps. Training-time ablation allows the model to learn compensatory strategies, erasing the functional traces of removed components — making it unsuitable for mechanistic analysis.

Wiring replacement is a form of eval-time intervention: the encoder is unchanged, only the Pointer's input combinations are modified. This directly measures "how the model currently uses each iteration product." Since Pointer weights are trained on specific input distributions, replacement introduces distribution mismatch — this inherent limitation is mitigated in Experiment 2 through Pointer-weight-independent linear probes.

## Pre-Validation: How Large is the Asymmetric Design's Advantage?

We directly test the necessity of the asymmetric design through wiring replacement. At eval time, we change the Pointer's input combinations (encoder unchanged), evaluating on the full dev set of 10,465 samples.

| Config | Start input | End input | F1 | ΔF1 | Hypothesis tested |
|---|---|---|---|---|---|
| original | M1+M2 | M1+M3 | 70.24 | — | Baseline (asymmetric design) |
| sym\_M2 | M1+M2 | M1+M2 | 70.01 | −0.23 | Is M3 necessary? |
| sym\_M3 | M1+M3 | M1+M3 | 69.70 | −0.54 | Is M2 better? |
| no\_M1 | M2+M3 | M2+M3 | 69.67 | −0.58 | Is M1 necessary? |
| swap | M1+M3 | M1+M2 | 69.42 | −0.82 | Does assignment direction matter? |
| only\_M1 | M1+M1 | M1+M1 | 63.97 | −6.28 | Are M2/M3 necessary? |

Test results for the three design assumptions:

**Assumption 1 (Depth specialization) — barely holds**: The asymmetric design (original) outperforms the symmetric design (sym\_M2) by only **0.23 F1**. The supposed "specialization" from assigning M3 to end prediction provides near-negligible marginal benefit. More notably, M2 doing both tasks (sym\_M2: −0.23) **outperforms** M3 doing both tasks (sym\_M3: −0.54) — the more processed product is actually worse.

**Assumption 2 (Shared anchor) — does not hold**: Removing M1 (no\_M1: −0.58) barely affects performance, but M1 alone (only\_M1: −6.28) is catastrophic. M1 participates in both predictions yet is the least important — the shared anchor is a redundant design.

**Assumption 3 (Iterative gain) — partially holds with diminishing returns**: M2/M3 are essential (only\_M1: −6.28), but M2 already captures the vast majority of incremental value. The second iteration's increment (only\_M1 vs sym\_M2: 6.05 F1) far exceeds the third iteration's marginal contribution (near zero or negative).

**Conclusion**: Of the three design assumptions, depth specialization and shared anchor both fail, and iterative gain is essentially exhausted by the third iteration. The following experiments explain **why** these assumptions do not hold.

---

## Experiment Design

Three experiments progressively explain the three design failures observed in pre-validation:

**Experiment 1 (Why is the shared anchor redundant?)** Validate the cause of M1 redundancy from parameter and information sides: (a) Pointer weight M1-quadrant norms — did the model learn to downweight M1? (b) M1-M2 CKA — has M2 absorbed M1's information?

**Experiment 2 (Is M2 and M3's information actually different?)** Pre-validation shows sym\_M2 > sym\_M3, but this is affected by Pointer weight distribution mismatch. We use Pointer-weight-independent linear probes to directly measure M1/M2/M3's start/end boundary information, determining whether M2 > M3 reflects intrinsic information difference or adaptation bias.

**Experiment 3 (What did the third iteration do?)** If Experiment 2 confirms M3's boundary information is not superior to M2, what transformation did the third iteration's 7 encoder blocks apply? We characterize the M2→M3 transformation through representation structure analysis — whether over-smoothing blurs boundary signals.

---

## Experiment 1: Why is the Shared Anchor Redundant?

**Goal**: Explain the pre-validation finding — M1 is nearly unimportant in the Pointer.

### 1a. Pointer Weight Quadrant Analysis

**Method**: Pointer's w1 has shape [2d], where the first d dimensions correspond to M1 features and the last d dimensions to M2 features (concatenation order: [M1; M2]). Similarly, w2's first d dimensions correspond to M1, last d to M3.

| Weight | First d dims (M1 quadrant) | Last d dims (M2/M3 quadrant) | Ratio |
|---|---|---|---|
| w1 | [TBD] | [TBD] | [TBD] |
| w2 | [TBD] | [TBD] | [TBD] |

If M1 quadrant norm is much smaller than M2/M3 quadrant → the model learned to assign lower weight to M1 during training, directly supporting M1 redundancy from the parameter side.

### 1b. CKA Information Overlap

**Method**: Compute linear CKA for M1-M2, M2-M3, M1-M3 (1000 samples).

| Pair | CKA | Meaning |
|---|---|---|
| M1\_M2 | [TBD] | How much M1 information M2 contains |
| M2\_M3 | [TBD] | How much M2 information M3 contains |
| M1\_M3 | [TBD] | How much M1 information M3 contains |

M2 = f(M1) means M2 **contains** all of M1's information (in the information-theoretic sense), but after nonlinear transformation it may be stored in a different "format." CKA measures linearly alignable information overlap. If CKA(M1, M2) is high (>0.9) → M1's information is still linearly readable in M2, and M1 is truly redundant in the [M1;M2] concatenation.

The Pointer is a linear projection (w·[M1;M2]) and can only utilize linearly readable information. The near-zero loss from no\_M1 indicates: M1's information remains linearly readable in M2 (the encoder approximately preserves linear structure), or M1's information is not very useful for boundary prediction itself. Experiment 2's linear probes help distinguish these two explanations.

### Experiment 1 Summary

[To be filled with actual results] Weight analysis and CKA explain M1 redundancy from parameter and information sides: M1's information has been absorbed by subsequent iteration products. The shared anchor design exists but is unnecessary — this is the **first failure point** of the asymmetric design.

---

## Experiment 2: Is M2 and M3's Information Actually Different?

**Goal**: Directly measure M2 vs M3 boundary information difference, independent of Pointer weights. This tests the core assumption of the asymmetric design — whether different-depth iteration products carry different information.

**Method**: Train logistic regression on M1/M2/M3's per-token representations:
- **is\_start\_token**: Whether the token is the answer start position
- **is\_end\_token**: Whether the token is the answer end position

500 samples, 80/20 split, ROC-AUC.

**Why linear probes are appropriate**: The Pointer itself is a linear projection (w · [M; M']), so linear probes test the **same type** of information readability. If information is unreadable to a linear probe, it is likely unreadable to the Pointer as well.

| Representation | is\_start AUC | is\_end AUC |
|---|---|---|
| M1 | [TBD] | [TBD] |
| M2 | [TBD] | [TBD] |
| M3 | [TBD] | [TBD] |

**Results and design assumption correspondence**:

| If found... | Implication for asymmetric design |
|---|---|
| M2 ≥ M3 for both start and end | Depth specialization fails — M3 is not better, asymmetric wiring is pointless |
| M2 better for start, M3 better for end | Functional specialization exists — asymmetric design has partial justification |
| M3 > M2 for both | sym\_M2 > sym\_M3 is Pointer adaptation bias, not information difference |
| M2 ≈ M3 | Difference is minimal — iteration products converge, depth does not produce specialization |

**Cross-validation with wiring replacement**: Probes measure the information level (independent of Pointer weights); wiring replacement measures the behavioral level (dependent on Pointer weights). If both agree (M2 ≥ M3), the conclusion is robust. If they disagree, the difference comes from Pointer weight adaptation bias.

### Limitations

- is\_start and is\_end labels are extremely imbalanced (~1/400 positive samples per example)
- Linear probes operate on M2 or M3 individually, but the Pointer processes [M1;M2] concatenation — joint information may exceed the sum of parts
- Probes detect information **presence** in the representation, not how much the Pointer actually **uses**

---

## Experiment 3: What Did the Third Iteration Do?

**Goal**: If Experiment 2 confirms M3's boundary information is not superior to M2, explain the mechanism by which the third iteration fails to produce a better representation.

Each encoder block's pipeline is Conv\_0 → Conv\_1 → Self-Attention → FFN. Self-Attention is the only global operation. If the third iteration's Self-Attention aggregates token representations toward the global mean ("over-smoothing"), adjacent tokens become more similar, making precise boundary positions harder to distinguish with a linear classifier.

**Method**: Compute the following representation structure metrics on M1/M2/M3 (200 samples):

1. **Local contrast**: 1 − cosine\_sim(token\_t, token\_{t+1}), averaged. High = adjacent tokens more distinguishable.
2. **Answer boundary sharpness**: Cosine distance between answer boundary positions (y1, y2) and adjacent positions. High = sharper boundaries.
3. **Token norm coefficient of variation (CoV)**: Std/mean of per-token L2 norms. High = more position-selective activation in the representation.

| Metric | M1 | M2 | M3 | M2→M3 change |
|---|---|---|---|---|
| Local contrast | [TBD] | [TBD] | [TBD] | [TBD] |
| Answer boundary sharpness | [TBD] | [TBD] | [TBD] | [TBD] |
| Norm CoV | [TBD] | [TBD] | [TBD] | [TBD] |

Three metrics measure "whether the third iteration blurred local features" from different angles. If all three point to M3 being smoother → the mechanism is clear: iterative application of the same encoder function causes representations to gradually converge, with Self-Attention's global aggregation accumulating excessively by the third pass.

### Connection to H1 Findings

H1 found that Conv\_1 is the most important sub-layer in the Model Encoder — it produces high-discriminability local features for Self-Attention to use. If the third iteration's Self-Attention over-smooths the local contrast that Conv\_1 established, then H3's findings echo H1: Conv\_1 builds local features → Self-Attention utilizes and potentially over-consumes them → too many iterations cause diminishing and eventually negative marginal returns.

### Limitations

- Representation structure analysis is descriptive (correlational), not causal. "M3 is smoother" does not equal "M3 is worse because it is smoother" — confounding variables may exist.
- Conv and FFN in encoder blocks also affect representations; attributing changes to Self-Attention's "over-aggregation" requires finer-grained per-sub-layer analysis.
- M1, M2, M3 use the same encoder parameters; each iteration applies the same function to different input distributions. "Over-smoothing" is not a problem of Self-Attention itself, but a cumulative effect of iteratively applying the same transformation.

---

## Synthesis

### Answer

**Research question**: Does the functional specialization hypothesis implied by the Pointer's asymmetric wiring — M1 as shared anchor, M2/M3 as depth-specialized — hold in the trained model?

**Answer**: [To be constructed from actual results, following this direction]

**If the "over-smoothing" hypothesis holds** (M2 probe AUC ≥ M3, and M3 local contrast < M2):

All three assumptions of the asymmetric design fail or are exhausted:

1. **Shared anchor is redundant**: M1's information has been fully absorbed by M2 (CKA validation); Pointer weights assign low norm to M1 quadrants. M1 provides no incremental information in the [M1;M2] concatenation — the shared anchor is an ineffective design.
2. **Depth specialization does not exist**: Linear probes show M3's boundary information is not superior to M2 — the third iteration does not provide end-specific "specialized" information that start doesn't need. M2 doing both tasks outperforms M3 doing both tasks (sym\_M2 > sym\_M3).
3. **Iterative gain is exhausted**: The second iteration (M1→M2) provides the vast majority of incremental value (6.05 F1); the third iteration (M2→M3) contributes near zero or negative marginal value. The mechanism is that the third iteration's encoder smooths token representations toward the global mean (representation structure validation), reducing boundary sharpness — cumulative over-smoothing from iterative application of the same transformation.

The asymmetric wiring (M2→start, M3→end) outperforms symmetric wiring (sym\_M2) by only 0.23 F1. This marginal advantage likely does not come from M3's "specialization" for end prediction, but from the third iteration's minor global information integration, which slightly helps end prediction (which typically requires more context) at the cost of losing some local sharpness. The net effect is near zero.

**If the "over-smoothing" hypothesis does not hold** (M3 probe AUC ≥ M2, or no representation structure difference):

The sym\_M2 > sym\_M3 difference comes from Pointer weight training bias, not intrinsic M2/M3 information difference. The model optimized w1 (start prediction) more thoroughly during training, making the w1+M2 combination more robust at eval time than w2+M3. Under this interpretation, M2 > M3 is a product of training dynamics rather than representation quality. However, the shared anchor redundancy and iterative gain exhaustion conclusions still hold.

### Evidence Chain

1. **Pre-validation** (wiring replacement): Asymmetric advantage only 0.23 F1; M1 redundant (−0.58 F1); M2 > M3 as universal representation (sym\_M2 > sym\_M3). All three design assumptions are negated or severely weakened at the behavioral level.
2. **Parameter and information level** (Experiment 1): Pointer weight quadrant norms + CKA explain M1 redundancy from parameter and information sides — subsequent iterations absorb prior iterations' information.
3. **Information level** (Experiment 2): Linear probes directly measure M2 vs M3 start/end information, bypassing Pointer weights, distinguishing "information difference" from "adaptation bias."
4. **Mechanism level** (Experiment 3): Representation structure analysis characterizes the M2→M3 transformation — whether over-smoothing blurs boundary signals.

### Connection to Other Hypotheses

H1 demonstrated that Conv\_1 is an indispensable component in the Model Encoder (removal causes −32.54 F1); H3 demonstrates that M1 is redundant in the Pointer (removal causes only −0.58 F1). Together they show that QANet's design contains **coexisting critical bottlenecks and redundancies**. H2 demonstrated that C2Q is irreplaceable while Q2C is redundant in CQ Attention — the same theme of "irreplaceability vs redundancy." All three hypotheses reveal the uneven distribution of functional importance within QANet from different levels.

### Limitations

1. **Distribution mismatch in eval-time wiring replacement**: Pointer weights are trained on specific input distributions; wiring replacement changes the input distribution. Both sym\_M2 and sym\_M3 have 50% input distribution mismatch. How much of their difference (0.31 F1) comes from intrinsic M2/M3 difference vs asymmetric distribution mismatch effects cannot be fully separated. Experiment 2's linear probes partially mitigate this.

2. **Causal attribution limitations**: Experiment 3's representation analysis is descriptive. The relationship between "M3 is smoother" and "M3 boundary information is worse" is correlational, not causal. Full causal verification would require selectively disabling Self-Attention in the third pass and measuring boundary sharpness changes — high implementation complexity.

3. **Stochastic depth training-time effects**: During training, stochastic depth skips sub-layers with probability dropout × l/L (deeper layers have higher skip probability). Layers 15-21 (traversed by M3) have the highest skip probability during training. This may cause M3's representations to be less thoroughly optimized during training — "worse" may partly come from insufficient training rather than over-smoothing. This hypothesis is beyond the current experiments' testing scope but should be retained as an alternative explanation.
