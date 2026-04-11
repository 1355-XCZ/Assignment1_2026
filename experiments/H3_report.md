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

**Note on swap**: swap (−0.82) is the largest drop among non-only\_M1 configs, but this does not indicate M2/M3 carry different information. swap causes **both** w1 and w2 to receive out-of-distribution inputs simultaneously (w1 trained on [M1;M2] but receives [M1;M3]; w2 vice versa), whereas sym\_M2 only mismatches w2 (−0.23) and sym\_M3 only mismatches w1 (−0.54). If the mismatch effects are independent, expected swap ≈ 0.23 + 0.54 = 0.77, actual 0.82 — highly consistent. The swap loss comes from **dual distribution mismatch**, not M2/M3 information difference.

**Conclusion**: Of the three design assumptions, the expected benefits of depth specialization and shared anchor do not manifest at the behavioral level, and iterative gain is essentially exhausted by the third iteration. The following experiments explore **why** these assumptions are not supported.

---

## Experiment Design

Three experiments progressively explain the three design failures observed in pre-validation:

**Experiment 1 (Why is the shared anchor redundant?)** Validate the cause of M1 redundancy from parameter and information sides: (a) Pointer weight M1-quadrant norms — did the model learn to downweight M1? (b) M1-M2 CKA — has M2 absorbed M1's information?

**Experiment 2 (Is M2 and M3's information actually different?)** Pre-validation shows sym\_M2 > sym\_M3, but this is affected by Pointer weight distribution mismatch. We use Pointer-weight-independent linear probes to directly measure M1/M2/M3's start/end boundary information, determining whether M2 > M3 reflects intrinsic information difference or adaptation bias.

**Experiment 3 (What did the third iteration do?)** Even if Experiment 2 shows M2 and M3's boundary information is nearly identical, did the third iteration still change the representation structure? We characterize the M2→M3 transformation through representation structure analysis.

---

## Experiment 1: Why is the Shared Anchor Redundant?

**Goal**: Explain the pre-validation finding — M1 is nearly unimportant in the Pointer.

### 1a. Pointer Weight Quadrant Analysis

**Method**: Pointer's w1 has shape [2d] (d=96), where the first d dimensions correspond to M1 features and the last d to M2 features (concatenation order: [M1; M2]). Similarly, w2's first d correspond to M1, last d to M3.

| Weight | First d dims (M1 quadrant) | Last d dims (M2/M3 quadrant) | Ratio |
|---|---|---|---|
| w1 | 0.0902 | 0.1773 | 1.97× |
| w2 | 0.0728 | 0.1232 | 1.69× |

The M2/M3 quadrant norms in w1 and w2 are **1.97×** and **1.69×** their M1 quadrant counterparts respectively. A natural interpretation is that the model learned to assign lower weight to M1 during training. **However**: in linear models, weight norms are influenced by input feature magnitudes — if M1 tokens have systematically larger average magnitude than M2/M3, smaller weights may simply reflect scale compensation rather than reduced dependence. The current experiment does not report absolute feature magnitudes for M1/M2/M3, so the quadrant ratio should be interpreted as **directional support** (consistent with behavioral evidence), not independent strong evidence.

### 1b. CKA Information Overlap

**Method**: Compute linear CKA for M1-M2, M2-M3, M1-M3 (full dev set).

| Pair | CKA | Meaning |
|---|---|---|
| M1\_M2 | 0.871 | M2 contains most of M1's linearly readable information |
| M2\_M3 | 0.954 | M2 and M3 share nearly identical representational structure |
| M1\_M3 | 0.717 | M3 is most distant from M1 — iterations progressively transform representations |

CKA(M1, M2) = 0.871 indicates some linear structure difference between M1 and M2 (not the >0.9 near-identity), meaning M2 does not perfectly preserve M1's linear format. However, no\_M1 causes only −0.58 F1 — indicating that the information in M1's "unique" linear structure (the ~13% CKA gap) is not important for the Pointer's boundary predictions. All useful information the Pointer needs already exists in M2 in a linearly readable form.

CKA(M2, M3) = 0.954 shows the third iteration barely changed the linear structure of representations — M2 and M3 essentially carry the same information.

### Experiment 1 Summary

Weight analysis and CKA explain M1 redundancy from two independent angles:

- **Parameter side**: Pointer weights' M2/M3 quadrant norms are 1.7-2.0× the M1 quadrant — directionally consistent with low M1 importance, but weight norms may be influenced by input feature scale and are not treated as independent strong evidence.
- **Information side**: CKA(M1, M2) = 0.871 shows M2 and M1 share substantial linear structure. Combined with no\_M1's −0.58 F1 behavioral evidence, M1 provides negligible incremental value in the [M1;M2] concatenation.

The shared anchor design exists but provides negligible incremental value — this is the **first unsupported assumption** of the asymmetric design.

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
| M1 | 0.977 | 0.979 |
| M2 | **0.979** | **0.981** |
| M3 | 0.978 | 0.980 |

**Core finding: no functional specialization detected.** M2 achieves the highest AUC for **both** start and end, while M3 is slightly lower for both. The expected "M2 excels at start, M3 excels at end" specialization pattern **does not appear**.

All differences are within ΔAUC = 0.001. For reference, the linear probes are evaluated on ~14,000 test tokens (of which only ~100 are positive), yielding a typical AUC standard error of approximately 0.005-0.01 — a difference of 0.001 falls well within one standard error and is in the range of statistical noise. The three iteration products' boundary information content is statistically indistinguishable. The progressive pattern is M1→M2 improvement (+0.002), M2→M3 slight regression (−0.001) — the second iteration marginally improves linearly readable boundary information, while the third iteration slightly retreats.

**Cross-validation with wiring replacement**: Probes (information level) and wiring replacement (behavioral level) consistently point to M2 ≥ M3. The convergence of two methods strengthens the conclusion's robustness: within the detection scope of our experiments, **no meaningful functional specialization is observed** between M2 and M3, and the asymmetric wiring lacks information-level support — this is the **second unsupported assumption** of the asymmetric design. Note: probes test individual M layers, not the concatenated [M1;M2] the Pointer uses; and linear probes may miss non-linearly encoded specialization.

### Limitations

- is\_start and is\_end labels are extremely imbalanced (~0.70% positive)
- Linear probes operate on M2 or M3 individually, but the Pointer processes [M1;M2] concatenation — joint information may exceed the sum of parts
- Probes detect information **presence** in the representation, not how much the Pointer actually **uses**

---

## Experiment 3: What Did the Third Iteration Do?

**Goal**: Experiment 2 shows M2 and M3's boundary information is nearly identical (ΔAUC = 0.001), yet CKA(M2, M3) = 0.954, not 1.0 — the third iteration did change something. What did it change?

Each encoder block's pipeline is Conv\_0 → Conv\_1 → Self-Attention → FFN. Self-Attention is the only global operation. If the third iteration's Self-Attention aggregates token representations toward the global mean ("over-smoothing"), adjacent tokens become more similar, making precise boundary positions harder to distinguish.

**Method**: Compute the following representation structure metrics on M1/M2/M3 (200 samples):

1. **Local contrast**: 1 − cosine\_sim(token\_t, token\_{t+1}), averaged. High = adjacent tokens more distinguishable.
2. **Answer boundary sharpness**: Cosine distance between answer boundary positions (y1, y2) and adjacent positions. High = sharper boundaries.
3. **Token norm coefficient of variation (CoV)**: Std/mean of per-token L2 norms. High = more position-selective activation.

| Metric | M1 | M2 | M3 | M2→M3 change |
|---|---|---|---|---|
| Local contrast | 0.1162 | 0.0669 | 0.0480 | −28.2% |
| Answer boundary sharpness | 0.2213 | 0.1354 | 0.1017 | −24.9% |
| Norm CoV | 0.1627 | 0.1361 | 0.1273 | −6.5% |

**All three metrics consistently indicate M3 is smoother**: local contrast drops 28%, answer boundary sharpness drops 25%, norm diversity drops 6.5%. The progressive pattern M1→M2→M3 shows continuous smoothing — each iteration reduces local discriminability.

**Why does M1→M2 smooth more yet improve information?** The full progressive pattern is worth examining:

| Iteration | Local contrast change | Probe AUC change (start) |
|---|---|---|
| M1→M2 | 0.116 → 0.067 (**−42%**) | 0.977 → 0.979 (**+0.002**) |
| M2→M3 | 0.067 → 0.048 (−28%) | 0.979 → 0.978 (−0.001) |

The second iteration smooths **more** than the third (42% vs 28%), yet simultaneously improves boundary information. This is not contradictory: M1 is the CQ Attention output after only one encoding pass, retaining substantial local noise. The second iteration's smoothing is **beneficial information compression** — filtering noise while integrating cross-position semantic relationships, making boundary information more linearly readable. By M2, the representation has approached an **approximate fixed point** of the shared transformation function (CKA(M2, M3) = 0.954), and useful information compression is complete. The third iteration continues smoothing but has **no new information to integrate** — it is "smoothing without return."

**Alternative hypothesis: stochastic depth causing insufficient training.** The "smoothing without return" explanation above assumes the third iteration's transformation is functionally redundant. A competing explanation exists: during training, stochastic depth skips sub-layers with probability dropout × l/L (deeper layers have higher skip probability). Layers 15-21 (traversed only by M3) have the highest skip probability, which may cause M3's representations to be **insufficiently trained** rather than "over-smoothed" — the third iteration's parameters may not have received enough gradient updates to learn useful transformations. If this explanation holds, the M2→M3 structural changes reflect a training regularization side-effect, not an inherent deficiency of weight-shared iteration. Current experiments cannot distinguish the two hypotheses, as both predict M3 representation quality below M2. Distinguishing them requires retraining with stochastic depth disabled — beyond current experimental scope. We retain both explanations in parallel.

**Key tension: structure changes substantially, but information barely changes.** The third iteration's representation structure changes are significant (20-28%), yet linear probe AUC changes by only 0.001. This means: the third iteration substantially smooths the geometric structure of representations — adjacent tokens become more similar, boundaries become blurrier, norms become more uniform — but boundary position information is **preserved** in a more distributed encoding, remaining extractable by linear classifiers. The information persists, but the "format" changes.

This explains the mere 0.31 F1 difference between sym\_M2 and sym\_M3: M3's representations are structurally smoother than M2's, but the Pointer's linear projection can still extract sufficient boundary signal — just slightly less efficiently.

### Connection to H1 Findings

H1 found that Conv\_1 is the most important sub-layer in the Model Encoder — it produces high-discriminability local features for Self-Attention to use. The third iteration's Self-Attention global aggregation indeed smooths the local contrast that Conv\_1 established (−28%), echoing H1: Conv\_1 builds local features → Self-Attention utilizes and progressively consumes them → too many iterations cause structural degradation, even though information is not lost.

### Limitations

- Representation structure analysis is descriptive (correlational), not causal. "M3 is smoother" does not equal "M3 is worse because it is smoother" — confounding variables may exist.
- Conv and FFN in encoder blocks also affect representations; attributing changes to Self-Attention's "over-aggregation" requires finer-grained per-sub-layer analysis.
- M1, M2, M3 use the same encoder parameters; each iteration applies the same function to different input distributions. "Over-smoothing" is not a problem of Self-Attention itself, but a cumulative effect of iteratively applying the same transformation.

---

## Synthesis

### Answer

**Research question**: Does the functional specialization hypothesis implied by the Pointer's asymmetric wiring — M1 as shared anchor, M2/M3 as depth-specialized — hold in the trained model?

**Answer: It is not supported in the trained model.** All three assumptions of the asymmetric design fail to find support in our experiments:

**1. Shared anchor — near-redundant.** Replacing M1 causes only −0.58 F1 (behavioral). Pointer weights' M2/M3 quadrant norms are 1.7-2.0× M1 quadrants (parameter level; w1: 1.97×, w2: 1.69×). CKA(M1, M2) = 0.871 shows M2 and M1 share substantial linear structure (information level). All three angles point in the same direction: **M1 provides negligible incremental value in the [M1;M2] concatenation**. Note that no\_M1 substitutes M2/M3 into M1's position (not pure zeroing); the −0.58 loss partly reflects representational similarity.

**2. Depth specialization — not detected.** Linear probes show M2 achieves the highest AUC for both start (0.979) and end (0.981), with M3 slightly lower for both (start=0.978, end=0.980). The expected "M2→start, M3→end" specialization pattern **does not appear** — all differences are within ΔAUC=0.001, well below the estimated standard error (~0.005-0.01), in the range of statistical noise. Wiring replacement (behavioral) and probes (informational) consistently point to M2 ≥ M3. Note: probes test individual M layers, not the concatenated [M1;M2] the Pointer uses; and linear probes may not capture non-linearly encoded specialization.

**3. Iterative gain — second iteration captures the vast majority of value; third iteration's marginal contribution is near zero.** The only\_M1 vs sym\_M2 gap of 6.05 F1 shows the second iteration is essential; but the third iteration's marginal F1 contribution is negligible. Experiment 3 observes a noteworthy phenomenon: the third iteration further smooths representation structure (local contrast −28%, boundary sharpness −25%), yet linearly readable boundary information barely changes (ΔAUC=0.001). Information persists in a more distributed encoding — **structural change does not equal information loss**. However, M3's structural changes may partly stem from stochastic depth causing insufficient training (see limitations), not solely from inherent effects of iteration.

The asymmetric wiring (M2→start, M3→end) outperforms symmetric wiring (sym\_M2) by only 0.23 F1. This marginal advantage finds no support in the boundary information measured by linear probes (ΔAUC=0.001), and more likely reflects Pointer weight adaptation to training distributions.

### Evidence Chain

1. **Behavioral level** (pre-validation): Asymmetric advantage only 0.23 F1; M1 redundant (−0.58); M2 > M3 universality (sym\_M2 > sym\_M3); swap's −0.82 explained by dual distribution mismatch (0.23 + 0.54 ≈ 0.77 vs actual 0.82).
2. **Parameter level** (Experiment 1a): w1/w2 M2/M3 quadrant norms are 1.7-2.0× M1 quadrants, directionally consistent with low M1 importance (weight norms may be influenced by input feature scale; directional, not independent evidence).
3. **Information level** (Experiment 1b + Experiment 2): CKA shows M1 and M2 share substantial linear structure (0.871), M2 and M3 are highly similar (0.954). Linear probes detect no M2/M3 functional specialization (ΔAUC=0.001, within statistical noise).
4. **Mechanism level** (Experiment 3): Third iteration substantially smooths representation structure (20-28%) without destroying linearly readable boundary information — information preserved but format changed.

### Connection to Other Hypotheses

H1 demonstrated that Conv\_1 is an indispensable component in the Model Encoder (removal causes −32.54 F1); H3 demonstrates that M1 is redundant in the Pointer (removal causes only −0.58 F1). Together they show that QANet's design contains **coexisting critical bottlenecks and redundancies**. H2 demonstrated that C2Q is irreplaceable while Q2C is redundant in CQ Attention — the same theme of "irreplaceability vs redundancy." All three hypotheses reveal the uneven distribution of functional importance within QANet from different levels.

The direct echo between Experiment 3 and H1 is particularly notable: H1 found that Conv\_1 building local features is the core function of the Model Encoder; H3 found that iteratively applied Self-Attention progressively consumes these local features (M1→M2→M3 local contrast drops from 0.116 to 0.048). This forms a complete picture: Conv\_1 produces, Self-Attention consumes, and too many iterations lead to over-consumption.

### Limitations

1. **Distribution mismatch in eval-time wiring replacement**: Pointer weights are trained on specific input distributions; wiring replacement changes the input distribution. Both sym\_M2 and sym\_M3 have 50% input distribution mismatch. How much of their difference (0.31 F1) comes from intrinsic M2/M3 difference vs asymmetric distribution mismatch effects cannot be fully separated. Experiment 2's linear probes partially mitigate this.

2. **Causal attribution limitations**: Experiment 3's representation analysis is descriptive. The co-occurrence of "M3 is smoother" and "boundary information format changes" is observed, not causally established. Full causal verification would require selectively disabling Self-Attention in the third pass and measuring structural changes — high implementation complexity.

3. **Stochastic depth training-time effects**: As discussed in Experiment 3's main text, M3's structural changes may partly stem from stochastic depth causing insufficient training rather than iterative over-smoothing. Current experiments cannot distinguish the two. This is the primary alternative explanation for Conclusion 3 (iterative gain exhaustion).
