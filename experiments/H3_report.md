# H3: Pointer Layer Asymmetric Wiring & Pass Role Differentiation

## Hypothesis

> QANet's asymmetric Pointer design (M1+M2 → start, M1+M3 → end) is causally justified: M2 and M3 encode distinct information, with M2 specializing in start-boundary features and M3 in end-boundary features. The asymmetric wiring exploits this specialization.

## Background

QANet's Model Encoder produces three intermediate representations:
- **M1**: Output after Pass 1 (7 blocks)
- **M2**: Output after Pass 2 (7 blocks)
- **M3**: Output after Pass 3 (7 blocks)

The Pointer layer computes:
```
P(start) = softmax(w1 · [M1; M2])
P(end)   = softmax(w2 · [M1; M3])
```

**Key question**: Why M2 for start and M3 for end? Is this an arbitrary design choice or does training induce genuine start/end specialization?

---

## Phase A: Pointer Wiring Replacement (Ablation)

**Method**: At eval time, swap the M1/M2/M3 inputs to the Pointer layer and measure F1/EM on the full dev set. This is a **causal intervention** — we change the wiring and observe the effect.

**Configurations tested**:

| Config | Start Input | End Input | Rationale |
|---|---|---|---|
| original | M1 + M2 | M1 + M3 | Paper design |
| swap | M1 + M3 | M1 + M2 | Test if M2/M3 are interchangeable |
| sym\_M2 | M1 + M2 | M1 + M2 | Use M2 for both |
| sym\_M3 | M1 + M3 | M1 + M3 | Use M3 for both |
| only\_M1 | M1 + M1 | M1 + M1 | Remove M2/M3 contribution |
| no\_M1 | M2 + M3 | M2 + M3 | Remove M1 contribution |

### Key Metrics to Report

| Config | F1 | EM | ΔF1 | ΔEM |
|---|---|---|---|---|
| original | [from h3_results.json] | | — | — |
| swap | | | | |
| sym\_M2 | | | | |
| sym\_M3 | | | | |
| only\_M1 | | | | |
| no\_M1 | | | | |

### Analysis Framework

1. **swap vs original**: If swap hurts → M2 and M3 are NOT interchangeable → they encode different information
2. **sym\_M2 vs sym\_M3**: Which single pass is more versatile? If sym\_M2 < sym\_M3 → M3 is a better universal representation
3. **only\_M1**: How much F1 is lost without M2/M3 → quantifies their combined contribution
4. **no\_M1**: How much F1 is lost without M1 → M1 may provide shared context that both start/end need
5. **Asymmetric advantage**: Is original strictly better than all symmetric alternatives?

### Limitations

- Wiring replacement is eval-time only — the model was trained with original wiring, so all alternatives may underperform simply due to distribution mismatch (the Pointer weights were optimized for the original input distribution)
- This is a necessary but not sufficient test: poor swap performance shows M2≠M3, but doesn't prove M2=start and M3=end

---

## Phase B: Representation Similarity Analysis (CKA + Cosine)

**Method**: Collect M1, M2, M3 token representations and compute pairwise similarity metrics.

**Metrics**:
- **Global Cosine Similarity**: Per-token cosine similarity averaged across tokens and samples
- **Position-Stratified Cosine (M2 vs M3)**: Compare similarity at answer-start, answer-interior, answer-end, and non-answer positions
- **Linear CKA**: Dataset-level representational similarity (robust to linear transforms)

**Samples**: 1000

### Key Metrics to Report

**Global Cosine Similarity**:

| Pair | Mean | ±95%CI |
|---|---|---|
| M1\_M2 | | |
| M1\_M3 | | |
| M2\_M3 | | |

**CKA**:

| Pair | CKA |
|---|---|
| M1\_M2 | |
| M1\_M3 | |
| M2\_M3 | |

### Analysis Framework

1. **Progressive transformation**: M2\_M3 CKA > M1\_M3 CKA → each pass progressively transforms representations
2. **M2/M3 divergence**: If CKA(M2, M3) < 1 → the passes don't converge to the same representation
3. **Position-stratified cosine**: If M2-M3 similarity is LOWER at answer boundaries → M2/M3 diverge most where it matters for start/end prediction

### Limitations

- CKA and cosine similarity are **correlation-based**, not causal
- High similarity ≠ same information; low similarity ≠ complementary information
- Cannot directly prove specialization direction (which encodes start vs end)

---

## Experiment C: Linear Probe — Start/End Token Specialization

**Method**: Train linear classifiers on M1/M2/M3 per-token representations to predict:
- `is_start_token`: 1 if token position = answer start, 0 otherwise
- `is_end_token`: 1 if token position = answer end, 0 otherwise

This provides **direct evidence** for the specialization claim: if M2 is better at predicting start tokens and M3 is better at predicting end tokens, the asymmetric wiring is causally justified.

**Samples**: 500 (80/20 train/test split)
**Classifier**: LogisticRegression (balanced class weight, C=1.0)
**Metric**: ROC-AUC (handles extreme class imbalance)

### Key Metrics to Report

| Representation | Target | AUC | Acc | F1(+) |
|---|---|---|---|---|
| M1 | start | | | |
| M1 | end | | | |
| M2 | start | | | |
| M2 | end | | | |
| M3 | start | | | |
| M3 | end | | | |

**Specialization scores**:
- M2 start advantage = AUC(M2→start) − AUC(M3→start)
- M3 end advantage = AUC(M3→end) − AUC(M2→end)

### Analysis Framework

1. **If M2 start advantage > 0 AND M3 end advantage > 0**: Specialization confirmed — M2 encodes start features, M3 encodes end features. Asymmetric wiring directly exploits this.
2. **If both advantages are near zero**: Specialization not linearly detectable. The Pointer's bilinear projection may exploit non-linear differences that linear probes cannot capture.
3. **If one advantage is positive but the other is negative**: Partial specialization — one pass specializes but the other does not.
4. **Progressive encoding**: Compare M1→M2→M3 AUC trajectories for start and end separately. If M2 peaks at start and M3 peaks at end → progressive specialization through encoder passes.

### Limitations

- Linear probes detect only linearly decodable information; non-linear specialization may exist but be undetectable
- Extreme class imbalance (~1 positive per sample out of ~400 tokens ≈ 0.25%)
- The Pointer uses bilinear projection (w1·[M1;M2]), not linear readout — the probe tests a simpler access mechanism than the model actually uses
- Start/end token labels are binary (single position), but the model may encode answer boundaries as soft gradients across neighboring positions

---

## Synthesis

### Evidence Chain

1. **Phase A (Wiring Ablation)**: Swap hurts → M2 ≠ M3 (necessary condition for specialization)
2. **Phase B (CKA/Cosine)**: M2 and M3 are similar but not identical; divergence is greater at answer boundaries → structural difference exists where it matters
3. **Experiment C (Linear Probe)**: M2 start AUC vs M3 start AUC, and M3 end AUC vs M2 end AUC → tests the directional specialization claim directly

### Cross-Method Consistency

| Claim | Phase A | Phase B | Experiment C |
|---|---|---|---|
| M2 ≠ M3 | swap ΔF1 < 0 | CKA < 1 | Different AUC profiles |
| M2 → start | swap hurts start | Lower M2-M3 sim at start | AUC(M2→start) > AUC(M3→start) |
| M3 → end | swap hurts end | Lower M2-M3 sim at end | AUC(M3→end) > AUC(M2→end) |
| Asymmetry justified | original > all symmetric | — | Specialization scores > 0 |

### Methodological Notes

- The three methods address the hypothesis at different levels: behavioral (F1/EM), representational (CKA/cosine), and informational (linear probe)
- Phase A is causal (intervention), Phase B is correlational, Experiment C is a hybrid (probing for causal readability)
- The eval-time wiring swap has a distribution mismatch caveat — the Pointer weights were trained for original wiring
