# H2: Directional Asymmetry in CQ Attention — Cross-Architecture Validation from BiDAF to QANet

## Motivation

QANet inherits its bidirectional CQ Attention design from BiDAF (Seo et al., 2017). This layer fuses the independently encoded Context and Question streams into a single representation. The output is a concatenation of four sub-components:

```
Output = [C;  A;  C⊙A;  C⊙B]     — total 4d dimensions
          ↑    ↑    ↑      ↑
        raw   C→Q  gated  Q→C→C
```

where A = S1·Q (C→Q direction), C⊙B derives from B = S1·S2ᵀ·C (Q→C direction), S1 is softmax over the Q dimension, and S2 is softmax over the C dimension. Both directions share the same similarity matrix S and trainable weight w; they differ only in the softmax normalization direction.

BiDAF (Seo et al., 2017, Table 1b) measured the relative importance of the two directions via training-time ablation (removing a component and retraining the model):

| Condition | F1 | ΔF1 |
|---|---|---|
| BiDAF (single) | 77.3 | — |
| Remove C2Q (A) | 67.7 | −9.6 |
| Remove Q2C (B) | 73.7 | −3.6 |

The C2Q direction is approximately **2.7× more important** than Q2C. However, BiDAF uses LSTMs as its encoder, while QANet replaces LSTMs entirely with convolutions and self-attention. **Does the asymmetry still hold after the architecture change?**

This question is non-trivial: QANet's Model Encoder (7 blocks × 3 passes = 21 layers) has far greater post-fusion processing capacity than BiDAF's two-layer BiLSTM, and could potentially compensate from the weaker direction's coarser signal. If QANet's stronger encoder "flattens" the directional gap, the asymmetry is an artifact of LSTM capacity bottleneck; if it persists or intensifies, it is an inherent property of the attention fusion mechanism.

Furthermore, BiDAF's ablation granularity was direction-level (C2Q vs Q2C), without separating the independent contributions of [C, A, C⊙A, C⊙B]. We have the opportunity to revisit this question at finer granularity.

## Research Question

Does BiDAF's C2Q > Q2C directional asymmetry (2.7×) hold in QANet's Conv+Attention architecture? If so, what mechanism maintains it across architectures?

## Methodological Note: Eval-Time Intervention and Mechanistic Analysis

BiDAF's ablation experiments (Seo et al., 2017, Table 1b) used **training-time ablation** — removing a component and retraining the entire model. This answers an architectural design question: *Is this component architecturally replaceable?*

We choose **eval-time ablation** — zeroing sub-component outputs directly on a fully trained model. This is a deliberate methodological choice, not a fallback due to resource constraints. The rationale:

Our research goal is **mechanistic analysis** — understanding the functional roles that components currently serve in the trained model — not architecture search. This aligns with the causal tracing methodology used by Meng et al. (2022) for localizing factual associations in GPT: intervening on a trained model (corruption + restoration) and observing behavioral changes to localize functional roles of specific computation steps. Training-time ablation allows the model to learn compensatory strategies during training, erasing the functional traces of the removed component — making it **unsuitable** for mechanistic analysis.

Eval-time ablation measures "how the model currently uses this component" — precisely the question we need to answer.

**How the comparison with BiDAF holds**: Due to methodological differences, the absolute ΔF1 values are **not directly comparable** (eval-time ablation tends to produce larger performance drops due to distribution shift). However, the methodological bias affects both C2Q and Q2C directions approximately equally — both are attention sub-components, and zeroing produces comparable distribution shift. Therefore, the **C2Q/Q2C direction ratio** is a robust cross-method comparison metric. Our comparison is strictly limited to the ratio level.

## Pre-Validation: CQ Attention is a Critical Bottleneck

Before dissecting CQ Attention internally, we first validate its importance as a whole. CQ Attention is the sole fusion point of the two streams — architecturally positioned as a bottleneck. We confirm this by zeroing its entire output (X = 0) at eval time and measuring F1/EM drop.

| Condition | F1 | EM | ΔF1 |
|---|---|---|---|
| Baseline | 70.24 | 59.22 | — |
| CQ output zeroed | 7.11 | — | −63.13 (−89.9%) |

F1 drops by 89.9%, far exceeding the 50% threshold. CQ Attention carries nearly all information the model's performance depends on. Dissecting its internal sub-components is well-motivated.

---

## Experiment 1: Does Directional Asymmetry Hold?

**Goal**: Measure the causal importance of four output sub-components, aggregate to direction-level, and cross-compare with BiDAF.

### 1a. Sub-Component Ablation

**Method**: CQ Attention output X has shape [B, 4d, L]. The four sub-components are arranged by dimension: X[:, :d] is C, X[:, d:2d] is A, X[:, 2d:3d] is C⊙A, X[:, 3d:4d] is C⊙B. Before X enters cq\_resizer, zero the target range while keeping the rest unchanged. Evaluate F1/EM on the full dev set.

| Condition | Zeroed content | Direction | F1 | ΔF1 |
|---|---|---|---|---|
| Baseline | None | — | 70.24 | — |
| −C | Raw context | No attention | 61.73 | −8.51 |
| −A | S1·Q | C→Q | 66.20 | −4.04 |
| −(C⊙A) | Gated | C→Q gated | 54.41 | −15.83 |
| −(C⊙B) | S1·S2ᵀ·C | Q→C→C | 69.21 | −1.03 |
| Only C | [C; 0; 0; 0] | No attention | 8.35 | −61.90 |

### 1b. Direction-Level Aggregation and BiDAF Comparison

Aggregate sub-component results to direction level: C2Q = simultaneously zero A and C⊙A; Q2C = zero C⊙B.

| Comparison | BiDAF (LSTM) | QANet (Conv+Attn) |
|---|---|---|
| −C2Q (remove A + C⊙A) | −9.6 F1 | −62.78 F1 |
| −Q2C (remove C⊙B) | −3.6 F1 | −1.03 F1 |
| C2Q / Q2C ratio | 2.7× | **61.0×** |

### 1c. cq\_resizer Weight Analysis

**Method**: cq\_resizer is Conv1d(4d, d, kernel\_size=1), with weight W of shape [d, 4d]. Partition W by columns into four [d, d] quadrants and compute each quadrant's Frobenius norm.

Ablation measures **performance loss** from removing a component (output side); weight norm measures the model's **learned dependency** on that component (parameter side). These are independent validation perspectives: if ablation ranking and weight norm ranking agree, conclusion credibility is significantly strengthened.

| Quadrant | Sub-component | Frobenius norm | % of total |
|---|---|---|---|
| W[:, :d] | C (raw) | 12.42 | 44.6% |
| W[:, d:2d] | A (C→Q) | **18.44** | **66.2%** |
| W[:, 2d:3d] | C⊙A (gated) | 11.90 | 42.8% |
| W[:, 3d:4d] | C⊙B (Q→C) | 11.78 | 42.3% |

### Experiment 1 Summary

**Finding 1 (Sub-component ranking)**: The causal importance ranking is **C⊙A (−15.83) ≫ C (−8.51) > A (−4.04) ≫ C⊙B (−1.03)**. The gated term C⊙A is the single most important sub-component; Q2C direction (C⊙B) is nearly negligible.

**Finding 2 (Cross-architecture comparison)**: QANet's C2Q/Q2C ratio is **61.0×** (BiDAF: 2.7×). The directional asymmetry not only **persists** after the architecture change but **dramatically intensifies** — from 2.7× to 61×. Removing C2Q drops F1 to 7.46 (approaching the 7.11 from full-layer removal), meaning C2Q carries nearly all of CQ Attention's value. Q2C (C⊙B) contributes only −1.03 F1, nearly redundant in QANet.

**Finding 3 (Weight validation)**: A (C→Q main term) has the highest Frobenius norm (18.44), significantly above the other three quadrants (11.78–12.42), **supporting** C2Q dominance from the parameter side. However, C⊙A and C⊙B have similar norms (11.90 vs 11.78), failing to predict their 15× ablation difference. Weight norms agree with ablation at the direction level (A largest → C2Q dominant), but at sub-component level, input signal quality equally determines causal effect.

**Finding 4 (C component role)**: Raw context passthrough (C) is the second most important sub-component (−8.51 F1), indicating the model relies on both attention-processed signals and unmodified raw context. C is direction-neutral (involves neither C2Q nor Q2C attention) and does not affect the directional asymmetry comparison.

**Intervention size control**: In the direction-level comparison, C2Q ablation zeros 2/4 dimensions (A + C⊙A) while Q2C zeros only 1/4 dimension (C⊙B) — different intervention scales. To rule out this confound, we check single-component comparisons (each zeroing the same 1/4 of dimensions): C⊙A (−15.83) vs C⊙B (−1.03) = **15.4×**, A (−4.04) vs C⊙B (−1.03) = **3.9×**. Even with strictly controlled intervention size, every C2Q sub-component significantly outweighs Q2C. The asymmetry holds at any granularity.

---

## Experiment 2: What Mechanism Maintains the Asymmetry?

**Premise**: Experiment 1 confirms the directional asymmetry holds (if it did not, this experiment's question becomes "what caused the asymmetry to disappear," but the experimental design remains the same).

**Question**: C2Q is more important. But C2Q and Q2C share the same similarity matrix S, differing only in softmax direction. Why does merely changing the normalization direction produce such a massive causal difference?

Starting from the mathematical definitions, the two directions perform fundamentally different operations:

- **C2Q (A = S1·Q)**: **Injects** Question embeddings into every Context position — introducing information that **did not previously exist** in Context.
- **Q2C (C⊙B, where B = S1·S2ᵀ·C)**: **Redistributes** Context information among Context positions, mediated by Question — introduces no new information, only reorganizes existing information.

If answer-relevant information in Context is spatially concentrated (the answer span occupies only a small fraction of all tokens), then "redistributing existing Context information" has low value — useful information is already at the correct positions. Meanwhile, "injecting Question information" is irreplaceable — the Question signal has no other pathway into CQ Attention's output.

We test the information concentration premise through selective corruption.

### Method

Apply noise (σ = 3 × std) only to specific Context positions, keeping others clean. 200 samples, 3 noise repeats per sample.

Conditions:
- **ans\_only**: Corrupt only answer-span positions (y1 to y2)
- **non\_ans\_only**: Corrupt only non-answer, non-padding positions
- **full\_context**: Corrupt all Context positions (baseline reference)

| Condition | Mean TE | ±95% CI | Avg #tokens | TE/token |
|---|---|---|---|---|
| ans\_only | 0.170 | ±0.034 | 3.1 | 55.7 × 10⁻³ |
| non\_ans\_only | 0.266 | ±0.041 | 140.5 | 1.89 × 10⁻³ |
| full\_context | 0.361 | ±0.045 | 143.6 | 2.52 × 10⁻³ |

**Information density ratio**: TE/token(answer) ÷ TE/token(non-answer) = **29.5×**

Supplementary observation: TE(ans) + TE(non) = 0.436 > TE(full) = 0.361, interaction = −0.075 (sub-additive), indicating partial overlap between answer-region and non-answer-region corruption effects.

### Reasoning

Density ratio = 29.5×, far exceeding the 5× threshold. Context's causal information is **highly concentrated** in the answer span (on average, 3.1 tokens account for 47% of total effect).

This finding validates the mechanism argument's premise:

1. **Q2C's redistribution function is nearly redundant**: Useful Context information is already concentrated at the correct positions; redistributing Context representations via Question mediation adds almost no new value. Furthermore, QANet's Model Encoder contains 21 layers of Self-Attention whose function is precisely to redistribute information among Context positions — highly overlapping with Q2C. Q2C's function is structurally subsumed by downstream processing.
2. **C2Q's injection function is irreplaceable**: Question information has no other pathway into the Context representation stream after CQ Attention. No matter how powerful the downstream encoder, it cannot generate Question signal from nothing — only C2Q can provide this.
3. **Information concentration amplifies the asymmetry**: The more concentrated the information, the lower the marginal value of "redistribution" (already at correct positions), while the value of "injecting new signal to correct positions" remains unchanged. The 29.5× concentration means Q2C's functional space is extremely compressed.

### Limitations

- Corrupting answer-span tokens directly removes the target signal; high TE is expected. The truly informative observation is the non-answer token effect and the specific density ratio value — 29.5× far exceeds baseline.
- Information concentration is measured at the Context input side. The "novel information injection vs existing information redistribution" functional distinction derives from CQ Attention's mathematical definition (A = S1·Q introduces Q, B = S1·S2ᵀ·C reorganizes C) — it is deductive, not directly measured experimentally.

---

## Synthesis

### Answer

**Research question**: Does BiDAF's C2Q > Q2C directional asymmetry hold in QANet? What mechanism maintains it?

**Answer**: The directional asymmetry is not only cross-architecture robust, but **dramatically intensifies** in QANet — from BiDAF's 2.7× to **61.0×**.

**Fact**: Under QANet's Conv+Attention architecture, the joint causal effect of C2Q sub-components (A, C⊙A) is 61.0 times that of Q2C (C⊙B). Removing C2Q drops F1 to 7.46 (approaching the 7.11 from full-layer removal); C2Q carries nearly all of CQ Attention's value. Q2C contributes only −1.03 F1, nearly redundant in QANet. Even under controlled single-component comparison (each zeroing 1/4 of dimensions), C⊙A vs C⊙B = 15.4×, A vs C⊙B = 3.9× — the asymmetry holds at every granularity. cq\_resizer weight norms independently validate C2Q dominance at the direction level.

**Mechanism**: The asymmetry's root cause is that the two directions perform **fundamentally different operations**:

- **C2Q (A = S1·Q)** injects Question embeddings into Context representations — the **sole source of new information**. Question signal has no other pathway into the Context representation stream after CQ Attention. No matter how powerful the downstream encoder, it cannot generate Question signal from nothing. C2Q is irreplaceable.
- **Q2C (C⊙B, where B = S1·S2ᵀ·C)** redistributes Context information among Context positions, mediated by Question — no new information is introduced, only existing information is reorganized. This function highly overlaps with QANet's Model Encoder's 21 layers of Self-Attention. Q2C is replaceable.

Experiment 2 validates this mechanism's precondition: Context's causal information is highly concentrated at the answer span (density ratio 29.5×, with 3.1 answer tokens accounting for 47% of total effect). Information is already at the correct positions — redistributing it has almost no value, but injecting Question signal to "mark" these positions is indispensable.

**Amplification effect** (2.7× → 61×): BiDAF's post-fusion processing consists of only 2 LSTM layers, insufficient to fully substitute Q2C's Context redistribution function, so Q2C retains −3.6 F1 of residual value in BiDAF. QANet has 21 layers including Self-Attention in its Model Encoder, whose Context redistribution capacity far exceeds Q2C's, compressing Q2C's residual value to −1.03 F1 (near zero). C2Q's irreplaceability is unchanged across architectures; Q2C's redundancy increases with encoder strength — hence the ratio amplifies from 2.7× to 61×.

### Evidence Chain

1. **Pre-validation**: CQ Attention full ablation causes F1 to drop **89.9%** (70.24 → 7.11), confirming its information bottleneck role; dissecting internal sub-components is well-motivated.
2. **Core fact** (Experiment 1): Sub-component ablation establishes ranking C⊙A ≫ C > A ≫ C⊙B; direction-level aggregation establishes C2Q/Q2C = **61.0×** (BiDAF: 2.7×); single-component comparison with controlled intervention size still holds (C⊙A/C⊙B = 15.4×); cq\_resizer weight A-quadrant norm highest (18.44), independently validating from the parameter side.
3. **Mechanism explanation** (Experiment 2 + architecture analysis): Answer token information density **29.5×** → Context information highly concentrated → Q2C's redistribution function redundant (information already at correct positions, and subsumed by Model Encoder's 21 Self-Attention layers) → C2Q's injection function irreplaceable (sole source of Question signal) → asymmetry is an inherent property of CQ Attention's operational structure, cross-architecture robust. Amplification stems from QANet's stronger downstream encoder further compressing Q2C's residual value.

### Limitations

1. **Sub-component interaction**: A and C⊙A have algebraic dependency (C⊙A = C × A); zeroing one individually may not reflect independent contribution. Direction-level ablation (simultaneously zeroing A + C⊙A) mitigates this and is the correct granularity for BiDAF comparison.

2. **Functional overlap not directly measured**: We argue that Q2C's Context redistribution function is subsumed by the Model Encoder's Self-Attention. This reasoning is based on their operational definitions (both are weighted aggregation within Context) and Q2C's near-zero causal effect (−1.03 F1) in QANet. However, we did not directly measure their functional overlap (e.g., whether Q2C becomes more important when Model Encoder Self-Attention is removed). Such a cross-ablation experiment could further validate the "functional substitution" hypothesis but has high implementation complexity.

3. **S1 and S2 share parameters**: Both directions' attention matrices derive from the same similarity matrix S under different normalization directions, sharing trainable weight w. They are not truly independent modules but two views of the same computation.
