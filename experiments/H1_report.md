# H1: Causal Importance of Convolution vs Self-Attention in QANet's Encoder

## Motivation

Each encoder block in QANet's Model Encoder contains four sub-layers — Conv\_0, Conv\_1, Self-Attention, and FFN. Each block follows this pipeline:

```
Position Encoding → Conv_0 → Conv_1 → Self-Attention → FFN
                    (each with residual connection)
```

The QANet paper's ablation study (Table 5, Yu et al. 2018) shows that convolution contributes more to performance than Self-Attention (−2.7 F1 vs −1.3 F1 upon removal, measured after retraining). However, the paper treats both convolution layers as a single unit and does not examine Conv\_0 and Conv\_1 separately.

By separating the two through eval-time ablation and causal tracing, both methods agree that Conv\_1 is the most important component by a large margin. Ablation (measuring necessity) yields:

> **Conv\_1 ≫ Self-Attention > FFN > Conv\_0**

Conv\_1 is the **most important** of all four component types (−32.54 F1 upon removal), while Conv\_0 is the **least important** (−1.09 F1) — a 30× gap. Yet Conv\_0 and Conv\_1 are **architecturally identical**: both are depthwise separable convolutions with kernel size k=5 and the same hidden dimension, trained end-to-end under the same objective.

The same architecture produces both the most important and the least important component. Architecture alone cannot explain this gap; the divergence must relate to their different positions in the pipeline (Conv\_0 receives the raw input, Conv\_1 receives Conv\_0's output) and the different functions they consequently learn during training.

## Research Question

Conv\_0 and Conv\_1 share identical architecture yet differ 30× in importance. What mechanistic differences between them explain this gap?

## Experimental Design

We address this question through three experiments, each building on the previous:

**Experiment 1 (Is the gap real?)** We quantify component importance using two independent methods — eval-time ablation and causal tracing — and check for output-magnitude confounds. The goal is to establish that the 30× gap between Conv\_1 and Conv\_0 is reproducible and not an artifact.

**Experiment 2 (What is the functional difference?)** Experiment 1 confirms that Conv\_1 is 30× more important than Conv\_0. So how do their outputs differ functionally? The most direct explanation would be that Conv\_1 encodes critical information that Conv\_0 does not. We first test this with linear probes and find that both encode nearly identical task-relevant information (AUC gap only 0.025) — information difference cannot explain a 30× importance gap. We therefore examine whether the two layers apply different transformations to their inputs (representational structure).

**Experiment 3 (How does the model fail?)** Experiment 2 characterizes the output differences between Conv\_1 and Conv\_0. What happens inside the model when Conv\_1 is removed? Since Self-Attention is Conv\_1's direct downstream consumer, we capture attention weights and observe how attention patterns change under Conv\_1 removal, using Conv\_0 removal as a control, to reveal the proximate mechanism of the performance collapse.

---

## Experiment 1: Establishing the Gap

**Goal**: Quantify the importance gap between Conv\_0 and Conv\_1, and verify its consistency across methods.

We use two complementary approaches: (a) ablation — remove a component and measure performance degradation, reflecting how necessary it is for the final output; (b) causal tracing — restore a component's clean activation under noisy input and measure performance recovery, reflecting its independent causal contribution. The two methods have different logic (removal vs restoration); agreement between them strengthens confidence in the result.

Additionally, we perform per-block ablation to examine how importance is distributed across the 7 blocks and 3 passes.

**Note on ablation methodology**: All ablations are performed at eval-time (zeroing out component outputs without retraining). Ideally, retraining after ablation would control for the model's compensatory adaptation, but each retraining run requires a full GPU training cycle, which is infeasible under Colab quota constraints. Since the core objective of this experiment is to determine the **relative importance ordering** among components (rather than precise absolute performance differences), eval-time ablation is sufficient for qualitative conclusions. The QANet paper's retrain-based ablation (Yu et al., 2018) also ranks convolution above Self-Attention, consistent with our ordering.

### 1a. Global Ablation

**Method**: Zero out all 21 instances of a component type (eval-time, full dev set, 10,465 samples). Zero-out = sub-layer output replaced with zeros before residual addition.

| Config | F1 | ΔF1 | EM |
|---|---|---|---|
| baseline | 70.24 | — | 59.22 |
| −Conv\_1 | 37.70 | **−32.54** | 23.59 |
| −Conv\_0 | 69.16 | **−1.09** | 57.85 |
| −Attn | 65.44 | −4.81 | 55.18 |
| −FFN | 66.92 | −3.32 | 54.56 |
| −Conv (both) | 27.14 | −43.11 | 13.60 |

**Finding 1**: Conv\_1 is the most important component; Conv\_0 is the least. The gap between architecturally identical layers is **30×** (32.54 / 1.09).

**Finding 2**: Joint removal (−43.11) exceeds the sum of individuals (−33.63) by 9.48 F1 — a super-additive interaction. Conv\_0 contributes non-redundant preprocessing that only becomes critical when Conv\_1 is also absent.

### 1b. Per-Block Ablation

**Method**: Remove conv (both) or self\_attn in one (pass, block). 42 configs, full dev set (10,465 samples).

**Finding 3**: Max single-block impact = −1.23 F1 (p0\_b0\_conv); collective removal = −43.11. A 35× gap reveals **distributed redundancy**: no single block is essential, but together they are indispensable.

**Finding 4**: Early passes matter more: M1 (−0.16 avg) >> M2 (−0.06) >> M3 (+0.01).

### 1c. Causal Tracing (Meng et al., 2022)

**Method**: Per sample: (1) clean run → collect 84 sub-layer activations; (2) corrupt context embeddings (Gaussian, σ = 3× std); (3) restore ONE sub-layer's clean activation → measure recovery (AIE). 200 samples, 3 noise repeats.

| Component | Mean AIE | Sum AIE | Share |
|---|---|---|---|
| Conv\_1 | 0.0147 | 0.3089 | 71.3% |
| Conv\_0 | 0.0032 | 0.0664 | 15.3% |
| Self-Attn | 0.0016 | 0.0330 | 7.6% |
| FFN | 0.0012 | 0.0246 | 5.7% |

**Finding 5**: Causal tracing independently confirms the gap — Conv\_1 accounts for 71.3% of total AIE. Conv\_0 ranks second (15.3%), above Attn and FFN, suggesting it does carry some causal weight even though ablation damage is minimal. This asymmetry (small ablation damage, moderate causal recovery) is consistent with Conv\_0 providing preprocessing that is useful but compensable via residual connections.

### Confound Check: Output Magnitude

Sub-layer L2 norms: Conv\_1 (250) > Conv\_0 (113) > FFN (104) > Attn (82). Pearson r(norm, AIE) = 0.893.

Could the gap simply reflect Conv\_1 having larger outputs? For Conv\_0/FFN/Attn, norm ordering (Conv\_0 > FFN > Attn) is **inversely** correlated with ablation ordering (Attn > FFN > Conv\_0) — ruling out magnitude as the driver for these three. Conv\_1's dominance direction is robust; the precise ratio over Attn (2–7×) carries uncertainty from norm co-correlation.

### Experiment 1 Summary

The 30× importance gap between identical architectures is real and reproducible across two independent methods. It cannot be explained by output magnitude alone. Something about what Conv\_1 *learns to do* is fundamentally different from Conv\_0.

---

## Experiment 2: Functional Differences Between Conv\_1 and Conv\_0

**Question**: Experiment 1 confirms that Conv\_1 is 30× more important than Conv\_0. How do their outputs differ functionally?

The most direct explanation would be that Conv\_1 encodes critical information that Conv\_0 does not — if so, the importance gap can be directly attributed to an information difference. We first test this with linear probes (2a). If information content cannot explain the gap, we then examine a deeper possibility: whether the two layers apply different transformations to their inputs, i.e., whether they differ in representational structure (2b, 2c).

### 2a. Information Content (Linear Probe)

**Method**: Linear probing is a standard technique in representation analysis (Alain & Bengio, 2017; Belinkov et al., 2017): a simple linear classifier is trained on frozen intermediate representations; if it can decode a property, that property is linearly readable in the representation. We train logistic regression on per-token sub-layer outputs (before residual addition), with label: inside answer span. 500 samples, 80/20 split, 5 (pass, block) positions.

| Component | Mean AUC |
|---|---|
| Conv\_1 | 0.901 |
| FFN | 0.898 |
| Self-Attn | 0.894 |
| Conv\_0 | 0.876 |

**Finding 6**: All components encode answer-position information to a similar degree. The Conv\_1 vs Conv\_0 gap is only 0.025 AUC, negligible compared to their 30× ablation gap. This is consistent with residual architecture: each layer's output = sub-layer transformation + skip-connection input, so all outputs contain the accumulated information in the residual stream. The difference is not about *what information is encoded*.

**Limitation**: We only probed one property (answer position). Differences in other task-relevant properties (e.g., POS tags, syntactic roles) were not tested. Additionally, linear probes can only capture linearly separable features; non-linearly encoded differences may be missed.

### 2b. Representational Structure (Discriminativity)

**Method**: Compare Conv\_1 vs Conv\_0 output properties directly. 200 samples, 5 (pass, block) positions. Metrics: local contrast (1 − lag-1 cosine similarity), token norm variance (CoV), effective rank (exp of singular value entropy).

| Metric | Conv\_0 | Conv\_1 | Conv\_1/Conv\_0 | Attn | FFN |
|---|---|---|---|---|---|
| Token Norm Variance | 4.57 | 14.48 | **3.2×** | 1.62 | 14.79 |
| Norm CoV | 0.345 | 0.216 | 0.6× | 0.176 | 0.410 |
| Local Contrast (1−cos) | 0.323 | 0.305 | 0.9× | 0.103 | 0.341 |
| Effective Rank | 32.5 | 43.2 | **1.3×** | 8.6 | 26.5 |

**Finding 7**: The results show a differentiated pattern. Conv\_1 is significantly higher than Conv\_0 on two dimensions: token-level amplitude variation (Norm Variance 3.2×) and output subspace dimensionality (Effective Rank 1.3×). This means Conv\_1 produces outputs with much greater magnitude differences across positions and utilizes more feature dimensions. However, in directional distinctiveness (Local Contrast) and normalized variation (Norm CoV), Conv\_0 is actually slightly higher. This indicates Conv\_1's transformation is primarily characterized by **amplitude modulation and dimensional utilization**, rather than directional separation of adjacent tokens. Conv\_0's outputs, while more directionally diverse, have minimal overall impact (ablation only −1.09 F1), suggesting this directional diversity does not critically contribute to downstream computation.

### 2c. Spatial Structure (Local Coherence)

**Method**: Mean cosine similarity between output vectors at positions t and t+k, at p0\_b0, 500 samples. Directly measures the spatial correlation structure that each component imposes.

| Component | lag=1 | lag=2 | lag=3 | lag=5 | decay(1→5) |
|---|---|---|---|---|---|
| Conv\_0 | 0.679 | 0.602 | 0.597 | 0.571 | 0.109 |
| Conv\_1 | 0.667 | 0.585 | 0.610 | 0.608 | 0.058 |
| Self-Attn | 0.841 | 0.819 | 0.810 | 0.804 | 0.037 |
| FFN | 0.595 | 0.564 | 0.562 | 0.556 | 0.039 |

**Finding 8**: Conv\_0's decay (0.109) is nearly double Conv\_1's (0.058), indicating Conv\_0's output has stronger locality — high similarity among nearby tokens that drops off with distance. Conv\_1's flatter decay means its output is more spatially uniform. Self-Attention has the highest overall similarity (0.84) with near-zero decay (0.037), consistent with its global mixing role. FFN has the lowest similarity (0.60), reflecting position-wise independent transformation.

### Experiment 2 Summary

The functional difference between Conv\_1 and Conv\_0 is not in information content (both outputs contain similar task-relevant information, AUC gap only 0.025) but in **specific dimensions of representational structure**. Conv\_1's output has significantly higher token-level amplitude variation (Norm Variance 3.2×) and subspace dimensionality (Effective Rank 1.3×) compared to Conv\_0, but the two are similar in directional distinctiveness (Local Contrast). Combined with spatial coherence data, Conv\_1's output is more spatially uniform (decay 0.058) while Conv\_0 is more localized (decay 0.109). In summary, Conv\_1's transformation is characterized by: **assigning greater amplitude differences across token positions while organizing features in a higher-dimensional subspace**.

The divergence across metrics has diagnostic value. Conv\_0 scores slightly higher than Conv\_1 on directional distinctiveness (Local Contrast) and normalized variation (Norm CoV), yet Conv\_0 is nearly dispensable (−1.09 F1) — this rules out directional distinctiveness as a sufficient condition for the importance gap. The two dimensions where Conv\_1 holds a clear advantage — amplitude modulation and subspace dimensionality — are therefore candidate mechanisms for the gap. This inference is consistent with dot-product attention mechanics: \( QK^T \) scores are sensitive to input magnitude differences, and higher-dimensional inputs allow Q, K projections to carry richer discriminative signals. Notably, FFN has nearly identical Norm Variance (14.79) to Conv\_1 but is far less important (−3.32 vs −32.54 F1) — FFN sits downstream of Self-Attention, so its output properties do not directly influence attention computation. This further indicates that amplitude modulation's impact depends on pipeline position; the property alone is not a sufficient condition. The inference has not been verified through direct intervention (e.g., artificially injecting high-amplitude features and observing attention changes) and remains correlational.

This observation alone does not explain the **root cause** of the gap — Conv\_1's distinctive representational structure could arise from its pipeline position (receiving Conv\_0's output rather than raw input), training dynamics, or their interaction. Separating these factors would require controlled experiments (e.g., swapping learned weights post-training, or retraining with altered architectural order), which is prohibitively expensive under Colab quota limits. What this experiment does establish is a key fact: **Conv\_1 and Conv\_0 produce measurably different outputs, particularly in amplitude distribution and subspace dimensionality**. Experiment 3 examines the concrete impact of removing these features on downstream Self-Attention computation.

---

## Experiment 3: How Does the Model Fail Without Conv\_1?

**Question**: Experiment 1 shows that removing Conv\_1 causes a −32.54 F1 performance collapse. Experiment 2 shows that Conv\_1's output is structurally distinct from Conv\_0's. What happens inside the model when Conv\_1 is removed? What is the proximate mechanism of the performance collapse?

**Method**: Self-Attention is the direct downstream consumer of Conv\_1's output. We use forward hooks to capture attention weights and observe how attention patterns change under Conv\_1 removal. Three conditions (50 samples): clean, −Conv\_1, −Conv\_0. Conv\_0 serves as a control condition.

### JS Divergence (clean vs ablated)

| Block | JSD(−Conv\_1) | JSD(−Conv\_0) | Ratio |
|---|---|---|---|
| 0 | 0.0331 | 0.0041 | 8.1× |
| 1 | 0.0870 | 0.0110 | 7.9× |
| 2 | 0.0749 | 0.0108 | 7.0× |
| 3 | 0.1222 | 0.0118 | 10.3× |
| 4 | 0.0946 | 0.0118 | 8.0× |
| 5 | 0.0888 | 0.0110 | 8.1× |
| 6 | 0.0898 | 0.0112 | 8.0× |
| **Avg** | **0.0843** | **0.0102** | **8.2×** |

**Finding 9**: Attention distortion under Conv\_1 removal is **8.2× greater** than under Conv\_0 removal. The ratio is consistent across all 7 blocks (7.0–10.3×).

### Attention Entropy and Answer Focus

| Condition | Entropy | ΔEntropy | Answer Mass | ΔMass |
|---|---|---|---|---|
| Clean | 4.621 | — | 0.0463 | — |
| −Conv\_1 | 4.362 | −0.259 | 0.0373 | −19% |
| −Conv\_0 | 4.554 | −0.067 | 0.0489 | +6% |

**Finding 10**: Without Conv\_1, attention does not diffuse — it **collapses** into over-concentrated patterns on wrong positions (entropy *decreases*, answer mass drops 19%). This reveals the proximate mechanism of the performance collapse: the model does not lose track of information — instead, attention becomes erroneously focused on non-answer positions.

### On Pipeline Distance

An intuitive objection is: does Conv\_1 removal affect attention more simply because Conv\_1 is Self-Attention's immediate upstream neighbor (0 layers apart), while Conv\_0 is buffered by Conv\_1 in between?

Our existing data addresses this. Conv\_0 is also an immediate upstream neighbor — of Conv\_1 (0 layers apart). Yet removing Conv\_0 causes only −1.09 F1 (Experiment 1). If being a direct upstream neighbor were sufficient to produce large impact, removing Conv\_0 should substantially impair Conv\_1's computation — but it does not. Experiment 2 shows that Conv\_0 does apply a non-trivial transformation (Local Contrast = 0.323, Effective Rank = 32.5 — not an identity mapping), but its contribution is compensable: the residual connection ensures Conv\_1 still receives the core information flow, and Conv\_0's additional contribution is not indispensable to downstream computation.

Therefore, **pipeline proximity is necessary but not sufficient for large impact**. What matters is whether the downstream layer **irreplaceably depends** on the upstream layer's specific output. Conv\_1→Attention: attention computation critically relies on Conv\_1's high-amplitude, high-dimensional features (Experiment 2); removal causes pattern collapse. Conv\_0→Conv\_1: Conv\_1 does not critically depend on Conv\_0's specific output (−1.09 F1 upon Conv\_0 removal); its contribution is compensable via the residual pathway. Distance and functional dependence jointly determine impact.

**Residual limitation**: Globally removing Conv\_1 alters the entire residual stream, so attention degradation reflects not only the direct Conv\_1→Attention impact but also indirect effects propagated through the residual stream. Fully isolating the direct impact is beyond the capacity of eval-time experiments.

---

## Synthesis

### Answering the Research Question

**Research question**: Conv\_0 and Conv\_1 share identical architecture yet differ 30× in importance. What mechanistic differences between them explain this gap?

**Answer**: The directly observable cause lies in the different transformations Conv\_1 and Conv\_0 apply, and Self-Attention's irreplaceable dependence on Conv\_1's specific output. Both layers encode similar task-relevant information (AUC gap only 0.025), but Conv\_1 produces outputs with significantly higher token-level amplitude variation (Norm Variance 3.2×) and subspace dimensionality (Effective Rank 1.3×). Conv\_0's transformation, while non-trivial, is compensable (removal only −1.09 F1). When Conv\_1 is removed, Self-Attention's attention patterns collapse onto wrong positions (JSD 8.2× vs Conv\_0 control, answer focus drops 19%) — this is the proximate mechanism of the performance breakdown.

### Chain of Evidence

The three experiments build this answer incrementally:

1. **The gap is real** (Experiment 1): Two independent methods confirm Conv\_1's overwhelming dominance. Ablation ordering: Conv\_1 ≫ Attn > FFN > Conv\_0. Causal tracing ordering: Conv\_1 ≫ Conv\_0 > Attn > FFN. Key invariant: Conv\_1 far exceeds all other components in both methods, and Conv\_1 > Conv\_0 is consistent in direction. Conv\_0 rises to second in causal tracing yet ranks last in ablation — consistent with it contributing useful but compensable preprocessing. The 30× gap cannot be explained by output magnitude alone (norm ordering is inversely correlated with ablation ordering among Conv\_0/FFN/Attn).

2. **The difference is in specific dimensions of representational structure, not information content** (Experiment 2): Linear probes show all components encode answer-position information to a similar degree (AUC gap only 0.025), ruling out information difference as the primary explanation. Representational analysis reveals Conv\_1's output has significantly higher amplitude variation (Norm Variance 3.2×) and subspace dimensionality (Effective Rank 1.3×), with more spatially uniform structure (decay 0.058 vs 0.109). Conv\_0 actually scores slightly higher on directional distinctiveness yet is nearly dispensable, ruling out directional distinctiveness as a sufficient condition for the importance gap.

3. **Removing Conv\_1 causes attention collapse** (Experiment 3): Attention distortion under Conv\_1 removal is 8.2× greater than under Conv\_0 removal (per-sample mean ratio 8.84×, 95% CI: [8.54×, 9.15×]); entropy decreases by 0.259 and answer focus drops 19%. Pipeline distance is necessary but not sufficient for this asymmetry: Conv\_0, as Conv\_1's immediate upstream neighbor, causes minimal impact upon removal (−1.09 F1) — proving that proximity alone does not determine impact; what matters is irreplaceable functional dependence.

### Limitations

1. **Position and learned function are confounded**: Conv\_1's active transformation may arise from its pipeline position (receiving Conv\_0's output rather than raw input), training dynamics, or their interaction. Our eval-time experiments can observe the output differences but cannot separate the independent contributions of position and learned function. Separating them would require controlled experiments (e.g., swapping weights post-training and evaluating), which is prohibitively expensive in time and compute.

2. **Limited probe coverage**: We only probed one property (answer position). Differences in encoding of other task-relevant properties between Conv\_1 and Conv\_0 were not tested.

3. **Propagation effect of global ablation**: Globally removing Conv\_1 alters the entire residual stream. The attention degradation observed in Experiment 3 includes both direct impact and indirect effects propagated through the residual stream; the two cannot be separated in our current setup.

### Methodological Note

We tested an intervention spectrum (zero/mean/noise replacement) to separate information content from output magnitude. Under residual connections, this decomposition fails: injecting incorrect signals (mean or noise) causes more damage than injecting nothing (zero), because the residual stream propagates the error downstream. This confirms zero-out as the cleanest ablation for residual architectures and motivates our consistent use of it.
