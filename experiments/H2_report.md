# H2: Context vs Question — Dual-Stream Information Flow in QANet

## Hypothesis

> Question-stream corruption is more destructive than Context-stream corruption. CQ Attention is the critical information bottleneck where the two streams merge, and context information is localized to the answer span while question information is distributed.

## Background

QANet processes Context (C) and Question (Q) through separate Embedding Encoder tracks before merging them at CQ Attention (BiDAF-style Context-Query Attention). The merged representation is then resized and processed through 3 passes of the Model Encoder.

```
Context → EmbEnc_C ─┐
                     ├→ CQ Attention → CQ Resizer → Model Encoder (×3 passes)
Question → EmbEnc_Q ─┘
```

**Key question**: Are C and Q equally important, or does one stream dominate? And is CQ Attention truly the information bottleneck?

---

## Experiment 1: ROME-Style Causal Tracing (3 Corruption Conditions)

**Method**: For each sample, corrupt C, Q, or both (Gaussian noise, σ = 3× input std). Then restore individual component outputs from clean activations and measure recovery.

- **Total Effect (TE)**: P(span | clean) − P(span | corrupted)
- **Indirect Effect (IE)**: P(span | corrupted + restore component) − P(span | corrupted)
- **Normalized IE (NIE)**: IE / TE

**Corruption conditions**: Context-only, Question-only, Both

**Restoration points**: Embedding Encoder sub-layers (C/Q separately), CQ Attention output, CQ Resizer output

**Samples**: 200, Noise repeats: 3

### Key Metrics to Report

| Corruption | Mean TE | ±95%CI |
|---|---|---|
| context | [from h2_results.json] | |
| question | [from h2_results.json] | |
| both | [from h2_results.json] | |

### Analysis Framework

1. **TE Comparison**: If TE(Q) > TE(C) → question stream carries more causal weight
2. **Additivity test**: TE(both) vs TE(C) + TE(Q) → sub-additive implies information overlap; super-additive implies synergy
3. **CQ Attention NIE**: If CQ-Att restores >70% of TE under all conditions → it is the critical bottleneck
4. **Per-component NIE**: Which EmbEnc sub-layers matter most for each stream?

### Expected Findings

- TE asymmetry: one stream dominates (predicted: Q)
- CQ Attention as bottleneck: high NIE across all corruption conditions
- Interaction pattern: sub-additive or super-additive reveals whether C and Q carry redundant or complementary information

---

## Experiment 2: CQ Attention Ablation (Independent Bottleneck Validation)

**Method**: Zero out CQ Attention output at eval time (all downstream processing receives zeros + residual). Evaluate F1/EM on full dev set.

This provides **independent validation** of the bottleneck hypothesis without relying on causal tracing (which uses probabilistic metrics on subsampled data).

### Key Metrics to Report

| Condition | F1 | EM | ΔF1 | ΔEM |
|---|---|---|---|---|
| clean | | | — | — |
| skip_cq_att | | | | |

### Analysis Framework

1. **Magnitude of F1 drop**: If >50% of baseline F1 → catastrophic, confirming CQ-Att as bottleneck
2. **Cross-validation with Exp 1**: Causal tracing measures information *flow* (recovery); ablation measures information *necessity* (removal). Both should agree.
3. **Residual contribution**: The model still has residual connections bypassing CQ attention. Any remaining F1 indicates information leakage through residuals.

### Limitations

- Zero ablation conflates information removal and magnitude removal
- CQ Resizer processes zeroed input → downstream effects are indirect
- Confirms necessity but not sufficiency

---

## Experiment 3: Selective Corruption — Information Localization

**Method**: Selectively corrupt only specific positions in the context to determine whether context information is localized (concentrated at the answer span) or distributed (spread across all tokens).

**Conditions**:
- `ans_only`: Corrupt only answer-span positions (y1 to y2)
- `non_ans_only`: Corrupt only non-answer, non-padding positions
- `full_context`: Corrupt all context positions (same as Exp 1 context condition)

Also measures CQ Attention IE for each condition to test whether the bottleneck mediates localized and distributed information differently.

**Samples**: 200, Noise repeats: 3

### Key Metrics to Report

| Condition | Mean TE | Avg #Tokens | TE/Token |
|---|---|---|---|
| ans_only | | | |
| non_ans_only | | | |
| full_context | | | |

**Information density ratio**: TE/token(answer) ÷ TE/token(non-answer)

### Analysis Framework

1. **TE per token comparison**: Higher TE/token at answer positions → context info is localized
2. **Additivity**: TE(ans) + TE(non-ans) vs TE(full) → reveals information interaction
3. **CQ Attention recovery by condition**: Does CQ-Att mediate localized and distributed info equally?
4. **Connection to TE asymmetry**: If context info is localized to ~N answer tokens but question info is spread across ~M tokens, and M < N\_context but M ≈ all Q tokens → question has higher global information density → explains why Q corruption is more destructive

### Limitations

- Corrupting answer-span tokens directly removes the target signal (high TE is expected and somewhat trivial)
- The informative comparison is whether non-answer corruption is substantial
- Position-specific noise may not perfectly isolate positional vs semantic information

---

## Synthesis

### Evidence Chain

1. **Exp 1 (Causal Tracing)**: Establishes TE asymmetry between C and Q streams; shows CQ-Att has high NIE (information flow evidence)
2. **Exp 2 (CQ Ablation)**: Independently confirms CQ-Att is necessary for performance (information necessity evidence)
3. **Exp 3 (Selective Corruption)**: Explains *why* the asymmetry exists through information localization analysis

### Cross-Method Consistency

| Claim | Exp 1 Evidence | Exp 2 Evidence | Exp 3 Evidence |
|---|---|---|---|
| CQ-Att is bottleneck | NIE > 70% | F1 drop > 50% | CQ recovery varies by locality |
| Q more important than C | TE(Q) > TE(C) | N/A | Info density analysis |
| Context info is localized | N/A | N/A | TE/token ratio |

### Methodological Notes

- All three experiments use different metrics (probability-based TE/IE, F1/EM, per-token TE density) to avoid single-method bias
- Causal tracing and ablation are complementary: one adds information back, the other removes it
- Selective corruption provides mechanism-level understanding that global metrics cannot
