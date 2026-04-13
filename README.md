# QANet — Bug Repair & Mechanistic Investigation

**COMP4329/5329 Deep Learning · University of Sydney · Semester 1, 2026**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eOBr8ZURxk3jv8Tn1UxyPXCV26MFCaMY)
&nbsp;
[GitHub](https://github.com/1355-XCZ/Assignment1_2026)
&nbsp;|&nbsp;
[Google Drive (data & checkpoints)](https://drive.google.com/drive/folders/1eOBr8ZURxk3jv8Tn1UxyPXCV26MFCaMY)

---

## Overview

This repository contains:

1. **Bug Repair** — Identification and correction of 56 bugs in a distributed QANet implementation for extractive QA on SQuAD v1.1.
2. **Mechanistic Investigation** — Three hypothesis-driven experiments (H1–H3) probing the internal mechanisms of the repaired model using eval-time ablation, causal tracing, linear probing, and representational analysis.

The entire pipeline — setup, training, evaluation, and all experiments — is driven from a single notebook: **`assignment1.ipynb`**.

---

## Repository Structure

```
Assignment1_2026/
├── assignment1.ipynb          # Main notebook (run everything from here)
├── requirements.txt           # Python dependencies
├── README.md
│
├── Models/                    # QANet model components
│   ├── qanet.py               #   Full QANet model
│   ├── encoder.py             #   Encoder block (Conv, Attn, FFN)
│   ├── attention.py           #   CQ Attention & Self-Attention
│   ├── embedding.py           #   Word + Char embedding
│   ├── conv.py                #   Depthwise separable convolution
│   ├── heads.py               #   Pointer output layer
│   ├── dropout.py             #   Stochastic depth & dropout
│   ├── Activations/           #   ReLU, LeakyReLU
│   ├── Normalizations/        #   LayerNorm, GroupNorm
│   └── Initializations/       #   Xavier, Kaiming
│
├── Data/                      # Dataset loading & preprocessing
├── Tools/                     #   Download, preprocess, utilities
├── Losses/                    #   NLL loss
├── Optimizers/                #   Adam, SGD, SGD+Momentum
├── Schedulers/                #   Lambda, Cosine, Step schedulers
├── TrainTools/                #   Training loop, EMA
├── EvaluateTools/             #   Official SQuAD evaluation
│
└── experiments/               # Analysis outputs
    ├── QANet_Draft.tex        #   Full report (LaTeX)
    ├── references.bib         #   Bibliography
    ├── results/               #   JSON experiment results (generated)
    │   ├── H1/                #     Conv₀ vs Conv₁ mechanistic divergence
    │   ├── H2/                #     CQ Attention directional asymmetry
    │   └── H3/                #     Pointer asymmetric wiring
    └── prism-uploads/         #   Publication figures (PNG)
```

---

## Quick Start (Google Colab)

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone the repository

```python
import os
REPO_URL     = "https://github.com/1355-XCZ/Assignment1_2026.git"
PROJECT_ROOT = "/content/drive/MyDrive/Assignment1_2026"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
```

> **Note:** `drive.mount()` mounts the **current user's own Drive**. If you access the repository via a shared folder, either add a shortcut to "My Drive" (right-click → "Add shortcut to Drive") to keep the default path, or adjust `PROJECT_ROOT` accordingly.

### 3. Open & run

Navigate to `MyDrive/Assignment1_2026/` in Drive and open **`assignment1.ipynb`** in Colab. Run cells sequentially — the notebook handles:

- Dependency installation
- Data download & preprocessing (one-time)
- Model training with Adam + warmup + EMA
- Full evaluation on SQuAD v1.1 dev set
- All H1/H2/H3 experiments and publication figure generation

---

## Model Checkpoints

| Checkpoint | Description |
|---|---|
| `_model/adam/model_best.pt` | **Default.** Best checkpoint from our training run (F1 ≈ 70.24). All reported experiments use this. |
| `_model/model.pt` | Fallback checkpoint. Use if reproducing from a fresh training run. |

The notebook automatically falls back to `_model/model.pt` if the default is unavailable. Different checkpoints may shift sensitive numerical values, but **conclusions and trends remain stable** (verified across multiple runs).

---

## Experiments Summary

| Hypothesis | Research Question | Key Finding |
|---|---|---|
| **H1** | Why do identical Conv₀ and Conv₁ differ by ~30× in importance? | Conv₁ produces high-amplitude, high-rank outputs that Self-Attention irreplaceably depends on; Conv₀'s contribution is compensable via residual. |
| **H2** | Why does C→Q dominate while Q→C is near-redundant? | Answer tokens carry ~30× higher information density per token; C→Q injects this concentrated signal, while Q→C redistributes already-available information. |
| **H3** | Does Pointer asymmetric wiring reflect genuine functional specialisation? | No meaningful specialisation detected (ΔF1 ≤ 0.23, ΔAUC = 0.001). The asymmetric design is inherited from BiDAF but lacks information-level support under QANet's weight-shared encoding. |

---

## Dependencies

All dependencies are installed automatically by the notebook. Core requirements:

- Python 3.10+
- PyTorch (CUDA recommended)
- spacy, tqdm, tensorboard, matplotlib, seaborn, scikit-learn

See `requirements.txt` for the full list.

---

## Reproducibility Notes

- All experiments use **eval-time interventions** (no retraining required) — results are deterministic given the same checkpoint and data.
- Causal tracing uses 3 noise repeats per sample; minor run-to-run variance is expected but qualitative conclusions are robust.
- Figures are saved to `experiments/prism-uploads/` and loaded by the LaTeX report.

---

## License

This project is coursework for COMP4329/5329 at the University of Sydney. Please do not redistribute..
