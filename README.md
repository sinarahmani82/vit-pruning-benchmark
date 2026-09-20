# Reproducible Vision Transformer (ViT) & Magnitude Pruning Benchmark

[![CI Pipeline](https://github.com/sinarahmani82/vit-pruning-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/sinarahmani82/vit-pruning-benchmark/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

A modular, ground-up implementation of **Vision Transformer (ViT)** for image classification on CIFAR-10, coupled with an empirical study on **Unstructured Weight Magnitude Pruning**.

---

### 🔬 Motivation & Research Question
While Vision Transformers achieve state-of-the-art accuracy, their dense multi-head attention and projection layers incur heavy compute footprints. This project investigates:
> *How much parametric sparsity can a Vision Transformer tolerate before catastrophic degradation in representation and classification accuracy occurs?*

---

### 📂 Project Structure

```text
vit-pruning-benchmark/
│
├── .github/workflows/
│   └── ci.yml               # Automated CI test pipeline
├── src/
│   ├── __init__.py
│   ├── dataset.py           # CIFAR-10 pipeline with augmentations
│   ├── model.py             # ViT architecture (PatchEmbed, MHSA, TransformerBlock)
│   └── pruner.py            # Magnitude-based weight pruning engine
├── tests/
│   ├── __init__.py
│   └── test_vit.py          # Tensor shape & sparsity unit tests
├── requirements.txt         # Project dependencies
├── train.py                 # Full training & benchmarking pipeline
└── README.md
```

---

### 📊 Empirical Benchmark Results

| Model Configuration | Sparsity (%) | Top-1 Accuracy (%) | Latency (ms/sample) |
|---|---|---|---|
| **Baseline ViT (Dense)** | **0.0%** | **78.4%** | **1.42 ms** |
| Pruned ViT (Low) | 20.0% | 77.9% | 1.41 ms |
| Pruned ViT (Moderate) | 40.0% | 75.2% | 1.40 ms |
| Pruned ViT (Aggressive) | 60.0% | 68.1% | 1.39 ms |

*Key finding: The self-attention projection weights exhibit intrinsic redundancy, tolerating up to ~30-40% sparsity without retraining before significant accuracy drop occurs.*

---

### 🛠️ How to Reproduce

1. **Clone repository:**
   ```bash
   git clone https://github.com/sinarahmani82/vit-pruning-benchmark.git
   cd vit-pruning-benchmark
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run automated unit tests:**
   ```bash
   python -m pytest tests/
   ```

4. **Run full training & benchmark pipeline:**
   ```bash
   python train.py
   ```
