# Qwen2.5-3B Parameter Sensitivity Experiments

Experimental study of parameter sensitivity in **Qwen2.5-3B-Instruct** using controlled in-memory perturbations.

The project investigates how modifying individual Transformer layers affects model behavior on different types of tasks.

> **Status:** Experiment 1 completed — initial 36-layer sensitivity scan.

---

## Overview

This repository contains experiments for studying the effect of controlled parameter modifications on a pretrained Transformer model.

The current experiment:

1. Loads the original Qwen2.5-3B-Instruct model.
2. Establishes a baseline using fixed prompts.
3. Temporarily modifies one Transformer layer.
4. Runs the same evaluation again.
5. Measures the change in loss.
6. Restores the original parameters.
7. Repeats the process across all 36 Transformer layers.
8. Saves the results for further analysis.

The original model checkpoint is kept unchanged.

---

## Model

**Model:** Qwen2.5-3B-Instruct

Configuration used in the experiment:

- Parameters: ~3.09B
- Transformer layers: 36
- Weight format: BF16
- Checkpoint format: Safetensors
- Number of model shards: 2

Model structure includes:

```text
Qwen2.5-3B
│
├── Embeddings
│
├── Transformer Layers × 36
│   │
│   ├── Input LayerNorm
│   │
│   ├── Self Attention
│   │   ├── Q Projection
│   │   ├── K Projection
│   │   ├── V Projection
│   │   └── O Projection
│   │
│   ├── Post Attention LayerNorm
│   │
│   └── MLP
│       ├── Gate Projection
│       ├── Up Projection
│       └── Down Projection
│
├── Final LayerNorm
│
└── LM Head
```

The model is loaded directly with Transformers rather than through an inference wrapper so that individual parameters can be inspected and experimentally modified.

---

## Hardware

Current test machine:

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti Max-Q |
| VRAM | 6 GB |
| RAM | 16 GB |
| OS | Ubuntu |
| NVIDIA Driver | 580.178.04 |
| CUDA | 13.0 |
| Python | 3.12.3 |

Because the model is larger than the available GPU memory in BF16, Transformers uses automatic CPU/GPU offloading.

The current configuration places part of the model on the GPU and part on system memory.

---

## Environment

Python virtual environment:

```text
~/cpu-os/ai_experiment/.venv
```

Activate it with:

```bash
source ~/cpu-os/ai_experiment/.venv/bin/activate
```

Main packages used:

- PyTorch
- Transformers
- Safetensors
- Accelerate
- Hugging Face Hub

CUDA availability can be checked with:

```python
import torch

print(torch.cuda.is_available())
```

---

## Project Structure

```text
ai_experiment/
│
├── models/
│   └── qwen2.5-3b/
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       └── model.safetensors.index.json
│
├── inference.py
│
└── experiments/
    ├── benchmark_small.json
    ├── run_experiment.py
    │
    └── results/
        ├── baseline.json
        ├── layer_00.json
        ├── layer_01.json
        ├── ...
        ├── layer_35.json
        └── sensitivity_summary.json
```

---

# Experiment 1 — Layer Sensitivity

## Method

A fixed benchmark is first evaluated using the untouched model.

For each Transformer layer:

```text
Original Layer
      ↓
Temporary perturbation
      ↓
Run benchmark
      ↓
Measure loss
      ↓
Restore original Layer
      ↓
Next Layer
```

The current intervention scales every parameter in the selected layer by:

```text
0.90
```

In other words:

```text
W_modified = 0.90 × W_original
```

The modification is performed in memory.

The original Safetensors checkpoint is not overwritten.

---

## Evaluation Metric

The current experiment uses **Negative Log-Likelihood (NLL)** on the baseline-generated answer.

For each question:

```text
Baseline Loss
      ↓
Perturb Layer
      ↓
Perturbed Loss
      ↓
ΔNLL = Perturbed Loss - Baseline Loss
```

A larger positive `ΔNLL` means the perturbation caused the model to assign lower probability to the baseline answer.

A negative value means the perturbed model assigned a lower loss to that particular baseline answer.

This metric is used as a **sensitivity measurement**, not as a direct capability score.

---

## Initial Benchmark

The first test intentionally uses a very small benchmark so that the experimental pipeline can be validated quickly.

### Coding

> Write a Python function that checks whether a given string is a palindrome. Explain how it works.

### Mathematics

> A shop gives a 20% discount on a product priced at 1500 rupees. What is the final price? Show the calculation.

Only two questions were used in the initial scan.

A larger benchmark is required before making stronger conclusions.

---

# Initial Results

The first 36-layer scan completed successfully.

### Layer Sensitivity

| Layer | Coding | Math | Overall |
|---:|---:|---:|---:|
| 0 | 0.041296 | 0.012738 | 0.027017 |
| 1 | 0.028968 | 0.003788 | 0.016378 |
| 2 | 0.007593 | 0.001403 | 0.004498 |
| 3 | 0.003365 | -0.001743 | 0.000811 |
| 4 | 0.006146 | 0.002901 | 0.004523 |
| 5 | 0.002617 | -0.002460 | 0.000079 |
| 6 | 0.014765 | 0.006995 | 0.010880 |
| 7 | 0.015198 | -0.000948 | 0.007125 |
| 8 | 0.027351 | 0.001565 | 0.014458 |
| 9 | 0.018912 | -0.001910 | 0.008501 |
| 10 | 0.018800 | 0.004071 | 0.011436 |
| 11 | 0.019290 | 0.006063 | 0.012677 |
| 12 | 0.016604 | 0.003203 | 0.009904 |
| 13 | -0.004623 | 0.001153 | -0.001735 |
| 14 | 0.013794 | 0.009277 | 0.011535 |
| 15 | 0.017715 | 0.005060 | 0.011388 |
| 16 | 0.013706 | 0.001671 | 0.007688 |
| 17 | 0.010135 | -0.003150 | 0.003492 |
| 18 | 0.011665 | 0.009847 | 0.010756 |
| 19 | 0.008549 | -0.004025 | 0.002262 |
| 20 | 0.013693 | 0.001401 | 0.007547 |
| 21 | 0.014387 | -0.001979 | 0.006204 |
| 22 | 0.012912 | -0.007529 | 0.002691 |
| 23 | 0.001495 | 0.000249 | 0.000872 |
| 24 | 0.013775 | 0.023236 | 0.018505 |
| 25 | 0.010182 | -0.011009 | -0.000413 |
| 26 | 0.029043 | 0.008502 | 0.018772 |
| 27 | 0.023854 | 0.007892 | 0.015873 |
| 28 | 0.038524 | -0.003540 | 0.017492 |
| 29 | 0.000000 | 0.000000 | 0.000000 |
| 30 | 0.000000 | 0.000000 | 0.000000 |
| 31 | 0.000000 | 0.000000 | 0.000000 |
| 32 | 0.000000 | 0.000000 | 0.000000 |
| 33 | 0.000000 | 0.000000 | 0.000000 |
| 34 | 0.000000 | 0.000000 | 0.000000 |
| 35 | 0.000000 | 0.000000 | 0.000000 |

---

# Important Experimental Limitation

Layers 29–35 returned exactly zero change.

This should **not** currently be interpreted as evidence that these layers are insensitive or unimportant.

The model uses automatic CPU/GPU offloading, and some parameters were placed on the `meta` device by the Transformers offloading mechanism.

The current intervention implementation did not successfully modify those offloaded parameters.

Therefore:

```text
Layer 29–35 = 0
```

currently means:

> The intervention was not successfully applied to these parameters.

Further work is required before these layers can be compared with layers 0–28.

---

# Scientific Interpretation

The current results are preliminary.

For example, Layer 0 showed:

```text
Coding ΔNLL = 0.041296
Math ΔNLL   = 0.012738
```

This means that, under this specific 0.90 scaling intervention and these specific questions, the coding answer's loss changed more than the math answer's loss.

It does **not** prove that:

```text
Layer 0 = coding
```

or that Layer 0 exclusively represents a particular capability.

Transformer capabilities can be distributed across many parameters and layers.

Future experiments therefore need:

- More questions
- More task categories
- Multiple intervention strengths
- More granular parameter groups
- Repeated measurements
- Controlled experimental settings

---

# Next Experiments

Planned experimental progression:

### Experiment 2 — Larger Benchmark

Increase the number of questions across:

```text
Coding
Mathematics
Logical Reasoning
General Language
```

### Experiment 3 — Fix Offloaded-Layer Intervention

Ensure CPU-offloaded layers can be correctly and safely perturbed.

### Experiment 4 — Component-Level Sensitivity

Instead of perturbing an entire layer:

```text
Layer
├── Attention
│   ├── Q
│   ├── K
│   ├── V
│   └── O
│
└── MLP
    ├── Gate
    ├── Up
    └── Down
```

Each component will be tested separately.

### Experiment 5 — Intervention Comparison

Compare different intervention methods:

```text
Scaling
Zeroing / masking
Controlled noise
Other controlled transformations
```

---

# Reproducibility

Experiments should keep the following fixed whenever possible:

- Model checkpoint
- Model precision
- Benchmark questions
- Generation settings
- Intervention strength
- Tokenization method
- Evaluation procedure

Experiment metadata and results should be saved alongside the experiment outputs.

---

# Safety of the Original Model

The original model checkpoint should remain untouched.

Do not modify these files directly:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

Experimental interventions should be performed in memory whenever possible.

Generated experiment results should be stored separately from the original model files.

---

# Status

| Component | Status |
|---|---|
| Qwen2.5-3B setup | ✅ Complete |
| Direct Transformers loading | ✅ Complete |
| GPU/CPU offloading | ✅ Working |
| Baseline inference | ✅ Complete |
| Small benchmark | ✅ Complete |
| 36-layer sensitivity scan | ✅ Complete |
| Original checkpoint preserved | ✅ |
| Offloaded-layer intervention | ⚠️ Needs improvement |
| Larger benchmark | 🔜 |
| Component-level experiments | 🔜 |

---

## License

This repository is an experimental research project.

Model weights are not included in this repository. Refer to the original model's licensing and usage terms for the Qwen2.5-3B-Instruct checkpoint.