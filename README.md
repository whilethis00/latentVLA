# LatentVLA: Hierarchical Latent Variable Models for Robot Action Generation

A research project comparing hierarchical latent variable model architectures for robotic manipulation tasks. The project explores how different approaches to modeling action distributions — deterministic vs. stochastic, VAE-based vs. flow-based — affect action generation quality and diversity.

---

## Overview

**Task:** Given visual observations, language instructions, and proprioceptive state, generate sequences of robot actions.

**Core Question:** How should the latent space be structured to best model diverse, multimodal action distributions?

The project implements and compares four model variants using **Conditional Flow Matching (OT-CFM)** as the underlying generative framework, with a frozen **SigLIP** encoder for vision-language grounding.

---

## Project Structure

```
VLA/
├── configs/
│   └── default.yaml               # Main hyperparameter configuration
├── data/
│   ├── robomimic_dataset.py        # HDF5 dataset loader (image / low_dim modes)
│   └── libero_dataset.py           # LIBERO task suite loader
├── models/
│   ├── encoders.py                 # ContextEncoder (SigLIP + proprio fusion)
│   ├── flat_flow.py                # [M1] Baseline: direct action flow
│   ├── det_latent.py               # [M2] Deterministic latent + action flow
│   ├── stoch_latent_vae.py         # [M3] VAE-based stochastic latent
│   ├── stoch_latent_flow_prior.py  # [M4] Two-level stochastic flow (proposed)
│   └── flow_utils.py               # OT-CFM utilities (VelocityMLP, Euler integration)
├── training/
│   ├── builder.py                  # Factory: datasets / dataloaders / models
│   └── trainer.py                  # Training loop, evaluation, checkpointing
├── evaluation/
│   └── metrics.py                  # OfflineEvaluator with best-of-K, z_shuffle_gap, etc.
├── scripts/
│   ├── train.py                    # Main training entry point
│   ├── evaluate_offline.py         # Offline evaluation (best-of-K, temperature sweep)
│   ├── plot_training.py            # Training curve visualization
│   ├── smoke_test.py               # Unit tests (no real data required)
│   └── run_ablations.sh            # Run all 4 core experiments sequentially
└── requirements.txt
```

---

## Model Variants

| ID | Model | Latent Type | Key Idea |
|----|-------|-------------|----------|
| M1 | **FlatFlow** | None | Baseline — direct flow from noise to actions |
| M2 | **DetLatent** | Deterministic | Predict latent z with a prior head, condition action flow on z |
| M3 | **StochVAE** | Stochastic (VAE) | Gaussian posterior/prior with KL regularization |
| M4 | **StochFlowPrior** | Stochastic (Flow) | Two-level flows: latent flow + action flow (no KL needed) |

### Architecture Summary

```
Context Encoder (shared):
  image (224×224) → SigLIP [frozen] → patch embeddings
  language → SigLIP text encoder [frozen]
  proprio → MLP
  → fused context C_t ∈ R^256

M1 FlatFlow:
  C_t → VelocityMLP → actions (H × action_dim)

M2 DetLatent:
  C_t → PriorHead → z_hat ∈ R^128
  (C_t, z_hat) → VelocityMLP → actions

M3 StochVAE:
  Posterior: q(z | C_t, a, future) → z ~ N(μ, σ²)
  Prior:     p(z | C_t) → z ~ N(μ_p, σ_p²)  [KL loss]
  (C_t, z) → VelocityMLP → actions

M4 StochFlowPrior (proposed):
  Latent flow:  ε → z  (flow matching, no KL)
  Action flow:  ε → a  conditioned on (C_t, z)
```

### Training Losses

- **Action loss:** OT-CFM flow matching — `||v_θ(x_t, t, C) − u_t||²`
- **Prior loss:** MSE (M2), KL divergence (M3), or flow matching (M4)
- **Semantic auxiliary loss:** `1 − cosine_sim(pred_future_emb, GT_future_emb)` × 0.1

---

## Key Results

All evaluations performed on held-out validation set (10% of dataset) after 100 epochs.

### StochFlowPrior — Temperature Sweep

| Temperature | Action MSE (prior) | Action MSE (posterior) | Best-of-5 | Sampling Diversity | z-Shuffle Gap |
|-------------|---------------------|------------------------|-----------|-------------------|---------------|
| 1.0         | 0.3797              | 0.2999                 | 0.2421    | 0.2781            | 0.7249        |
| 0.7         | 0.3886              | 0.2999                 | 0.2467    | 0.2452            | 0.7041        |
| 0.5         | 0.3751              | 0.3093                 | 0.2802    | 0.2359            | 0.6864        |
| 0.3         | 0.3848              | 0.3180                 | 0.3027    | 0.2307            | 0.6986        |

**Future cosine similarity: 0.9999** across all temperatures (excellent semantic grounding)

### Training Curve (StochFlowPrior)

| Epoch | Total Loss | Action Flow Loss | Prior Flow Loss | Semantic Loss |
|-------|------------|-----------------|-----------------|---------------|
| 1     | 3.823      | 1.628           | 2.103           | 0.918         |
| 10    | ~1.2       | ~0.6            | ~0.6            | ~0.03         |
| 50    | ~0.45      | ~0.38           | ~0.07           | ~0.0001       |
| 100   | 0.390      | 0.355           | 0.035           | 0.000074      |

---

## Experimental Insights

### 1. Latent Space is Critical (z-Shuffle Gap)
Shuffling the latent z across batch elements causes ~72% performance degradation (`z_shuffle_gap ≈ 0.72`). This confirms the latent code carries meaningful action-conditioning information, not just noise.

### 2. Prior Recovery is Decent but Imperfect
The prior-posterior gap of ~8% (`prior_posterior_gap ≈ 0.080`) at test time shows the prior reasonably approximates the posterior — but there is room for improvement. The flow-based prior avoids the posterior collapse risk of VAE.

### 3. Best-of-K Shows Real Diversity Benefit
`best_of_5 = 0.242` vs `best_of_1 = 0.386` — sampling 5 candidates and picking the best reduces MSE by ~37%. This demonstrates the stochastic model generates meaningfully diverse action modes.

### 4. Temperature Trade-off
- Low temperature (0.3–0.5): reduces diversity, but MSE doesn't reliably improve — the mode can shift away from GT
- Temperature = 1.0 gives the best balance: highest diversity with competitive MSE

### 5. Semantic Auxiliary Loss Converges Immediately
Semantic loss drops from 0.918 → ~0.03 within 10 epochs and becomes negligible by epoch 20. The future SigLIP embedding is predicted near-perfectly (`cosine_sim ≈ 0.9999`), but this auxiliary loss still improves action prediction by grounding the context representation.

### 6. Two-Level Flow vs VAE: Hypothesis
Unlike VAE, the StochFlowPrior avoids explicit KL regularization. Instead, the prior flow directly matches the posterior's z-distribution. This is expected to:
- Avoid posterior collapse
- Allow richer multimodal z distributions
- Be more stable to train (no β-VAE tuning needed)

### 7. Ablation Summary (from run directories)

| Experiment | Description |
|-----------|-------------|
| `stoch_flow_prior` | Full model (M4) with semantic loss |
| `sfp_nosem` | M4 without semantic auxiliary loss |
| `sfp_planner_full` | M4 with full state as planner input |
| `sfp_planner_object_only` | M4 with only object state (10-dim) |
| `sfp_planner_proprio_only` | M4 with only proprioception |
| `sfp_zdim32` / `sfp_zdim64` | M4 with latent dim 32 / 64 |
| `flat_flow` | Baseline M1 |
| `det_latent` | Baseline M2 |
| `stoch_vae` | Baseline M3 |
| `svae_nosem` | M3 without semantic loss |

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Smoke Test (no data required)
```bash
python scripts/smoke_test.py
```

### Training
```bash
# Train StochFlowPrior (M4)
python scripts/train.py \
    --config configs/default.yaml \
    --override model.type=stoch_flow_prior \
               data.dataset_type=robomimic \
               data.dataset_path=/path/to/dataset.hdf5

# Train all 4 baselines sequentially
bash scripts/run_ablations.sh
```

### Evaluation
```bash
python scripts/evaluate_offline.py \
    --run_dir outputs/runs/stoch_flow_prior \
    --best_of_k 1 3 5 10 \
    --temperature_sweep 1.0 0.7 0.5 0.3
```

### Plot Training Curves
```bash
python scripts/plot_training.py \
    --runs outputs/runs/flat_flow outputs/runs/stoch_flow_prior \
    --metric action_flow_loss
```

---

## Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  type: stoch_flow_prior   # flat_flow | det_latent | stoch_vae | stoch_flow_prior
  flow_steps: 10           # ODE integration steps at inference
  planner_input: full      # full | object_only | proprio_only

latent:
  z_dim: 128               # Latent dimension

loss:
  semantic_future_weight: 0.1   # Auxiliary semantic loss weight
  kl_beta: 1.0                  # VAE KL weight (M3 only)

training:
  num_epochs: 100
  learning_rate: 3.0e-4
  batch_size: 64
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `action_mse_prior` | MSE between prior-sampled actions and ground truth |
| `action_mse_posterior` | MSE using posterior z (upper bound) |
| `best_of_K` | Min MSE across K samples — measures diversity utility |
| `future_cosine_sim` | Cosine similarity of predicted vs. actual future embedding |
| `sampling_diversity` | Std dev of sampled actions — measures output variance |
| `z_shuffle_gap` | Performance drop when z is shuffled — measures z importance |
| `prior_posterior_gap` | `mse_prior - mse_posterior` — measures prior quality |

---

## Dependencies

- PyTorch >= 2.1.0
- Transformers >= 4.37.0 (SigLIP)
- einops, h5py, pyyaml, tqdm, matplotlib, scikit-learn
- wandb (optional, for experiment tracking)

---

## Datasets

- **Robomimic** — HDF5 format, image or low-dim modes supported
- **LIBERO** — Task suite (libero_object, libero_long), includes language annotations
