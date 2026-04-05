# Experiment Summary: M1 ~ M4

## What is this project?

This project studies **how to make robots learn complex manipulation tasks** from demonstration data.

The key idea: a robot needs to understand **"what is the goal of this task?"** before deciding **"how to move my arm?"**. We model this with a two-level architecture:

1. **Context encoder** — reads the current image + proprioception + language instruction → produces a task context vector
2. **Latent variable z** — a compact "plan" that captures what the robot should do next (the quality of z is what we're investigating)
3. **Flow-based action decoder** — takes z + context → generates a sequence of robot actions

The four models (M1–M4) differ only in **how z is computed**, ranging from "no z at all" to "z learned with a flow-based prior".

---

## Key Metrics Explained

| Metric | What it means | Lower/Higher = better? |
|--------|--------------|------------------------|
| **MSE (prior)** | Action prediction error using only the context (no ground-truth future info) — **real inference condition** | Lower |
| **MSE (posterior)** | Action prediction error when z has access to future ground-truth — upper bound on z quality | Lower |
| **z_shuffle_gap** | How much performance drops when z is randomly shuffled across samples. High value → z carries task-critical information | Higher |
| **prior_posterior_gap** | Gap between prior and posterior MSE. High → the model heavily relies on z at inference | Higher |
| **best_of_5** | Best MSE among 5 sampled z's — measures sampling diversity benefit | Lower |

---

## Models

| ID | Model | z Type | One-line description |
|----|-------|--------|----------------------|
| M1 | **FlatFlow** | None | Direct flow: context → actions, no latent variable |
| M2 | **DetLatent** | Deterministic | z = MLP(future image feature); fixed at train time, no uncertainty |
| M3 | **StochVAE** | Stochastic (VAE) | z ~ VAE posterior; KL-regularized, Gaussian prior |
| M4 | **StochFlowPrior** | Stochastic (Flow) | z ~ flow-based prior; richer prior distribution |

---

## Results (Epoch 100, LIBERO-Object dataset)

| Model | MSE (prior) ↓ | MSE (posterior) ↓ | z_shuffle_gap ↑ | prior_posterior_gap ↑ | best_of_5 ↓ |
|-------|:-------------:|:-----------------:|:---------------:|:---------------------:|:-----------:|
| M1 FlatFlow      | 0.5530 | —      | —       | —      | —      |
| M2 DetLatent     | 0.4776 | 0.0017 | **0.7837** | **0.4759** | —      |
| M3 StochVAE      | 0.6321 | 0.6205 | -0.0039 | 0.0116 | 0.3180 |
| M4 StochFlowPrior| 0.6540 | 0.5395 | 0.0428  | 0.1144 | **0.3638** |

---

## Key Takeaways

### M1 (FlatFlow) — Baseline
- No latent variable. The context vector directly drives the flow decoder.
- MSE 0.553: reasonable, but no mechanism to express multi-modal futures.

### M2 (DetLatent) — Best MSE, but uses oracle future
- z is computed from the **ground-truth future image** at training time.
- `mse_posterior = 0.0017` → near-perfect when future is known.
- `z_shuffle_gap = 0.784` → z carries extremely rich task information.
- **Catch**: at inference, future image is not available, so z must be predicted from context. This is the unsolved gap.

### M3 (StochVAE) — VAE prior collapses
- `z_shuffle_gap ≈ 0` and `prior_posterior_gap ≈ 0.01` → z carries almost no useful information.
- KL regularization forces z toward a simple Gaussian, **killing z's expressiveness**.
- Prior ≈ Posterior MSE → the model ignores z entirely and solves the task without it (KL collapse).

### M4 (StochFlowPrior) — Flow prior maintains diversity
- A normalizing flow replaces the Gaussian prior, allowing a richer distribution.
- `z_shuffle_gap = 0.043` and `prior_posterior_gap = 0.114` → z is more utilized than VAE.
- `best_of_5 = 0.364` → sampling multiple z's and picking the best helps.
- Still much weaker z utilization than M2 → the prior is not expressive enough yet.

### Overall message
**z quality determines action quality.** M2 proves this clearly (z_shuffle_gap 0.78, MSE 0.0017 with oracle z). The challenge is learning a prior that matches the posterior quality of M2 — this is what the VLM-based system (M5, LatentVLA) aims to solve.

---

## Experiment Directories

| Run | Directory |
|-----|-----------|
| M1  | `outputs/runs/flat_flow_100ep_20260404/` |
| M2  | `outputs/runs/det_latent_100ep_20260405/` |
| M3  | `outputs/runs/stoch_vae_100ep_20260405/` |
| M4  | `outputs/runs/sfp_100ep_20260401/` |
