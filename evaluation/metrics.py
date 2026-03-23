"""
Offline Evaluation Metrics
──────────────────────────

Required metrics (from experiment spec):
  1. action_pred_loss   — MSE between predicted and GT action chunk
  2. future_cosine_sim  — cosine similarity between predicted & GT future embedding
  3. z_shuffle_gap      — performance drop when z is shuffled within batch
  4. prior_posterior_gap— posterior z vs prior z performance delta
  5. sampling_diversity  — std of generated action chunks across n_samples
  6. [optional] latent_stats — mu/std statistics for analysis

Usage:
    evaluator = OfflineEvaluator(context_encoder, policy, device)
    metrics = evaluator.evaluate(val_loader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
import numpy as np


class OfflineEvaluator:
    def __init__(
        self,
        context_encoder: nn.Module,
        policy: nn.Module,
        device: torch.device,
        n_diversity_samples: int = 5,
        best_of_ks: list = None,
        std_scale: float = 1.0,
        semantic_feat_dim: Optional[int] = None,
    ):
        self.context_encoder = context_encoder
        self.policy = policy
        self.device = device
        self.n_diversity_samples = n_diversity_samples
        self.best_of_ks = best_of_ks or [5]
        self.std_scale = std_scale  # temperature: scale applied to sampling noise

        # Detect model type
        policy_cls = type(policy).__name__
        self.model_type = {
            "FlatFlow": "flat_flow",
            "DetLatent": "det_latent",
            "StochLatentVAE": "stoch_vae",
            "StochLatentFlowPrior": "stoch_flow_prior",
        }.get(policy_cls, "unknown")

        self.has_latent = self.model_type != "flat_flow"
        self.is_stochastic = self.model_type in ("stoch_vae", "stoch_flow_prior")

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.context_encoder.eval()
        self.policy.eval()

        best_of_k_keys = [f"best_of_{k}" for k in self.best_of_ks]
        all_metrics = {
            "action_mse_prior": [],      # MSE(pred_prior, GT)
            "action_mse_posterior": [],  # MSE(pred_posterior, GT) — upper bound
            **{k: [] for k in best_of_k_keys},  # best-of-K (stochastic only)
            "future_cosine_sim": [],
            "sampling_diversity": [],    # std across samples (stochastic only)
            "z_shuffle_gap": [],         # only for latent models
        }

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Support both image-mode and low_dim-mode batches
            if "image" in batch:
                image = batch["image"].to(self.device)
            else:
                image = batch["state"].to(self.device)
            proprio = batch["proprio"].to(self.device)
            actions = batch["actions"].to(self.device)
            future_input = batch["future_image"].to(self.device)
            langs = list(batch["language"])

            input_ids, attention_mask = self.context_encoder.tokenize(langs, self.device)
            context = self.context_encoder(image, proprio, input_ids, attention_mask)

            # Future feature
            future_feat = self.context_encoder.encode_future_image(future_input)

            B, H, action_dim = actions.shape

            # 1. Action MSE (prior mode)
            if self.model_type == "flat_flow":
                pred_prior = self.policy.predict(context)  # (B, H, action_dim)
            else:
                pred_prior = self.policy.predict(context, use_posterior=False)

            mse_prior = F.mse_loss(pred_prior, actions).item()
            all_metrics["action_mse_prior"].append(mse_prior)

            # 2. Action MSE (posterior mode) — upper bound
            if self.has_latent:
                pred_post = self.policy.predict(
                    context,
                    use_posterior=True,
                    actions_for_posterior=actions,
                    future_feat_for_posterior=future_feat,
                )
                mse_post = F.mse_loss(pred_post, actions).item()
                all_metrics["action_mse_posterior"].append(mse_post)

            # 3. Future cosine similarity
            # Use policy's semantic head if available, else skip
            future_cos = self._compute_future_cosine(context, actions, future_feat)
            if future_cos is not None:
                all_metrics["future_cosine_sim"].append(future_cos)

            # 4. z-shuffle gap (latent models only)
            if self.has_latent:
                shuffle_gap = self._compute_z_shuffle_gap(context, actions)
                all_metrics["z_shuffle_gap"].append(shuffle_gap)

            # 5. Sampling diversity + best-of-K (stochastic models only)
            if self.is_stochastic:
                max_k = max(self.best_of_ks)
                div, best_k_mses = self._compute_diversity_and_best_k(context, actions, max_k)
                all_metrics["sampling_diversity"].append(div)
                for k in self.best_of_ks:
                    all_metrics[f"best_of_{k}"].append(best_k_mses[k])

        # Aggregate
        result = {}
        for k, vals in all_metrics.items():
            if vals:
                result[k] = float(np.mean(vals))

        # Prior-posterior gap (derived)
        if "action_mse_prior" in result and "action_mse_posterior" in result:
            result["prior_posterior_gap"] = (
                result["action_mse_prior"] - result["action_mse_posterior"]
            )

        return result

    # ── Individual metric computations ───────────────────────────────────

    def _compute_future_cosine(self, context, actions, future_feat):
        """Compute cosine sim between z's semantic prediction and GT future feat."""
        policy = self.policy
        if not hasattr(policy, "semantic_head"):
            return None
        if not hasattr(policy, "posterior_enc"):
            return None

        try:
            from models.stoch_latent_vae import GaussianEncoder
            B = context.shape[0]
            action_dim = actions.shape[-1]
            H = actions.shape[1]

            # Get posterior z*
            parts = [context, actions.reshape(B, -1)]
            if future_feat is not None and getattr(policy, "use_future", False):
                parts.append(future_feat)
            q_in = torch.cat(parts, -1)
            mu_q, logvar_q = policy.posterior_enc(q_in)
            z = mu_q  # use mean for evaluation

            pred_future = policy.semantic_head(z)
            cos_sim = F.cosine_similarity(
                F.normalize(pred_future, dim=-1),
                F.normalize(future_feat, dim=-1),
                dim=-1,
            ).mean().item()
            return cos_sim
        except Exception:
            return None

    def _compute_z_shuffle_gap(self, context, actions):
        """
        Compute drop in action MSE when z is shuffled within the batch.
        Larger gap → z carries useful information.
        """
        B = context.shape[0]
        if B < 2:
            return 0.0

        # Get baseline (prior z)
        pred_normal = self.policy.predict(context, use_posterior=False)
        mse_normal = F.mse_loss(pred_normal, actions).item()

        # Shuffle z within batch
        policy = self.policy
        if not hasattr(policy, "prior") and not hasattr(policy, "prior_enc") and not hasattr(policy, "prior_flow"):
            return 0.0

        try:
            # Get z from prior
            z = self._get_prior_z(policy, context)
            if z is None:
                return 0.0

            # Shuffle z
            perm = torch.randperm(B, device=self.device)
            z_shuffled = z[perm]

            # Reconstruct prediction with shuffled z
            pred_shuffled = self._predict_with_z(policy, context, z_shuffled, actions.shape)
            if pred_shuffled is None:
                return 0.0

            mse_shuffled = F.mse_loss(pred_shuffled, actions).item()
            return mse_shuffled - mse_normal  # positive = z was useful
        except Exception:
            return 0.0

    def _get_prior_z(self, policy, context):
        """Extract prior z (deterministic/mean for VAE, flow sample for FlowPrior)."""
        from models.det_latent import DetLatent
        from models.stoch_latent_vae import StochLatentVAE
        from models.stoch_latent_flow_prior import StochLatentFlowPrior
        from models.flow_utils import euler_integrate

        if isinstance(policy, DetLatent):
            return policy.prior(context)
        elif isinstance(policy, StochLatentVAE):
            mu_p, _ = policy.prior_enc(context)
            return mu_p
        elif isinstance(policy, StochLatentFlowPrior):
            # Use prior flow to sample z (noise → z)
            return euler_integrate(policy.prior_flow, context, policy.z_dim, policy.flow_steps)
        return None

    def _predict_with_z(self, policy, context, z, action_shape):
        """Generate actions using an externally provided z."""
        from models.flow_utils import euler_integrate
        from models.det_latent import DetLatent
        from models.stoch_latent_vae import StochLatentVAE
        from models.stoch_latent_flow_prior import StochLatentFlowPrior

        B, H, action_dim = action_shape
        x_dim = H * action_dim

        try:
            if isinstance(policy, DetLatent):
                cond = torch.cat([context, z], dim=-1)
                x = euler_integrate(policy.velocity_net, cond, x_dim, policy.flow_steps)
            elif isinstance(policy, StochLatentVAE):
                cond = torch.cat([context, z], dim=-1)
                x = euler_integrate(policy.velocity_net, cond, x_dim, policy.flow_steps)
            elif isinstance(policy, StochLatentFlowPrior):
                cond = torch.cat([context, z], dim=-1)
                x = euler_integrate(policy.action_flow, cond, x_dim, policy.flow_steps)
            else:
                return None
            return x.reshape(B, H, action_dim)
        except Exception:
            return None

    def _compute_diversity_and_best_k(self, context, actions, max_k):
        """
        Generate max_k samples, compute diversity and best-of-k for each k in best_of_ks.
        Returns (diversity_std, {k: best_of_k_mse}).
        """
        zero_result = (0.0, {k: 0.0 for k in self.best_of_ks})
        try:
            samples = self.policy.predict(
                context,
                use_posterior=False,
                n_samples=max_k,
                std_scale=self.std_scale,
            )
            # samples: (B, max_k, H, action_dim)
            if samples.dim() != 4:
                return zero_result
            std = samples.std(dim=1).mean().item()
            gt = actions.unsqueeze(1)  # (B, 1, H, action_dim)
            mse_per_k = ((samples - gt) ** 2).mean(dim=(-1, -2))  # (B, max_k)
            best_k_mses = {}
            for k in self.best_of_ks:
                best_mse = mse_per_k[:, :k].min(dim=1).values.mean().item()
                best_k_mses[k] = best_mse
            return std, best_k_mses
        except Exception:
            pass
        return zero_result
