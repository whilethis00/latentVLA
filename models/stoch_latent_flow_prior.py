"""
Stoch-Latent-FlowPrior
──────────────────────
Most expressive variant. Two-level flow model:

  Posterior: q_φ(z | C_t, a, future) = N(μ_q, σ_q^2)  [same as VAE]
  Prior:     ε ~ N(0,I) → flow_prior(ε, C_t) → z       [latent flow]
  Low-level: ε ~ N(0,I) → flow_action(ε, C_t || z) → a [action flow]

Essentially:  noise → z → action  (two-stage flow)

Losses:
  L_action   = flow_matching_loss(a | C_t, z~q)
  L_prior    = flow_matching_loss(z* | C_t)   where z* from posterior
  L_semantic = cosine(pred_future, future_feat)  [optional]

Note: no KL needed — prior is trained to recover the posterior distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow_utils import VelocityMLP, flow_matching_loss, euler_integrate
from .stoch_latent_vae import GaussianEncoder


class StochLatentFlowPrior(nn.Module):
    """
    Args:
        context_dim     : C_t dimension
        action_dim      : per-step action dim
        action_horizon  : H
        z_dim           : latent z dimension
        future_feat_dim : SigLIP future feature dim (0 = disabled)
        prior_weight    : weight on latent prior flow loss
        flow_hidden / depth / steps: shared VelocityMLP config
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        z_dim: int = 128,
        future_feat_dim: int = 768,
        prior_weight: float = 1.0,
        flow_hidden: int = 512,
        flow_depth: int = 4,
        flow_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.z_dim = z_dim
        self.prior_weight = prior_weight
        self.flow_steps = flow_steps
        self.x_dim = action_dim * action_horizon
        self.use_future = future_feat_dim > 0

        # Posterior q(z | C_t, a, future) → Gaussian for training target
        posterior_in = context_dim + action_dim * action_horizon
        if self.use_future:
            posterior_in += future_feat_dim
        self.posterior_enc = GaussianEncoder(posterior_in, z_dim)

        # Prior: latent-space flow p(z | C_t)
        # Takes noisy z and context → predicts velocity in z-space
        self.prior_flow = VelocityMLP(
            x_dim=z_dim,
            cond_dim=context_dim,
            hidden=flow_hidden,
            depth=flow_depth,
        )

        # Low-level: action flow p(a | C_t, z)
        self.action_flow = VelocityMLP(
            x_dim=self.x_dim,
            cond_dim=context_dim + z_dim,
            hidden=flow_hidden,
            depth=flow_depth,
        )

        # Semantic future head
        if self.use_future:
            self.semantic_head = nn.Linear(z_dim, future_feat_dim)
            self.future_feat_dim = future_feat_dim

    def reparameterize(self, mu, logvar):
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    # ── Training ──────────────────────────────────────────────────────────

    def compute_loss(
        self,
        context: torch.Tensor,                  # (B, context_dim) — full state, for action decoder
        actions: torch.Tensor,                  # (B, H, action_dim)
        future_feat: torch.Tensor = None,
        planner_context: torch.Tensor = None,   # (B, context_dim) — planner subset; None = use context
        semantic_weight: float = 0.1,
        prior_weight: float = None,
    ) -> dict:
        B = actions.shape[0]
        prior_weight = prior_weight if prior_weight is not None else self.prior_weight
        _planner = planner_context if planner_context is not None else context

        # 1. Posterior z* (used as training target for prior flow)
        parts = [_planner, actions.reshape(B, -1)]
        if self.use_future and future_feat is not None:
            parts.append(future_feat)
        q_in = torch.cat(parts, -1)
        mu_q, logvar_q = self.posterior_enc(q_in)
        z_star = self.reparameterize(mu_q, logvar_q)

        # 2. Action flow loss (conditioned on z* from posterior) — always uses full context
        action_cond = torch.cat([context, z_star.detach()], dim=-1)
        loss_action = flow_matching_loss(
            self.action_flow, actions.reshape(B, -1), action_cond
        )

        # 3. Latent prior flow loss — conditioned on planner context
        loss_prior = flow_matching_loss(
            self.prior_flow, z_star.detach(), _planner
        )

        # 4. Semantic future loss
        loss_semantic = torch.tensor(0.0, device=context.device)
        if self.use_future and future_feat is not None:
            pred_future = self.semantic_head(z_star)
            loss_semantic = 1.0 - F.cosine_similarity(
                F.normalize(pred_future, dim=-1),
                F.normalize(future_feat, dim=-1),
                dim=-1,
            ).mean()

        total = (
            loss_action
            + prior_weight * loss_prior
            + semantic_weight * loss_semantic
        )
        return {
            "total_loss": total,
            "action_flow_loss": loss_action,
            "prior_flow_loss": loss_prior,
            "semantic_future_loss": loss_semantic,
        }

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        planner_context: torch.Tensor = None,
        use_posterior: bool = False,
        actions_for_posterior: torch.Tensor = None,
        future_feat_for_posterior: torch.Tensor = None,
        steps: int = None,
        n_samples: int = 1,
        std_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Two-stage generation:
          1. ε → z  via prior_flow  (conditioned on planner_context)
          2. ε → a  via action_flow conditioned on (context, z)  [full state]
        """
        steps = steps or self.flow_steps
        B = context.shape[0]
        _planner = planner_context if planner_context is not None else context

        results = []
        for _ in range(n_samples):
            if use_posterior and actions_for_posterior is not None:
                parts = [_planner, actions_for_posterior.reshape(B, -1)]
                if self.use_future and future_feat_for_posterior is not None:
                    parts.append(future_feat_for_posterior)
                q_in = torch.cat(parts, -1)
                mu_q, logvar_q = self.posterior_enc(q_in)
                z = self.reparameterize(mu_q, logvar_q)
            else:
                # Sample z via latent prior flow (conditioned on planner context)
                z = euler_integrate(self.prior_flow, _planner, self.z_dim, steps, std_scale=std_scale)

            # Sample action chunk conditioned on (full context, z)
            action_cond = torch.cat([context, z], dim=-1)
            x = euler_integrate(self.action_flow, action_cond, self.x_dim, steps)
            results.append(x.reshape(B, self.action_horizon, self.action_dim))

        if n_samples == 1:
            return results[0]
        return torch.stack(results, dim=1)

    def forward(self, context, actions=None, future_feat=None, planner_context=None, **kwargs):
        if actions is not None:
            return self.compute_loss(context, actions, future_feat, planner_context=planner_context, **kwargs)
        return self.predict(context, planner_context=planner_context)
