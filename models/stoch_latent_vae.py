"""
Stoch-Latent-VAE
────────────────
Stochastic latent via Gaussian VAE.

  Posterior: q_φ(z | C_t, a_{t:t+H-1}, future) = N(μ_q, σ_q^2)
  Prior:     p_θ(z | C_t)                       = N(μ_p, σ_p^2)
  Low-level: p_ψ(a | C_t, z)  via flow matching

Losses:
  L_action   = flow_matching_loss(a | C_t, z~q)
  L_KL       = KL(q || p)
  L_semantic = cosine(pred_future, future_feat)  [optional]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow_utils import VelocityMLP, flow_matching_loss, euler_integrate


def kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """KL(N(mu_q, exp(logvar_q)) || N(mu_p, exp(logvar_p)))"""
    return 0.5 * (
        logvar_p - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1
    ).sum(-1).mean()


class GaussianEncoder(nn.Module):
    """Generic encoder → (mu, logvar) of shape (B, z_dim)."""

    def __init__(self, in_dim: int, z_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 256),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h).clamp(-10, 4)


class StochLatentVAE(nn.Module):
    """
    Args:
        context_dim     : C_t dimension
        action_dim      : per-step action dim
        action_horizon  : H
        z_dim           : latent z dimension
        future_feat_dim : SigLIP future feature dim (0 = disabled)
        kl_beta         : KL weight (β-VAE style)
        flow_hidden / depth / steps: VelocityMLP config
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        z_dim: int = 128,
        future_feat_dim: int = 768,
        kl_beta: float = 1.0,
        flow_hidden: int = 512,
        flow_depth: int = 4,
        flow_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.z_dim = z_dim
        self.kl_beta = kl_beta
        self.flow_steps = flow_steps
        self.x_dim = action_dim * action_horizon
        self.use_future = future_feat_dim > 0

        # Posterior q(z|C_t, a, future)
        posterior_in = context_dim + action_dim * action_horizon
        if self.use_future:
            posterior_in += future_feat_dim
        self.posterior_enc = GaussianEncoder(posterior_in, z_dim)

        # Prior p(z|C_t)
        self.prior_enc = GaussianEncoder(context_dim, z_dim)

        # Low-level action flow (conditioned on C_t || z)
        self.velocity_net = VelocityMLP(
            x_dim=self.x_dim,
            cond_dim=context_dim + z_dim,
            hidden=flow_hidden,
            depth=flow_depth,
        )

        # Semantic future head
        if self.use_future:
            self.semantic_head = nn.Linear(z_dim, future_feat_dim)
            self.future_feat_dim = future_feat_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── Training ──────────────────────────────────────────────────────────

    def compute_loss(
        self,
        context: torch.Tensor,         # (B, context_dim)
        actions: torch.Tensor,         # (B, H, action_dim)
        future_feat: torch.Tensor = None,
        semantic_weight: float = 0.1,
        kl_weight: float = None,
    ) -> dict:
        B = actions.shape[0]
        kl_weight = kl_weight if kl_weight is not None else self.kl_beta

        # Posterior
        parts = [context, actions.reshape(B, -1)]
        if self.use_future and future_feat is not None:
            parts.append(future_feat)
        q_in = torch.cat(parts, dim=-1)
        mu_q, logvar_q = self.posterior_enc(q_in)
        z = self.reparameterize(mu_q, logvar_q)

        # Prior
        mu_p, logvar_p = self.prior_enc(context)

        # Action flow loss (z from posterior)
        cond = torch.cat([context, z], dim=-1)
        x1 = actions.reshape(B, -1)
        loss_action = flow_matching_loss(self.velocity_net, x1, cond)

        # KL loss
        loss_kl = kl_gaussian(mu_q, logvar_q, mu_p, logvar_p)

        # Semantic future loss
        loss_semantic = torch.tensor(0.0, device=context.device)
        if self.use_future and future_feat is not None:
            pred_future = self.semantic_head(z)
            loss_semantic = 1.0 - F.cosine_similarity(
                F.normalize(pred_future, dim=-1),
                F.normalize(future_feat, dim=-1),
                dim=-1,
            ).mean()

        total = loss_action + kl_weight * loss_kl + semantic_weight * loss_semantic
        return {
            "total_loss": total,
            "action_flow_loss": loss_action,
            "kl_loss": loss_kl,
            "semantic_future_loss": loss_semantic,
            # For logging: prior/posterior gap analysis
            "z_mu_q": mu_q.detach(),
            "z_mu_p": mu_p.detach(),
        }

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        use_posterior: bool = False,
        actions_for_posterior: torch.Tensor = None,
        future_feat_for_posterior: torch.Tensor = None,
        steps: int = None,
        n_samples: int = 1,
        std_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        n_samples > 1: return (B, n_samples, H, action_dim) for diversity eval.
        """
        steps = steps or self.flow_steps
        B = context.shape[0]

        if use_posterior and actions_for_posterior is not None:
            parts = [context, actions_for_posterior.reshape(B, -1)]
            if self.use_future and future_feat_for_posterior is not None:
                parts.append(future_feat_for_posterior)
            q_in = torch.cat(parts, -1)
            mu, logvar = self.posterior_enc(q_in)
        else:
            mu, logvar = self.prior_enc(context)

        results = []
        for _ in range(n_samples):
            std = (0.5 * logvar).exp() * std_scale
            z = mu + std * torch.randn_like(mu)
            cond = torch.cat([context, z], dim=-1)
            x = euler_integrate(self.velocity_net, cond, self.x_dim, steps)
            results.append(x.reshape(B, self.action_horizon, self.action_dim))

        if n_samples == 1:
            return results[0]
        return torch.stack(results, dim=1)  # (B, n_samples, H, action_dim)

    def forward(self, context, actions=None, future_feat=None, **kwargs):
        if actions is not None:
            return self.compute_loss(context, actions, future_feat, **kwargs)
        return self.predict(context)
