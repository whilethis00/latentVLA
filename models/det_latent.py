"""
Det-Latent
──────────
Deterministic latent high-level interface.

  Posterior (teacher):  z* = f(C_t, a_{t:t+H-1}, [future_img_feat])
  Prior:                z_hat = g(C_t)
  Low-level:            p(a | C_t, z*)  via flow matching  [train]
                        p(a | C_t, z_hat)                  [inference]

Losses:
  L_action  = flow_matching_loss(a | C_t, z*)
  L_prior   = MSE(z_hat, z*.detach())
  L_semantic = cosine_sim(proj(z* or C_t+z*), future_img_feat)  [optional]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow_utils import VelocityMLP, flow_matching_loss, euler_integrate


class PosteriorEncoder(nn.Module):
    """Maps (C_t, flattened actions, [future_feat]) → z*."""

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        z_dim: int = 128,
        future_feat_dim: int = 0,
    ):
        super().__init__()
        in_dim = context_dim + action_dim * action_horizon + future_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(
        self,
        context: torch.Tensor,         # (B, context_dim)
        actions: torch.Tensor,         # (B, H, action_dim)
        future_feat: torch.Tensor = None,  # (B, future_feat_dim)
    ) -> torch.Tensor:
        B = context.shape[0]
        parts = [context, actions.reshape(B, -1)]
        if future_feat is not None:
            parts.append(future_feat)
        x = torch.cat(parts, dim=-1)
        return self.net(x)


class PriorHead(nn.Module):
    """Maps C_t → z_hat (deterministic)."""

    def __init__(self, context_dim: int, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class DetLatent(nn.Module):
    """
    Args:
        context_dim       : C_t dim
        action_dim        : per-step action dim
        action_horizon    : H
        z_dim             : latent z dimension
        future_feat_dim   : SigLIP future image feature dim (0 = disabled)
        flow_hidden / depth / steps: VelocityMLP config
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        z_dim: int = 128,
        future_feat_dim: int = 768,
        flow_hidden: int = 512,
        flow_depth: int = 4,
        flow_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.z_dim = z_dim
        self.flow_steps = flow_steps
        self.x_dim = action_dim * action_horizon
        self.use_future = future_feat_dim > 0

        self.posterior = PosteriorEncoder(
            context_dim, action_dim, action_horizon, z_dim,
            future_feat_dim if self.use_future else 0,
        )
        self.prior = PriorHead(context_dim, z_dim)

        # Low-level: conditioned on (C_t, z)
        self.velocity_net = VelocityMLP(
            x_dim=self.x_dim,
            cond_dim=context_dim + z_dim,
            hidden=flow_hidden,
            depth=flow_depth,
        )

        # Semantic future prediction head
        if self.use_future:
            self.semantic_head = nn.Linear(z_dim, future_feat_dim)
            self.future_feat_dim = future_feat_dim

    # ── Training ──────────────────────────────────────────────────────────

    def compute_loss(
        self,
        context: torch.Tensor,         # (B, context_dim)
        actions: torch.Tensor,         # (B, H, action_dim)
        future_feat: torch.Tensor = None,  # (B, future_feat_dim)
        semantic_weight: float = 0.1,
        prior_weight: float = 1.0,
    ) -> dict:
        # 1. Posterior z*
        z_star = self.posterior(context, actions, future_feat if self.use_future else None)

        # 2. Action flow loss (using teacher z*)
        cond_action = torch.cat([context, z_star.detach()], dim=-1)
        B = actions.shape[0]
        x1 = actions.reshape(B, -1)
        loss_action = flow_matching_loss(self.velocity_net, x1, cond_action)

        # 3. Prior loss
        z_hat = self.prior(context)
        loss_prior = F.mse_loss(z_hat, z_star.detach())

        # 4. Semantic future loss
        loss_semantic = torch.tensor(0.0, device=context.device)
        if self.use_future and future_feat is not None:
            pred_future = self.semantic_head(z_star)
            # Cosine similarity loss (1 - cosine_sim)
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
            "prior_loss": loss_prior,
            "semantic_future_loss": loss_semantic,
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
    ) -> torch.Tensor:
        """
        use_posterior=True: use teacher z* (upper bound).
        use_posterior=False: use prior z_hat (inference mode).
        """
        steps = steps or self.flow_steps
        if use_posterior and actions_for_posterior is not None:
            z = self.posterior(context, actions_for_posterior, future_feat_for_posterior)
        else:
            z = self.prior(context)

        cond = torch.cat([context, z], dim=-1)
        x = euler_integrate(self.velocity_net, cond, self.x_dim, steps)
        return x.reshape(-1, self.action_horizon, self.action_dim)

    def forward(self, context, actions=None, future_feat=None, **kwargs):
        if actions is not None:
            return self.compute_loss(context, actions, future_feat, **kwargs)
        return self.predict(context)
