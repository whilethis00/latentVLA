"""
Flat-Flow
─────────
Baseline: no hierarchy.
  p(a_{t:t+H-1} | C_t)  via flow matching.

No latent z. Direct action generation conditioned on context.
"""

import torch
import torch.nn as nn
from .flow_utils import VelocityMLP, flow_matching_loss, euler_integrate


class FlatFlow(nn.Module):
    """
    Args:
        context_dim  : C_t dimension
        action_dim   : per-timestep action dim
        action_horizon: H
        flow_hidden  : VelocityMLP hidden dim
        flow_depth   : VelocityMLP residual block count
        flow_steps   : Euler ODE steps at inference
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        flow_hidden: int = 512,
        flow_depth: int = 4,
        flow_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flow_steps = flow_steps
        self.x_dim = action_dim * action_horizon

        self.velocity_net = VelocityMLP(
            x_dim=self.x_dim,
            cond_dim=context_dim,
            hidden=flow_hidden,
            depth=flow_depth,
        )

    # ── Training ──────────────────────────────────────────────────────────

    def compute_loss(
        self,
        context: torch.Tensor,  # (B, context_dim)
        actions: torch.Tensor,  # (B, H, action_dim)
    ) -> dict:
        B = actions.shape[0]
        x1 = actions.reshape(B, -1)  # (B, x_dim)
        loss = flow_matching_loss(self.velocity_net, x1, context)
        return {"action_flow_loss": loss, "total_loss": loss}

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,  # (B, context_dim)
        steps: int = None,
    ) -> torch.Tensor:
        """Returns predicted action chunk (B, H, action_dim)."""
        steps = steps or self.flow_steps
        x = euler_integrate(self.velocity_net, context, self.x_dim, steps)
        return x.reshape(-1, self.action_horizon, self.action_dim)

    def forward(self, context, actions=None):
        """
        Train mode: pass actions → returns loss dict.
        Eval mode: pass context only → returns predicted actions.
        """
        if actions is not None:
            return self.compute_loss(context, actions)
        return self.predict(context)
