"""
Flow Matching Utilities
───────────────────────
OT-CFM (Optimal Transport Conditional Flow Matching)

Forward (training):
  x0 ~ N(0, I)  — noise
  x1            — data (action chunk or latent z)
  t  ~ U[0, 1]
  xt = (1-t)*x0 + t*x1
  ut = x1 - x0  — target velocity

Loss:
  L = ||v_theta(xt, t, cond) - ut||^2

Inference:
  Euler ODE from t=0 to t=1:
  x_{t+dt} = x_t + v_theta(x_t, t, cond) * dt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0,1] → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float()
            * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None]  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ── Velocity network (MLP-based) ─────────────────────────────────────────────

class VelocityMLP(nn.Module):
    """
    Predicts velocity v(x_t, t, cond) for flow matching.

    Args:
        x_dim    : dimensionality of x (e.g. action_horizon * action_dim)
        cond_dim : conditioning vector dimension (C_t or C_t || z)
        hidden   : hidden dimension
        depth    : number of residual blocks
        time_dim : sinusoidal time embedding dim
    """

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        hidden: int = 512,
        depth: int = 4,
        time_dim: int = 128,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        in_dim = x_dim + time_dim + cond_dim
        self.input_proj = nn.Linear(in_dim, hidden)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden) for _ in range(depth)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, x_dim),
        )

    def forward(
        self,
        x: torch.Tensor,    # (B, x_dim)
        t: torch.Tensor,    # (B,) in [0,1]
        cond: torch.Tensor, # (B, cond_dim)
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)           # (B, time_dim)
        h = torch.cat([x, t_emb, cond], -1)  # (B, x_dim+time_dim+cond_dim)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)           # (B, x_dim)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


# ── Flow matching loss ────────────────────────────────────────────────────────

def flow_matching_loss(
    velocity_net: nn.Module,
    x1: torch.Tensor,      # (B, x_dim) — target data
    cond: torch.Tensor,    # (B, cond_dim)
    t_min: float = 1e-4,
    t_max: float = 1.0 - 1e-4,
) -> torch.Tensor:
    """OT-CFM loss. x1 is data (actions or latent z)."""
    B, x_dim = x1.shape
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device) * (t_max - t_min) + t_min
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    ut = x1 - x0  # target velocity
    vt = velocity_net(xt, t, cond)
    return F.mse_loss(vt, ut)


# ── ODE integration (inference) ──────────────────────────────────────────────

@torch.no_grad()
def euler_integrate(
    velocity_net: nn.Module,
    cond: torch.Tensor,    # (B, cond_dim)
    x_dim: int,
    steps: int = 10,
    device: torch.device = None,
    std_scale: float = 1.0,
) -> torch.Tensor:
    """Euler ODE from noise (t=0) to data (t=1). Returns (B, x_dim)."""
    if device is None:
        device = cond.device
    B = cond.shape[0]
    x = torch.randn(B, x_dim, device=device) * std_scale
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        t = torch.full((B,), t_val, device=device)
        v = velocity_net(x, t, cond)
        x = x + v * dt
    return x
