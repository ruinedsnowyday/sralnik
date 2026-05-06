"""Lightweight latent diffusion (ε-prediction) on the render bottleneck l."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _betas_linear(num_diffusion_timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, num_diffusion_timesteps)


def _alphas(betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = 1.0 - betas
    ab = torch.cumprod(a, dim=0)
    return a, ab


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, *, mlp_mult: int = 2):
        super().__init__()
        self.dim = dim
        hidden = max(dim * mlp_mult, 64)
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(0, half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class ResBlock2d(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        g = min(32, ch)
        if ch < g:
            g = 1
        elif ch % g != 0:
            g = min(8, ch)
        self.gn1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.cond(cond).chunk(2, dim=-1)
        shift = shift[..., None, None]
        scale = scale[..., None, None]
        h = self.gn1(x)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.c1(h)
        h = F.silu(self.gn2(h))
        h = self.c2(h)
        return x + h


class TinyLatentUNet(nn.Module):
    """Minimal conv stack on 8×8 (or encoder) latent maps — not a full UNet pyramid."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.diffusion_channels
        self.in_conv = nn.Conv2d(c, c, 3, padding=1)
        te = cfg.diffusion_time_dim
        self.t_emb = SinusoidalTimeEmbedding(te, mlp_mult=cfg.diffusion_time_mlp_mult)
        cond_in = cfg.deter_dim + cfg.stoch_dim
        self.cond_fc = nn.Linear(cond_in + te, c * 2)
        depth = max(1, cfg.diffusion_depth)
        blocks = []
        for _ in range(depth):
            blocks.append(ResBlock2d(c, c * 2))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, h_state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        cond0 = torch.cat([h_state, z, te], dim=-1)
        cond = self.cond_fc(cond0)
        feat = F.silu(self.in_conv(x))
        for blk in self.blocks:
            feat = blk(feat, cond)
        return self.out(feat)


class LatentDiffusion(nn.Module):
    def __init__(self, cfg: ModelConfig, *, spatial_hw: tuple[int, int]):
        super().__init__()
        self.cfg = cfg
        self.spatial_hw = spatial_hw
        self.unet = TinyLatentUNet(cfg)
        betas = _betas_linear(cfg.diffusion_timesteps)
        self.register_buffer("betas", betas, persistent=False)
        a, ab = _alphas(betas)
        self.register_buffer("sqrt_ab", torch.sqrt(ab), persistent=False)
        self.register_buffer("sqrt_omab", torch.sqrt(1.0 - ab), persistent=False)
        self.register_buffer("omab_over_sqrt", (1.0 - ab) / torch.sqrt(1.0 - ab + 1e-8), persistent=False)

    def training_loss(self, l0: torch.Tensor, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict ε in the forward diffusion process."""

        B, device = l0.shape[0], l0.device
        t = torch.randint(0, self.cfg.diffusion_timesteps, (B,), device=device, dtype=torch.long)
        eps = torch.randn_like(l0)
        sqrt_ab = self.sqrt_ab[t].view(B, 1, 1, 1)
        sqrt_om = self.sqrt_omab[t].view(B, 1, 1, 1)
        x_t = sqrt_ab * l0 + sqrt_om * eps
        eps_hat = self.unet(x_t, t, h.detach(), z.detach())
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, z: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """DDIM-style few-step sampler (ε prediction, simplified)."""

        B, c, h2, w2 = (
            h.shape[0],
            self.cfg.diffusion_channels,
            self.spatial_hw[0],
            self.spatial_hw[1],
        )
        device = h.device
        x = torch.randn(B, c, h2, w2, device=device)
        ts = torch.linspace(self.cfg.diffusion_timesteps - 1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            t_batch = t.expand(B)
            eps = self.unet(x, t_batch, h, z)
            # one-step Euler denoise toward l0 estimate
            sqrt_ab = self.sqrt_ab[t]
            sqrt_om = self.sqrt_omab[t]
            l0_hat = (x - sqrt_om * eps) / (sqrt_ab + 1e-8)
            if i < len(ts) - 1:
                t_next = ts[i + 1]
                a_next = self.sqrt_ab[t_next]
                om_next = self.sqrt_omab[t_next]
                x = a_next * l0_hat + om_next * torch.randn_like(x)
            else:
                x = l0_hat
        return x
