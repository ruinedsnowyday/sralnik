"""CNN image encoder → spatial features + Gaussian stochastic latent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _make_norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)


class Encoder(nn.Module):
    """Maps RGB (B,3,H,W) to feature map (B,C,h,w) and posterior N(μ,σ² I) over z."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = list(cfg.enc_channels)
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in c:
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                _make_norm(out_ch),
                nn.SiLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # Infer spatial size with dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
            feat = self.conv(dummy)
            _, _, h, w = feat.shape
        self.feat_hw = (h, w)
        flat = c[-1] * h * w
        self.to_post = nn.Sequential(
            nn.Linear(flat, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * cfg.stoch_dim),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs: (B,3,H,W) in [0,1]. Returns feat (B,C,h,w), post_mu, post_std (B,dz)."""

        feat = self.conv(obs)
        B = feat.shape[0]
        flat = feat.reshape(B, -1)
        stats = self.to_post(flat)
        mu, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + 1e-4
        return feat, mu, std
