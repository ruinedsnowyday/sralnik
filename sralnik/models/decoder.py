"""Decode deterministic state + stochastic latent to RGB (and optional render bottleneck)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Upsample from concatenated (h, z) to RGB.

    Two upsampling backbones:
      * Default: ConvTranspose2d 2x per stage (matches M0-M3 ablation).
      * ``cfg.use_pixel_shuffle=True``: Conv(c→4c) + PixelShuffle(2). Sub-pixel
        convolution avoids ConvTranspose's checkerboard artefacts and tends to
        produce visibly cleaner edges at the same parameter count.
    """

    def __init__(self, cfg, *, spatial_hw: tuple[int, int]):
        super().__init__()
        self.cfg = cfg
        h, w = spatial_hw
        c = cfg.render_channels
        in_dim = cfg.deter_dim + cfg.stoch_dim
        self.fc = nn.Linear(in_dim, c * h * w)
        self.h, self.w = h, w
        self.c = c
        out = cfg.image_size
        blocks: list[nn.Module] = []
        cur = h
        while cur < out:
            if getattr(cfg, "use_pixel_shuffle", False):
                blocks += [
                    nn.Conv2d(c, c * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.GroupNorm(min(32, c), c),
                    nn.SiLU(),
                ]
            else:
                blocks += [
                    nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),
                    nn.GroupNorm(min(32, c), c),
                    nn.SiLU(),
                ]
            cur *= 2
        self.deconv = nn.Sequential(*blocks)
        self.to_rgb = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns (B,3,H,W) in [0,1] via sigmoid."""

        x = torch.cat([h, z], dim=-1)
        x = self.fc(x).view(-1, self.c, self.h, self.w)
        x = self.deconv(x)
        H, W = x.shape[-2:]
        tgt = self.cfg.image_size
        if H != tgt or W != tgt:
            x = torch.nn.functional.interpolate(
                x, size=(tgt, tgt), mode="bilinear", align_corners=False
            )
        return torch.sigmoid(self.to_rgb(x))


class RenderBottleneck(nn.Module):
    """Maps (h,z) to spatial tensor l used as the generative bottleneck for latent diffusion."""

    def __init__(self, cfg, *, spatial_hw: tuple[int, int]):
        super().__init__()
        self.cfg = cfg
        h2, w2 = spatial_hw
        c = cfg.diffusion_channels
        self.h2, self.w2 = h2, w2
        self.c = c
        self.fc = nn.Linear(cfg.deter_dim + cfg.stoch_dim, c * h2 * w2)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x).view(-1, self.c, self.h2, self.w2)
