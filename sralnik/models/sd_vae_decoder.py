"""Stable Diffusion VAE decoder as the world model's renderer (v3 architecture).

Replaces the from-scratch CNN ``Decoder`` with the frozen pretrained SD-VAE
decoder from ``stabilityai/sd-vae-ft-mse``. The world model predicts an
SD-VAE latent ``(4, 32, 32)`` (the VAE's native shape for 256×256 inputs);
the VAE decodes it to a 256×256 RGB image, leveraging natural-image priors
that no from-scratch CNN at this scale can reach.

Used opt-in via ``ModelConfig.use_sd_vae``. The VAE itself is frozen and
held in eval mode at all times — only the small ``proj`` Linear (~1.2M
params) projecting ``(h, z) → SD-VAE-latent`` is trainable. Gradients flow
through the frozen VAE during backward (chain rule), but VAE weights do not
update.

Output convention: ``[0, 1]`` (sigmoid-equivalent), matching the rest of the
codebase (encoder input, L1/LPIPS targets, etc.). SD-VAE natively outputs
``[-1, 1]``; we shift+scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class SDVAEDecoder(nn.Module):
    """``(h, z) → SD-VAE latent → frozen VAE decoder → RGB``.

    Structure:
        proj  : Linear(deter_dim + stoch_dim → 4 * 32 * 32 = 4096)  -- trainable
        vae   : diffusers.AutoencoderKL                              -- frozen
        scale : SD-VAE scaling factor (~0.18215)                     -- buffer

    The VAE is frozen at construction; this module's ``train()`` always
    keeps the VAE in eval mode regardless of what the parent calls.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "diffusers package not installed. Add 'diffusers>=0.27' to "
                "pyproject.toml and re-sync the env."
            ) from e

        self.cfg = cfg
        # SD-VAE is hard-coded for 256→32 (8x downscale, 4 latent channels).
        # We don't support image_size != 256 in the v3 path.
        if cfg.image_size != 256:
            raise ValueError(
                f"SDVAEDecoder requires cfg.image_size=256 (got {cfg.image_size}); "
                "the SD-VAE was trained on 8x-downscaled inputs."
            )

        # VAE: load and freeze.
        vae = AutoencoderKL.from_pretrained(cfg.sd_vae_model_id)
        for p in vae.parameters():
            p.requires_grad_(False)
        vae.eval()
        self.vae = vae

        # Projector from world-model state to SD-VAE latent space (B, 4, 32, 32).
        # 4 latent channels × 32 × 32 = 4096 floats — same total dim as the existing
        # Decoder's first FC, so we don't increase world-model expressive capacity;
        # the fidelity gain comes from the VAE's pretrained natural-image prior.
        in_dim = cfg.deter_dim + cfg.stoch_dim
        self._latent_c = 4
        self._latent_hw = 32
        self.proj = nn.Linear(in_dim, self._latent_c * self._latent_hw * self._latent_hw)

        # SD-VAE scale factor: the VAE's internal latent is ~unit-variance after
        # division by 0.18215. We follow the standard SD usage (divide before decode).
        self.register_buffer(
            "scale_factor",
            torch.tensor(0.18215, dtype=torch.float32),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "SDVAEDecoder":
        # Always keep the VAE in eval mode (frozen BN stats, no dropout).
        super().train(mode)
        self.vae.eval()
        return self

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``(h, z) → RGB``.

        Args:
            h: (N, deter_dim) deterministic recurrent state.
            z: (N, stoch_dim) stochastic latent.

        Returns:
            x_hat: (N, 3, 256, 256) in [0, 1].
        """
        # Project to SD-VAE latent.
        flat = self.proj(torch.cat([h, z], dim=-1))                  # (N, 4096)
        latent = flat.view(-1, self._latent_c, self._latent_hw, self._latent_hw)
        latent = latent / self.scale_factor                          # SD-VAE convention

        # Decode through the frozen VAE. The VAE is bf16/fp32 sensitive; we let
        # autocast handle dtype. .sample on the DecoderOutput pulls the tensor.
        rgb_signed = self.vae.decode(latent).sample                  # (N, 3, 256, 256) in [-1, 1]

        # SD-VAE outputs are signed; shift to [0, 1] to match the rest of the pipeline.
        rgb = rgb_signed.add(1.0).mul(0.5).clamp(0.0, 1.0)
        return rgb
