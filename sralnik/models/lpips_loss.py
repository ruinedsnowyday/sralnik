"""LPIPS perceptual loss wrapper (lazy import).

LPIPS punishes image dissimilarity in VGG feature space rather than pixel space,
so it penalises blur in a way pure L1/L2 cannot. Adding it as an auxiliary
reconstruction term is the cheapest known intervention for sharper outputs from
small CNN decoders.

Used opt-in via ``ModelConfig.use_lpips``. The 14M-param VGG backbone is created
once per model, frozen, and put in eval mode so it neither receives gradient nor
updates batchnorm stats.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    """Frozen LPIPS metric used as a training-time perceptual loss."""

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as e:  # pragma: no cover - import-time error path
            raise ImportError(
                "lpips package not installed. Add 'lpips>=0.1.4' to pyproject.toml "
                "and re-sync the env."
            ) from e

        # spatial=False -> single scalar per image; verbose=False suppresses the
        # `Setting up [LPIPS]` print at every model construction.
        self._lpips = lpips.LPIPS(net=net, verbose=False, spatial=False)
        for p in self._lpips.parameters():
            p.requires_grad_(False)
        self._lpips.eval()

    def train(self, mode: bool = True) -> "LPIPSLoss":
        # Always keep the LPIPS backbone in eval mode (frozen batchnorm stats),
        # regardless of what nn.Module.train() the parent calls down the tree.
        super().train(mode)
        self._lpips.eval()
        return self

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-frame perceptual distance.

        x_hat, x: (B, 3, H, W) tensors in [0, 1].
        Returns: shape (B,) per-batch-row LPIPS distance, so caller can apply
        per-sample weighting (e.g. phase upweighting).
        """
        # LPIPS expects inputs in [-1, 1]. Our decoder produces sigmoid outputs
        # in [0, 1] and the dataset normalises RGB to [0, 1] already.
        x_hat_n = x_hat.mul(2.0).sub(1.0)
        x_n = x.mul(2.0).sub(1.0)
        # spatial=False produces shape (B, 1, 1, 1); flatten to (B,).
        return self._lpips(x_hat_n, x_n).view(x_hat.shape[0])
