"""Hyper-parameters for the SRALNIK world model (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryMode(str, Enum):
    """Memory augmentation condition (M0–M3)."""

    NONE = "none"  # M0
    CONCAT = "concat"  # M1
    ATTENTION = "attention"  # M2
    GATED = "gated"  # M3


@dataclass
class ModelConfig:
    """Defaults sized for 256×256 RGB and the THOR discrete action space."""

    image_size: int = 256
    action_vocab_size: int = 17  # len(ACTION_NAMES) in sralnik.data.actions

    # Latent sizes
    deter_dim: int = 256
    stoch_dim: int = 32

    # Encoder CNN (output spatial 8×8 for 256 input with stride-32)
    enc_channels: tuple[int, ...] = (32, 64, 128, 256, 256)

    # Memory
    memory_mode: MemoryMode = MemoryMode.NONE
    memory_heads: int = 4
    memory_topk: int = 4  # M1 concat top-k
    gate_hidden: int = 128

    # Training
    free_bits: float = 1.0  # nats per dim for KL clamp (0 = off)
    kl_balance: float = 0.8  # Dreamer-style: α for postgrad on prior, (1-α) on post

    # Decoder / diffusion (low-res grid is encoder ``feat_hw``).
    # Defaults favor **fast** latent diffusion; quality trade-off vs heavy l-UNet.
    use_latent_diffusion: bool = False
    render_channels: int = 64
    diffusion_channels: int = 32
    diffusion_depth: int = 1  # number of residual blocks (1 = minimal)
    diffusion_timesteps: int = 32  # training T; small is OK in 8×8 latent space
    diffusion_time_dim: int = 32  # sinusoidal + MLP width (keep small)
    diffusion_time_mlp_mult: int = 2  # hidden = dim * mult (was 4 in heavy variant)
    diffusion_loss_weight: float = 0.06  # λ on ε-loss; lower when step budget is small

    def to_checkpoint_dict(self) -> dict:
        return {
            "image_size": self.image_size,
            "action_vocab_size": self.action_vocab_size,
            "deter_dim": self.deter_dim,
            "stoch_dim": self.stoch_dim,
            "enc_channels": list(self.enc_channels),
            "memory_mode": self.memory_mode.value,
            "memory_heads": self.memory_heads,
            "memory_topk": self.memory_topk,
            "gate_hidden": self.gate_hidden,
            "free_bits": self.free_bits,
            "kl_balance": self.kl_balance,
            "use_latent_diffusion": self.use_latent_diffusion,
            "render_channels": self.render_channels,
            "diffusion_channels": self.diffusion_channels,
            "diffusion_depth": self.diffusion_depth,
            "diffusion_timesteps": self.diffusion_timesteps,
            "diffusion_time_dim": self.diffusion_time_dim,
            "diffusion_time_mlp_mult": self.diffusion_time_mlp_mult,
            "diffusion_loss_weight": self.diffusion_loss_weight,
        }

    @staticmethod
    def from_checkpoint_dict(d: dict) -> "ModelConfig":
        dd = dict(d)
        dd["memory_mode"] = MemoryMode(dd["memory_mode"])
        dd["enc_channels"] = tuple(dd["enc_channels"])
        return ModelConfig(**dd)
