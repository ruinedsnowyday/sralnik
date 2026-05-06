"""Memory read / fusion (conditions M1–M3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryMode, ModelConfig


class MemoryFusion(nn.Module):
    """Reads from a variable-length history of stored latents (keys = values)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.memory_mode
        dz = cfg.stoch_dim
        dh = cfg.deter_dim
        q_in = dh + dz
        self._q = nn.Linear(q_in, dz)

        if self.mode is MemoryMode.CONCAT:
            in_concat = dh + cfg.memory_topk * dz
            self._to_delta = nn.Sequential(
                nn.Linear(in_concat, cfg.gate_hidden),
                nn.SiLU(),
                nn.Linear(cfg.gate_hidden, dh),
            )
        if self.mode in (MemoryMode.ATTENTION, MemoryMode.GATED):
            self._att_in = nn.Linear(dz, dz)
            self._att_out = nn.Linear(dz, dh)
            self._nhead = cfg.memory_heads
            if dz % self._nhead != 0:
                raise ValueError(f"stoch_dim {dz} must divide memory_heads {self._nhead}")

        if self.mode is MemoryMode.GATED:
            self._gate = nn.Sequential(
                nn.Linear(dh + dz, cfg.gate_hidden),
                nn.SiLU(),
                nn.Linear(cfg.gate_hidden, 1),
            )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        hist_z: torch.Tensor | None,
        hist_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """hist_z: (B,L,dz), hist_mask: (B,L) bool True=valid."""

        if (
            self.mode is MemoryMode.NONE
            or hist_z is None
            or hist_z.shape[1] == 0
            or hist_mask is None
        ):
            return h

        row_ok = hist_mask.any(dim=-1)  # (B,)
        if not row_ok.any():
            return h

        B, L, D = hist_z.shape
        scores = torch.matmul(hist_z, self._q(torch.cat([h, z], dim=-1)).unsqueeze(-1)).squeeze(
            -1
        )
        scores = scores.masked_fill(~hist_mask, -1e9)

        if self.mode is MemoryMode.CONCAT:
            k = min(self.cfg.memory_topk, L)
            _, idx = torch.topk(scores, k=k, dim=-1)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
            top_z = torch.gather(hist_z, 1, idx_exp).reshape(B, -1)
            out = h + self._to_delta(torch.cat([h, top_z], dim=-1))
            return torch.where(row_ok.unsqueeze(-1), out, h)

        q = self._q(torch.cat([h, z], dim=-1)).view(B, 1, D)
        k_lin = self._att_in(hist_z)
        v = hist_z
        key_keep = hist_mask
        ctx, _ = F.scaled_dot_product_attention(
            q,
            k_lin,
            v,
            attn_mask=key_keep.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
        )
        ctx = ctx.squeeze(1)
        delta = self._att_out(ctx)

        if self.mode is MemoryMode.ATTENTION:
            out = h + delta
            return torch.where(row_ok.unsqueeze(-1), out, h)

        g = torch.sigmoid(self._gate(torch.cat([h, z], dim=-1)))
        out = h + g * delta
        return torch.where(row_ok.unsqueeze(-1), out, h)
