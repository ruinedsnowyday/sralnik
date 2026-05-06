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
        # _q is the query projection used as the SDPA query in ATTENTION/GATED.
        # NOT used in CONCAT: the original "score + topk + concat" plan relied on
        # torch.topk for selection, but topk's index output is non-differentiable,
        # so _q would never receive gradient and DDP errors with
        # "Expected to have finished reduction in the prior iteration".
        # CONCAT now uses last-k retrieval (no scoring) instead.
        if self.mode in (MemoryMode.ATTENTION, MemoryMode.GATED):
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

        if self.mode is MemoryMode.CONCAT:
            # Last-k retrieval: take the k most recent valid history entries (in
            # episode order). No scoring -> no _q -> no DDP unused-grad error.
            # Equivalent to the original spec when memory_topk >= L; differs only
            # in how we sub-sample when L > memory_topk (recency vs scored top-k).
            k = min(self.cfg.memory_topk, L)
            top_z = hist_z[:, -k:, :].reshape(B, -1)
            pad_to = self.cfg.memory_topk * D
            if top_z.shape[1] < pad_to:
                top_z = F.pad(top_z, (0, pad_to - top_z.shape[1]))
            out = h + self._to_delta(torch.cat([h, top_z], dim=-1))
            return torch.where(row_ok.unsqueeze(-1), out, h)

        # ATTENTION / GATED: cross-attention via SDPA with a learned query.
        q = self._q(torch.cat([h, z], dim=-1)).view(B, 1, D)
        key_keep = hist_mask
        k_lin = self._att_in(hist_z)
        v = hist_z
        # F.scaled_dot_product_attention returns a single Tensor in PyTorch 2.x.
        # The previous `ctx, _ = ...` destructured along dim 0 (batch), silently
        # picking only row 0 with B=2 and crashing with B>=3. ATTENTION/GATED
        # never worked at training-time batch=4 because of this.
        ctx = F.scaled_dot_product_attention(
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
