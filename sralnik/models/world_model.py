"""RSSM world model with optional memory fusion and latent diffusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryMode, ModelConfig
from .decoder import Decoder, RenderBottleneck
from .encoder import Encoder
from .latent_diffusion import LatentDiffusion
from .lpips_loss import LPIPSLoss
from .memory import MemoryFusion
from .sd_vae_decoder import SDVAEDecoder


def _kl_balanced(
    post_mu: torch.Tensor,
    post_std: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_std: torch.Tensor,
    free_bits: float,
    balance: float,
) -> torch.Tensor:
    """Dreamer-style KL with free-bits; stop-grad mix on prior/post grads."""

    post_dist = torch.distributions.Normal(post_mu, post_std)
    prior_dist = torch.distributions.Normal(prior_mu, prior_std)
    kl_sg_p = torch.distributions.kl_divergence(post_dist, torch.distributions.Normal(prior_mu.detach(), prior_std.detach()))
    kl_sg_q = torch.distributions.kl_divergence(torch.distributions.Normal(post_mu.detach(), post_std.detach()), prior_dist)
    kl = balance * kl_sg_p + ( 1.0 - balance) * kl_sg_q
    if free_bits > 0:
        dim = post_mu.shape[-1]
        kl = kl.clamp_min(free_bits / dim)
    return kl.sum(dim=-1)


def _write_mask_from_phase(phase: torch.Tensor) -> torch.Tensor:
    """phase codes 0=A,1=B,2=C -> write on first frame and on A->B transitions."""

    B, T = phase.shape
    m = torch.zeros(B, T, dtype=torch.bool, device=phase.device)
    m[:, 0] = True
    if T > 1:
        m[:, :-1] |= (phase[:, :-1] == 0) & (phase[:, 1:] == 1)
    return m


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        self.encoder = Encoder(self.cfg)
        feat_hw = self.encoder.feat_hw
        self.decoder = Decoder(self.cfg, spatial_hw=feat_hw)
        self.memory = MemoryFusion(self.cfg)
        # Bottleneck + diffusion only exist when latent diffusion is enabled. Creating them
        # unconditionally leaves RenderBottleneck params unused in forward, which makes DDP
        # error out ("Expected to have finished reduction in the prior iteration").
        self.bottleneck: RenderBottleneck | None
        self.diffusion: LatentDiffusion | None
        if self.cfg.use_latent_diffusion:
            self.bottleneck = RenderBottleneck(self.cfg, spatial_hw=feat_hw)
            self.diffusion = LatentDiffusion(self.cfg, spatial_hw=feat_hw)
        else:
            self.bottleneck = None
            self.diffusion = None

        # Optional LPIPS perceptual loss. Frozen VGG backbone, no DDP grad.
        # Tagged as a buffer-style attribute so DDP's find_unused_parameters doesn't
        # complain (no params marked trainable). The wrapper sets requires_grad=False
        # on every internal parameter at init.
        self.lpips: LPIPSLoss | None
        if getattr(self.cfg, "use_lpips", False):
            self.lpips = LPIPSLoss(net="vgg")
        else:
            self.lpips = None

        # Optional v3: replace from-scratch CNN decoder with frozen SD-VAE decoder.
        # When this is on, ``self.decoder`` is still constructed (small param count) but
        # never called in forward; ``self.sd_vae`` produces ``x_hat`` instead. We keep
        # ``self.decoder`` in the module so checkpoints from non-SD-VAE runs can resume
        # cleanly if someone toggles the flag during fine-tuning.
        self.sd_vae: SDVAEDecoder | None
        if getattr(self.cfg, "use_sd_vae", False):
            self.sd_vae = SDVAEDecoder(self.cfg)
        else:
            self.sd_vae = None

        a_dim = 32
        self.act_dim = a_dim
        self.register_parameter("h0", nn.Parameter(torch.zeros(1, self.cfg.deter_dim)))
        self.act_emb = nn.Embedding(self.cfg.action_vocab_size + 1, a_dim)  # +1 padding index 0 = start
        self.gru = nn.GRUCell(self.cfg.stoch_dim + a_dim + 1, self.cfg.deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(self.cfg.deter_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * self.cfg.stoch_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_success: torch.Tensor,
        *,
        write_mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        phase_weight: float = 2.0,
        posterior_sample: bool = True,
        return_reconstructions: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced training pass.

        obs: (B,T,3,H,W) in [0,1]
        actions: (B,T) int64 indices into ACTION_NAMES
        action_success: (B,T) bool
        write_mask: (B,T) optional bool; if None, derived from phase
        phase: (B,T) int {0,1,2} optional for loss weighting + write mask
        """

        B, T, C, H, W = obs.shape
        device = obs.device
        actions = actions.clamp(0, self.cfg.action_vocab_size - 1)

        if write_mask is None:
            if phase is None:
                write_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            else:
                write_mask = _write_mask_from_phase(phase)
        else:
            write_mask = write_mask.bool()

        if phase is None:
            w_phase = torch.ones(B, T, device=device)
        else:
            # Up-weight Phase C (code 2) frames
            w_phase = torch.where(phase == 2, phase_weight, torch.ones_like(phase, dtype=torch.float32))

        h = self.h0.expand(B, -1)
        kl_steps: list[torch.Tensor] = []
        rec_steps: list[torch.Tensor] = []
        diff_steps: list[torch.Tensor] = []

        z_hist: list[torch.Tensor] = []
        rec_frames: list[torch.Tensor] = []

        # v3: when self.sd_vae is on we defer the decode and run it ONCE on the whole
        # (B*T, ...) batch after the causal loop finishes — saves T sequential VAE
        # forwards. Per-step h, z snapshots needed for that batched decode are
        # collected in h_steps / z_steps below.
        h_steps: list[torch.Tensor] = []
        z_steps: list[torch.Tensor] = []
        use_sd_vae = self.sd_vae is not None

        for t in range(T):
            prior_raw = self.prior_net(h)
            prior_mu, prior_logstd = torch.chunk(prior_raw, 2, dim=-1)
            prior_std = F.softplus(prior_logstd) + 1e-4

            _, post_mu, post_std = self.encoder(obs[:, t])
            if posterior_sample:
                z = post_mu + post_std * torch.randn_like(post_std)
            else:
                z = post_mu

            # Build memory context from z_0..z_{t-1}; mask matches write schedule.
            if z_hist and self.cfg.memory_mode is not MemoryMode.NONE:
                hist = torch.stack(z_hist, dim=1)
                mem_mask = write_mask[:, : hist.shape[1]].bool()
                h_mem = self.memory(h, z, hist, mem_mask)
            else:
                h_mem = h

            act_e = self.act_emb(actions[:, t] + 1)
            succ = action_success[:, t].float().unsqueeze(-1)
            gru_in = torch.cat([z, act_e, succ], dim=-1)
            h = self.gru(gru_in, h_mem)

            z_hist.append(z.detach())

            kl = _kl_balanced(
                post_mu,
                post_std,
                prior_mu,
                prior_std,
                free_bits=self.cfg.free_bits,
                balance=self.cfg.kl_balance,
            )
            kl_steps.append(kl)

            if use_sd_vae:
                # Defer decode + rec computation until after the loop (batched).
                h_steps.append(h)
                z_steps.append(z)
            else:
                # Original per-step CNN decode path. Unchanged behaviour for M0–M3.
                x_hat = self.decoder(h, z)
                if return_reconstructions:
                    rec_frames.append(x_hat)
                rec_l1 = F.l1_loss(x_hat, obs[:, t], reduction="none").mean(dim=(1, 2, 3))
                if self.lpips is not None:
                    lp = self.lpips(x_hat, obs[:, t])
                    rec_t = rec_l1 + self.cfg.lpips_weight * lp
                else:
                    rec_t = rec_l1
                rec_steps.append(rec_t * w_phase[:, t])

            if self.diffusion is not None:
                l0 = self.bottleneck(h, z).detach()
                ld = self.diffusion.training_loss(l0, h, z)
                diff_steps.append(ld.expand(B))

        # v3: batched SD-VAE decode + rec losses.
        if use_sd_vae:
            H = torch.stack(h_steps, dim=1)                      # (B, T, dh)
            Z = torch.stack(z_steps, dim=1)                      # (B, T, dz)
            H_flat = H.reshape(B * T, H.shape[-1])
            Z_flat = Z.reshape(B * T, Z.shape[-1])
            x_hat_flat = self.sd_vae(H_flat, Z_flat)             # (B*T, 3, 256, 256)
            x_hat_BT = x_hat_flat.view(B, T, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
            for t in range(T):
                x_hat_t = x_hat_BT[:, t]
                if return_reconstructions:
                    rec_frames.append(x_hat_t)
                rec_l1 = F.l1_loss(x_hat_t, obs[:, t], reduction="none").mean(dim=(1, 2, 3))
                if self.lpips is not None:
                    lp = self.lpips(x_hat_t, obs[:, t])
                    rec_t = rec_l1 + self.cfg.lpips_weight * lp
                else:
                    rec_t = rec_l1
                rec_steps.append(rec_t * w_phase[:, t])

        out = {
            "loss_kl": torch.stack(kl_steps, dim=1).mean(),
            "loss_rec": torch.stack(rec_steps, dim=1).mean(),
        }
        if diff_steps:
            out["loss_diff"] = torch.stack(diff_steps, dim=1).mean()
            out["loss_total"] = (
                out["loss_rec"] + out["loss_kl"] + self.cfg.diffusion_loss_weight * out["loss_diff"]
            )
        else:
            out["loss_total"] = out["loss_rec"] + out["loss_kl"]
        if return_reconstructions and rec_frames:
            out["x_hat"] = torch.stack(rec_frames, dim=1)
        return out
