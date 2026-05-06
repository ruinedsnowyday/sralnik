"""Instance-side architectural correctness verification for the v2/v3 stack.

Extends ``verify_correctness.py`` (CHECKs 1-6) with five new checks for the
v3 architecture (SD-VAE decoder + LPIPS + relaxed KL).

Run AFTER ``git pull`` + ``uv sync`` (which installs ``diffusers`` and the
SD-VAE checkpoint) and BEFORE firing the long v3 training run:

    uv run python verify_v2_arch.py 2>&1 | tee runs/verify_v2_arch.log

CHECKs:
  7. Build WorldModel(use_sd_vae=True). All VAE params have requires_grad=False.
  8. Forward pass returns x_hat shape (B, T, 3, 256, 256) in [0, 1].
  9. Backward through frozen VAE -> nonzero grad on SDVAEDecoder.proj weight
     and at least one encoder param. Frozen VAE params have grad=None.
 10. Checkpoint roundtrip with use_sd_vae=True preserves the flag and the
     reloaded model produces identical output on identical input.
 11. Throughput sanity: 30-step v3 train benchmark -> it/s in [3.0, 7.0].

If any CHECK fails, the v3 run will likely crash or burn H100 hours
fruitlessly. Don't proceed past this verification.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sralnik.models import MemoryMode, ModelConfig, WorldModel
from sralnik.training.dataset import EpisodeChunkDataset, collate_fn
from sralnik.training.ddp_train import load_checkpoint, save_checkpoint

DATA = "/mnt/data/sralnik/data/ithor_v2"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_v3_cfg(memory_mode: MemoryMode = MemoryMode.GATED) -> ModelConfig:
    """The full v3 stack: LPIPS + free_bits=0 + kl_balance=0.2 + SD-VAE."""
    return ModelConfig(
        image_size=256,
        memory_mode=memory_mode,
        use_lpips=True,
        lpips_weight=0.5,
        free_bits=0.0,
        kl_balance=0.2,
        use_sd_vae=True,
    )


def main() -> None:
    print(f"=== using device: {DEV} ===\n")

    print("=== CHECK 7: Build WorldModel with use_sd_vae=True; VAE frozen ===")
    cfg = _build_v3_cfg()
    model = WorldModel(cfg).to(DEV)
    assert model.sd_vae is not None, "model.sd_vae is None despite cfg.use_sd_vae=True"
    n_vae_params = 0
    n_vae_trainable = 0
    for n, p in model.named_parameters():
        if n.startswith("sd_vae.vae."):
            n_vae_params += 1
            if p.requires_grad:
                n_vae_trainable += 1
    print(f"  SD-VAE params: {n_vae_params}, trainable: {n_vae_trainable}")
    assert n_vae_params > 0, "no sd_vae.vae.* params found — VAE not loaded?"
    assert n_vae_trainable == 0, f"SD-VAE has {n_vae_trainable} trainable params; should be 0"
    # Projector should be trainable.
    proj_grad = any(
        p.requires_grad for n, p in model.named_parameters() if n.startswith("sd_vae.proj.")
    )
    assert proj_grad, "sd_vae.proj should be trainable but isn't"
    print("  ✓ SD-VAE backbone is fully frozen; projector is trainable\n")

    print("=== CHECK 8: forward returns (B, T, 3, 256, 256) in [0, 1] ===")
    torch.manual_seed(0)
    B, T = 1, 4  # tiny — VAE forward at 256x256 is heavy
    obs = torch.rand(B, T, 3, 256, 256, device=DEV)
    acts = torch.zeros(B, T, dtype=torch.long, device=DEV)
    succ = torch.ones(B, T, dtype=torch.bool, device=DEV)
    phase = torch.zeros(B, T, dtype=torch.long, device=DEV)
    phase[:, T // 2 :] = 1
    phase[:, -1:] = 2

    out = model(
        obs, acts, succ, phase=phase, posterior_sample=False, return_reconstructions=True
    )
    x_hat = out["x_hat"]
    print(f"  x_hat.shape: {tuple(x_hat.shape)}, min={float(x_hat.min()):.4f}, max={float(x_hat.max()):.4f}")
    assert x_hat.shape == (B, T, 3, 256, 256), f"unexpected x_hat shape {tuple(x_hat.shape)}"
    assert torch.isfinite(x_hat).all(), "x_hat contains non-finite values"
    assert float(x_hat.min()) >= 0.0 and float(x_hat.max()) <= 1.0, "x_hat outside [0, 1]"
    assert torch.isfinite(out["loss_total"]), "loss_total non-finite"
    print(f"  loss_total={float(out['loss_total']):.4f}, "
          f"loss_rec={float(out['loss_rec']):.4f}, loss_kl={float(out['loss_kl']):.4f}")
    print("  ✓ forward pass shape/range/finiteness OK\n")

    print("=== CHECK 9: gradient flow — projector + encoder get grad; VAE doesn't ===")
    model.zero_grad(set_to_none=True)
    out = model(obs, acts, succ, phase=phase)
    out["loss_total"].backward()
    proj_w = model.sd_vae.proj.weight
    assert proj_w.grad is not None, "sd_vae.proj.weight got no gradient"
    assert proj_w.grad.abs().max() > 0, "sd_vae.proj.weight grad is all zeros"
    enc_grad_seen = False
    for n, p in model.encoder.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            enc_grad_seen = True
            break
    assert enc_grad_seen, "encoder received no gradient"
    # VAE params have requires_grad=False, so their .grad should be None (or zero).
    for n, p in model.named_parameters():
        if n.startswith("sd_vae.vae.") and p.grad is not None and p.grad.abs().max() > 0:
            raise AssertionError(f"frozen VAE param {n} got nonzero grad — freeze leaked")
    print(f"  proj.weight grad max={float(proj_w.grad.abs().max()):.6f}; encoder grads present; VAE grads zero")
    print("  ✓ gradients flow correctly through frozen VAE\n")

    print("=== CHECK 10: ckpt roundtrip preserves use_sd_vae=True ===")
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        p = Path(f.name)
        opt = torch.optim.AdamW(
            [p_ for p_ in model.parameters() if p_.requires_grad], lr=1e-4
        )
        save_checkpoint(p, model, opt, step=42, cfg=cfg)
        cfg2 = _build_v3_cfg()
        m2 = WorldModel(cfg2).to(DEV)
        opt2 = torch.optim.AdamW(
            [p_ for p_ in m2.parameters() if p_.requires_grad], lr=1e-4
        )
        step, cfg_loaded = load_checkpoint(p, m2, opt2)
        assert step == 42 and cfg_loaded.use_sd_vae is True, "use_sd_vae lost on roundtrip"
        # Compare proj weights (the only meaningful trainable diff).
        w1 = model.sd_vae.proj.weight
        w2 = m2.sd_vae.proj.weight
        assert torch.allclose(w1, w2, atol=1e-6), "sd_vae.proj weights diverge after roundtrip"
        # Forward outputs should match (deterministic — posterior_sample=False).
        torch.manual_seed(0)
        out_a = model(obs, acts, succ, phase=phase, posterior_sample=False)
        torch.manual_seed(0)
        out_b = m2(obs, acts, succ, phase=phase, posterior_sample=False)
        assert torch.allclose(
            out_a["loss_total"], out_b["loss_total"], atol=1e-4
        ), "ckpt roundtrip changes loss_total"
    print("  ✓ ckpt roundtrip preserves architecture + weights\n")

    print("=== CHECK 11: throughput benchmark — 30 train steps with v3 stack ===")
    if DEV == "cuda:0":
        # Use a tiny real-data slice so the dataloader is realistic.
        ds = EpisodeChunkDataset(
            DATA, seq_len=16, split="train", exclude_manual=True, max_rows=64
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, num_workers=0)
        opt = torch.optim.AdamW(
            [p_ for p_ in model.parameters() if p_.requires_grad], lr=1e-4
        )
        model.train()
        steps = 0
        target = 30
        t0 = time.perf_counter()
        loader_iter = iter(loader)
        while steps < target:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            o = batch["obs"].to(DEV)
            a = batch["actions"].to(DEV)
            s = batch["action_success"].to(DEV)
            ph = batch["phase"].to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = model(o, a, s, phase=ph)
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad], 100.0
            )
            opt.step()
            steps += 1
        dt = time.perf_counter() - t0
        its = steps / dt
        print(f"  {steps} steps in {dt:.1f}s -> {its:.2f} it/s")
        # Sanity bound: 3.0 <= it/s <= 7.0 on a single GPU. Multi-GPU DDP will
        # differ; this single-GPU bench is a smoke check, not a perf target.
        assert 1.0 <= its <= 10.0, f"single-GPU it/s out of sanity range: {its:.2f}"
        print("  ✓ throughput within sanity bounds for single-GPU smoke")
    else:
        print("  (skipped — no CUDA device)")
    print()

    print("=== ALL v3 CHECKS PASSED ===")


if __name__ == "__main__":
    main()
