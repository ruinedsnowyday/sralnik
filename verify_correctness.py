"""Pre-flight correctness checks before burning H100 hours on the long ablation.

Run on the instance after `uv sync` and the data merge:
    uv run python verify_correctness.py 2>&1 | tee runs/verify_correctness.log

If any assertion fails, do NOT start M0. Diagnose via the workflow in the run plan.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sralnik.models import MemoryMode, ModelConfig, WorldModel
from sralnik.training.dataset import EpisodeChunkDataset, collate_fn
from sralnik.training.ddp_train import load_checkpoint, save_checkpoint

DATA = "/mnt/data/sralnik/data/ithor_v2"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print(f"=== using device: {DEV} ===\n")

    print("=== CHECK 1: All 4 memory modes give finite, *different* outputs ===")
    torch.manual_seed(0)
    B, T = 2, 16
    obs = torch.rand(B, T, 3, 64, 64, device=DEV)
    acts = torch.zeros(B, T, dtype=torch.long, device=DEV)
    succ = torch.ones(B, T, dtype=torch.bool, device=DEV)
    phase = torch.zeros(B, T, dtype=torch.long, device=DEV)
    phase[:, T // 2 :] = 1
    phase[:, -2:] = 2  # crude A/B/C split

    losses: dict[str, float] = {}
    for mode in ["none", "concat", "attention", "gated"]:
        cfg = ModelConfig(image_size=64, memory_mode=MemoryMode(mode))
        torch.manual_seed(42)
        m = WorldModel(cfg).to(DEV)
        out = m(obs, acts, succ, phase=phase, posterior_sample=False)
        assert torch.isfinite(out["loss_total"]), f"{mode}: non-finite loss"
        losses[mode] = float(out["loss_total"])
        print(f"  {mode:10s} loss={losses[mode]:.6f}")
    for mode_name in ["concat", "attention", "gated"]:
        assert (
            abs(losses["none"] - losses[mode_name]) > 1e-4
        ), f"{mode_name} loss matches M0 — memory module silently disengaged"
    print("  ✓ memory modes produce distinct outputs\n")

    print("=== CHECK 2: Gradients flow into MemoryFusion params (M3 / GATED) ===")
    cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.GATED)
    m = WorldModel(cfg).to(DEV)
    m(obs, acts, succ, phase=phase)["loss_total"].backward()
    mem_params = [(n, p) for n, p in m.named_parameters() if n.startswith("memory.")]
    assert mem_params, "MemoryFusion has no parameters? Check module wiring"
    nz = sum(1 for _, p in mem_params if p.grad is not None and p.grad.abs().max() > 0)
    print(f"  {nz}/{len(mem_params)} memory params received nonzero grad")
    assert nz > 0, "Memory module never receives gradient — M1/M2/M3 will train identically to M0"
    print("  ✓ memory module trains\n")

    print("=== CHECK 3: Checkpoint save → load roundtrip ===")
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        p = Path(f.name)
        cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.GATED, use_latent_diffusion=False)
        m1 = WorldModel(cfg).to(DEV)
        o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4)
        save_checkpoint(p, m1, o1, step=42, cfg=cfg)
        m2 = WorldModel(ModelConfig(image_size=64, memory_mode=MemoryMode.GATED)).to(DEV)
        o2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
        step, cfg2 = load_checkpoint(p, m2, o2)
        assert step == 42, f"step mismatch: {step}"
        assert cfg2.memory_mode == MemoryMode.GATED, "memory_mode lost in roundtrip"
        for (n1, q1), (_, q2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.allclose(q1, q2), f"param mismatch at {n1}"
    print("  ✓ ckpt roundtrip preserves params + cfg\n")

    print("=== CHECK 4: Phase-C frames + memory-write events present in real data ===")
    ds = EpisodeChunkDataset(
        DATA, seq_len=16, split="train", exclude_manual=True, max_rows=64, return_meta=True
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    phase_c = 0
    write_evts = 0
    for batch in loader:
        ph = batch["phase"]
        phase_c += int((ph == 2).sum())
        write_evts += int(ph.shape[0])  # first frames always written
        if ph.shape[1] > 1:
            write_evts += int(((ph[:, :-1] == 0) & (ph[:, 1:] == 1)).sum())
    print(f"  Phase-C frames in 64 episodes × 16-frame crops: {phase_c}")
    print(f"  Memory-write events (first-frame + A→B): {write_evts}")
    assert phase_c > 0, "NO PHASE-C FRAMES — eval will be empty; check phase encoding"
    assert write_evts > 0, "NO WRITE EVENTS — memory bank stays empty across all batches"
    print("  ✓ phase + write-mask logic engages on real data\n")

    print("=== CHECK 5: bf16 autocast doesn't NaN over 20 micro-steps ===")
    if DEV == "cuda:0":
        cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.GATED)
        m = WorldModel(cfg).to(DEV)
        o = torch.optim.AdamW(m.parameters(), lr=1e-4)
        for s in range(20):
            o.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                L = m(obs, acts, succ, phase=phase)["loss_total"]
            L.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0)
            o.step()
            assert torch.isfinite(L), f"loss NaN at step {s} under bf16"
        print(f"  ✓ 20-step bf16 micro-train stable (final loss={float(L):.4f})\n")
    else:
        print("  (skipped — no CUDA device)\n")

    print("=== CHECK 6: Eval pipeline runs end-to-end on a fresh ckpt ===")
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        p = Path(f.name)
        cfg = ModelConfig(image_size=64, memory_mode=MemoryMode.NONE)
        m = WorldModel(cfg).to(DEV)
        o = torch.optim.AdamW(m.parameters(), lr=1e-4)
        save_checkpoint(p, m, o, step=0, cfg=cfg)
        from sralnik.training.eval_run import run_eval

        args = argparse.Namespace(
            checkpoint=p,
            data=DATA,
            device=DEV,
            split="val",
            batch=4,
            seq=16,
            num_workers=2,
            seed=0,
            max_rows=8,
            out_parquet=None,
            progress=False,
        )
        run_eval(args)
    print("  ✓ eval runs end-to-end\n")

    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
