"""Minimal training loop and CPU smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sralnik.models import MemoryMode, ModelConfig, WorldModel

from .dataset import EpisodeChunkDataset, collate_fn
from .ddp_train import run_train
from .eval_memory import (
    run_eval_latent_cache,
    run_eval_latent_probes,
    run_eval_memory_intervention,
    run_eval_memory_trace,
)
from .eval_rollout import run_eval_rollout
from .eval_run import run_eval


def smoke_synthetic(
    *,
    batch: int = 2,
    seq: int = 12,
    image_size: int = 256,
    device: str = "cpu",
    memory: MemoryMode = MemoryMode.NONE,
    diffusion: bool = False,
) -> None:
    cfg = ModelConfig(
        image_size=image_size,
        memory_mode=memory,
        use_latent_diffusion=diffusion,
    )
    model = WorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    model.train()
    obs = torch.rand(batch, seq, 3, image_size, image_size, device=device)
    actions = torch.zeros(batch, seq, dtype=torch.long, device=device)
    succ = torch.ones(batch, seq, dtype=torch.bool, device=device)
    phase = torch.zeros(batch, seq, dtype=torch.long, device=device)
    phase[:, seq // 2 :] = 1
    phase[:, -2:] = 2
    losses = model(obs, actions, succ, phase=phase)
    losses["loss_total"].backward()
    opt.step()
    print(
        "smoke_synthetic OK",
        {k: float(v.detach().cpu()) for k, v in losses.items()},
    )


def smoke_fit(
    data_root: Path | str,
    *,
    batch: int = 2,
    seq: int = 12,
    steps: int = 2,
    device: str = "cpu",
    memory: MemoryMode = MemoryMode.NONE,
    diffusion: bool = False,
    max_rows: int | None = 8,
) -> None:
    root = Path(data_root)
    ds = EpisodeChunkDataset(
        root,
        seq_len=seq,
        split="train",
        exclude_manual=True,
        max_rows=max_rows,
    )
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    batch0 = next(iter(loader))
    H, W = batch0["obs"].shape[-2], batch0["obs"].shape[-1]
    cfg = ModelConfig(
        image_size=H,
        memory_mode=memory,
        use_latent_diffusion=diffusion,
    )
    model = WorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    model.train()
    it = iter(loader)
    for s in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        succ = batch["action_success"].to(device)
        phase = batch["phase"].to(device)
        opt.zero_grad(set_to_none=True)
        losses = model(obs, actions, succ, phase=phase)
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        print(f"step {s}", {k: float(v.detach().cpu()) for k, v in losses.items()})


def _parse_memory(s: str) -> MemoryMode:
    return MemoryMode(s.lower())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sralnik.training")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser("smoke-synthetic", help="Rand data + one optim step (no HDF5).")
    p_syn.add_argument("--device", default="cpu")
    p_syn.add_argument("--batch", type=int, default=2)
    p_syn.add_argument("--seq", type=int, default=12)
    p_syn.add_argument("--image-size", type=int, default=64)
    p_syn.add_argument("--memory", type=_parse_memory, default=MemoryMode.NONE)
    p_syn.add_argument("--diffusion", action="store_true")

    p_fit = sub.add_parser("smoke-fit", help="1–N steps on manifest+HDF5 (tiny subset).")
    p_fit.add_argument("--data", type=Path, required=True)
    p_fit.add_argument("--device", default="cpu")
    p_fit.add_argument("--batch", type=int, default=2)
    p_fit.add_argument("--seq", type=int, default=12)
    p_fit.add_argument("--steps", type=int, default=2)
    p_fit.add_argument("--memory", type=_parse_memory, default=MemoryMode.NONE)
    p_fit.add_argument("--diffusion", action="store_true")
    p_fit.add_argument("--max-rows", type=int, default=8)

    p_tr = sub.add_parser(
        "train",
        help="Full training (single GPU or torchrun multi-GPU). On 8×H100: torchrun --nproc_per_node=8 ...",
    )
    p_tr.add_argument("--data", type=Path, required=True)
    p_tr.add_argument(
        "--device",
        default=None,
        help="Force device when not using torchrun (e.g. cpu, cuda:0). Ignored under distributed.",
    )
    p_tr.add_argument("--split", default="train")
    p_tr.add_argument("--batch", type=int, default=4)
    p_tr.add_argument("--seq", type=int, default=16)
    p_tr.add_argument("--max-steps", type=int, default=1000)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--weight-decay", type=float, default=1e-6)
    p_tr.add_argument("--grad-clip", type=float, default=100.0)
    p_tr.add_argument("--image-size", type=int, default=256)
    p_tr.add_argument("--memory", type=_parse_memory, default=MemoryMode.NONE)
    p_tr.add_argument("--diffusion", action="store_true")
    p_tr.add_argument("--bf16", action="store_true", help="bf16 autocast on CUDA (recommended on H100).")
    p_tr.add_argument("--num-workers", type=int, default=8)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--max-rows", type=int, default=None)
    p_tr.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p_tr.add_argument("--ckpt-every", type=int, default=500)
    p_tr.add_argument("--resume", type=Path, default=None)
    # v2 fidelity knobs — all opt-in, leave unset to reproduce M0-M3 ablation behaviour.
    p_tr.add_argument(
        "--lpips",
        action="store_true",
        help="Add LPIPS perceptual loss to L1 reconstruction. ~30%% slower per step.",
    )
    p_tr.add_argument(
        "--lpips-weight",
        type=float,
        default=0.5,
        help="Weight on LPIPS term: loss_rec = L1 + lpips_weight * LPIPS.",
    )
    p_tr.add_argument(
        "--pixel-shuffle",
        action="store_true",
        help="Decoder upsampling via PixelShuffle (sub-pixel conv) instead of ConvTranspose.",
    )
    p_tr.add_argument(
        "--free-bits",
        type=float,
        default=None,
        help="Override ModelConfig.free_bits. Set to 0.0 to let KL escape the floor.",
    )
    p_tr.add_argument(
        "--kl-balance",
        type=float,
        default=None,
        help="Override ModelConfig.kl_balance. Code's formula: balance*KL_with_post_grad + "
             "(1-balance)*KL_with_prior_grad. Dreamer-V2 'α=0.8 favoring prior' = 0.2 here.",
    )
    p_tr.add_argument(
        "--sd-vae",
        action="store_true",
        help="v3: replace the from-scratch CNN decoder with the frozen pretrained "
             "stabilityai/sd-vae-ft-mse decoder. Pretrained natural-image prior; "
             "~25%% slower per step. Requires diffusers >=0.27 in the env.",
    )

    p_ev = sub.add_parser("eval", help="Phase-C L1/MSE tables from a checkpoint (see docs/ARCHITECTURE.md §7).")
    p_ev.add_argument("--checkpoint", type=Path, required=True)
    p_ev.add_argument("--data", type=Path, required=True)
    p_ev.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p_ev.add_argument("--split", default="val")
    p_ev.add_argument("--batch", type=int, default=8)
    p_ev.add_argument("--seq", type=int, default=16)
    p_ev.add_argument("--num-workers", type=int, default=4)
    p_ev.add_argument("--seed", type=int, default=0)
    p_ev.add_argument("--max-rows", type=int, default=None)
    p_ev.add_argument("--out-parquet", type=Path, default=None)
    p_ev.add_argument("--no-progress", action="store_true")

    p_er = sub.add_parser(
        "eval-rollout",
        help="Open-loop Phase-C rollout: imagine the last K frames + write GIFs (paper qualitative figure).",
    )
    p_er.add_argument("--checkpoint", type=Path, required=True)
    p_er.add_argument("--data", type=Path, required=True)
    p_er.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p_er.add_argument(
        "--split",
        default="expert_eval",
        help="Manifest split to pull episodes from. Default 'expert_eval' (manually recorded).",
    )
    p_er.add_argument(
        "--k-imagine",
        type=int,
        default=8,
        help="Number of last frames to roll out open-loop. Frames before this are encoder-warmup.",
    )
    p_er.add_argument("--max-rows", type=int, default=None)
    p_er.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination dir; defaults to <checkpoint_parent>/rollout_eval",
    )
    p_er.add_argument(
        "--gif-frame-ms",
        type=int,
        default=250,
        help="Per-frame duration in the output GIF, milliseconds. 250 = 4fps.",
    )

    p_mt = sub.add_parser("eval-memory-trace", help="One-GPU memory gate/retrieval trace eval.")
    p_mt.add_argument("--checkpoint", type=Path, required=True)
    p_mt.add_argument("--data", type=Path, required=True)
    p_mt.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p_mt.add_argument("--split", default="val")
    p_mt.add_argument("--max-rows", type=int, default=None)
    p_mt.add_argument("--out-dir", type=Path, required=True)

    p_mi = sub.add_parser("eval-memory-intervention", help="One-GPU eval-time memory ablations.")
    p_mi.add_argument("--checkpoint", type=Path, required=True)
    p_mi.add_argument("--data", type=Path, required=True)
    p_mi.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p_mi.add_argument("--split", default="val")
    p_mi.add_argument("--max-rows", type=int, default=None)
    p_mi.add_argument("--k-imagine", type=int, default=8)
    p_mi.add_argument(
        "--interventions",
        default="normal,disabled_all,disabled_phase_c,shuffled_memory,all_history",
        help="Comma-separated variants: normal,disabled_all,disabled_phase_c,shuffled_memory,all_history.",
    )
    p_mi.add_argument("--max-gifs", type=int, default=6)
    p_mi.add_argument("--out-dir", type=Path, required=True)

    p_lc = sub.add_parser("eval-latent-cache", help="Extract reusable frozen h/z/prior features.")
    p_lc.add_argument("--checkpoint", type=Path, required=True)
    p_lc.add_argument("--data", type=Path, required=True)
    p_lc.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p_lc.add_argument("--split", default="val")
    p_lc.add_argument("--max-rows", type=int, default=None)
    p_lc.add_argument("--out-dir", type=Path, required=True)

    p_lp = sub.add_parser("eval-latent-probes", help="Train simple probes on M0/M3 latent caches.")
    p_lp.add_argument("--m0-cache", type=Path, required=True)
    p_lp.add_argument("--m3-cache", type=Path, required=True)
    p_lp.add_argument("--seed", type=int, default=0)
    p_lp.add_argument("--out-dir", type=Path, required=True)

    args = p.parse_args(argv)

    if args.cmd == "smoke-synthetic":
        smoke_synthetic(
            batch=args.batch,
            seq=args.seq,
            image_size=args.image_size,
            device=args.device,
            memory=args.memory,
            diffusion=args.diffusion,
        )
        return 0
    if args.cmd == "smoke-fit":
        smoke_fit(
            args.data,
            batch=args.batch,
            seq=args.seq,
            steps=args.steps,
            device=args.device,
            memory=args.memory,
            diffusion=args.diffusion,
            max_rows=args.max_rows,
        )
        return 0
    if args.cmd == "train":
        run_train(args)
        return 0
    if args.cmd == "eval":
        args.progress = not args.no_progress
        run_eval(args)
        return 0
    if args.cmd == "eval-rollout":
        if args.out_dir is None:
            args.out_dir = Path(args.checkpoint).parent / "rollout_eval"
        run_eval_rollout(args)
        return 0
    if args.cmd == "eval-memory-trace":
        run_eval_memory_trace(args)
        return 0
    if args.cmd == "eval-memory-intervention":
        run_eval_memory_intervention(args)
        return 0
    if args.cmd == "eval-latent-cache":
        run_eval_latent_cache(args)
        return 0
    if args.cmd == "eval-latent-probes":
        run_eval_latent_probes(args)
        return 0
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
