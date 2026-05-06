"""Distributed and single-GPU training for ``WorldModel`` (8× H100 friendly)."""

from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from multiprocessing import Value
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sralnik.models import MemoryMode, ModelConfig, WorldModel

from .dataset import EpisodeChunkDataset, collate_fn


def _is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _init_distributed() -> torch.device:
    if not _is_dist():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    lr = _local_rank()
    torch.cuda.set_device(lr)
    return torch.device("cuda", lr)


def _parse_memory(s: str) -> MemoryMode:
    return MemoryMode(s.lower())


def _h100_perf_defaults() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, cfg: ModelConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    payload: dict = {
        "step": int(step),
        "model_cfg": cfg.to_checkpoint_dict(),
        "model_state": raw.state_dict(),
        "opt_state": opt.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: WorldModel,
    opt: torch.optim.Optimizer | None,
) -> tuple[int, ModelConfig]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_checkpoint_dict(ck["model_cfg"])
    model.load_state_dict(ck["model_state"])
    if opt is not None and ck.get("opt_state") is not None:
        opt.load_state_dict(ck["opt_state"])
    return int(ck.get("step", 0)), cfg


def run_train(args: Namespace) -> None:
    _h100_perf_defaults()
    if getattr(args, "device", None) and not _is_dist():
        device = torch.device(args.device)
    else:
        device = _init_distributed()
    rank = _rank()
    ws = _world_size()
    is_main = rank == 0
    use_bf16 = bool(args.bf16) and device.type == "cuda"
    epoch_shared = Value("i", 0) if args.num_workers > 0 else None

    resume = getattr(args, "resume", None)
    start_step = 0
    if resume:
        ck = torch.load(Path(resume), map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_checkpoint_dict(ck["model_cfg"])
    else:
        cfg = ModelConfig(
            image_size=args.image_size,
            memory_mode=_parse_memory(args.memory),
            use_latent_diffusion=bool(args.diffusion),
        )
    model = WorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume:
        start_step, _ = load_checkpoint(Path(resume), model, opt)

    if _is_dist():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=bool(args.diffusion),
        )

    ds = EpisodeChunkDataset(
        args.data,
        seq_len=args.seq,
        split=args.split,
        exclude_manual=True,
        max_rows=args.max_rows,
        seed=args.seed,
        epoch_shared=epoch_shared,
    )
    sampler = (
        DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True, seed=args.seed) if _is_dist() else None
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=_is_dist(),
    )

    model.train()
    global_step = start_step
    t0 = time.perf_counter()
    epochs = max(1, (args.max_steps + len(loader)) // max(len(loader), 1))

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ds.set_epoch(epoch)
        it = loader
        if is_main:
            it = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)

        for batch in it:
            if global_step >= args.max_steps:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            succ = batch["action_success"].to(device, non_blocking=True)
            phase = batch["phase"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = model(obs, actions, succ, phase=phase)
                losses["loss_total"].backward()
            else:
                losses = model(obs, actions, succ, phase=phase)
                losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            global_step += 1
            if is_main and isinstance(it, tqdm):
                it.set_postfix(
                    loss=float(losses["loss_total"].detach()),
                    step=global_step,
                )
            if is_main:
                metrics_path = Path(args.ckpt_dir) / "metrics.jsonl"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a") as mf:
                    mf.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "loss_total": float(losses["loss_total"].detach()),
                                "loss_rec": float(losses.get("loss_rec", torch.tensor(0.0)).detach()) if "loss_rec" in losses else 0.0,
                                "loss_kl": float(losses.get("loss_kl", torch.tensor(0.0)).detach()) if "loss_kl" in losses else 0.0,
                                "loss_diff": float(losses.get("loss_diff", torch.tensor(0.0)).detach()) if "loss_diff" in losses else 0.0,
                                "t_wall": time.time(),
                            }
                        )
                        + "\n"
                    )
            if is_main and global_step % int(args.ckpt_every) == 0:
                ck_path = Path(args.ckpt_dir) / f"step_{global_step:06d}.pt"
                save_checkpoint(ck_path, model, opt, global_step, cfg)
            if global_step >= args.max_steps:
                break

    if is_main:
        out = Path(args.ckpt_dir) / "last.pt"
        save_checkpoint(out, model, opt, global_step, cfg)
        dt = time.perf_counter() - t0
        print(f"done {global_step} steps in {dt:.1f}s", flush=True)

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()
