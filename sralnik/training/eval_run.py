"""Phase-C reconstruction metrics stratified by manifest fields (probe, gap, scene)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sralnik.models import ModelConfig, WorldModel

from .dataset import EpisodeChunkDataset, collate_fn

_GAP_EXACT = frozenset({20, 100, 300, 1000})


def _gap_bucket(gl: int) -> str:
    if gl < 0:
        return "na"
    if gl in _GAP_EXACT:
        return f"g{gl}"
    return "other"


def _summ_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return (
        df.groupby(key, dropna=False)
        .agg(n=("l1", "size"), l1_mean=("l1", "mean"), mse_mean=("mse", "mean"))
        .sort_values("l1_mean")
    )


def run_eval(args: Namespace) -> None:
    device = torch.device(args.device)
    ck_path = Path(args.checkpoint)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_checkpoint_dict(ck["model_cfg"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = EpisodeChunkDataset(
        args.data,
        seq_len=args.seq,
        split=args.split,
        exclude_manual=True,
        max_rows=args.max_rows,
        seed=args.seed,
        return_meta=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    rows: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, disable=not args.progress):
            obs = batch["obs"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            succ = batch["action_success"].to(device, non_blocking=True)
            phase = batch["phase"].to(device, non_blocking=True)
            out = model(
                obs,
                actions,
                succ,
                phase=phase,
                posterior_sample=False,
                return_reconstructions=True,
            )
            x_hat = out["x_hat"]
            B, T = phase.shape
            for b in range(B):
                pn = batch["probe_name"][b]
                sc = batch["scene"][b]
                et = batch["episode_type"][b]
                gl = int(batch["gap_length"][b].item())
                gb = _gap_bucket(gl)
                for t in range(T):
                    if int(phase[b, t].item()) != 2:
                        continue
                    l1 = F.l1_loss(x_hat[b, t], obs[b, t], reduction="mean").item()
                    mse = F.mse_loss(x_hat[b, t], obs[b, t], reduction="mean").item()
                    rows.append(
                        {
                            "l1": l1,
                            "mse": mse,
                            "probe_name": pn,
                            "gap_length": gl,
                            "gap_bucket": gb,
                            "scene": sc,
                            "episode_type": et,
                        }
                    )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("no Phase-C frames in split (check data / phase labels).")
        return

    overall = pd.Series({"n": len(df), "l1_mean": df["l1"].mean(), "mse_mean": df["mse"].mean()})

    print("\n=== Overall (Phase C, teacher-forced recon, z=μ) ===")
    print(overall.to_string())

    print("\n=== By probe_name ===")
    print(_summ_table(df, "probe_name").to_string())

    print("\n=== By gap_bucket ===")
    print(_summ_table(df, "gap_bucket").to_string())

    print("\n=== By scene ===")
    print(_summ_table(df, "scene").to_string())

    print("\n=== By probe × gap_bucket (sorted by l1_mean) ===")
    both = (
        df.groupby(["probe_name", "gap_bucket"], dropna=False)
        .agg(n=("l1", "size"), l1_mean=("l1", "mean"), mse_mean=("mse", "mean"))
        .reset_index()
        .sort_values("l1_mean")
    )
    print(both.head(48).to_string(index=False))

    if args.out_parquet:
        out_p = Path(args.out_parquet)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_p, index=False)
        print(f"\nwrote per-frame rows to {out_p}")
