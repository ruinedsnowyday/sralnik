"""Load HDF5 episodes described by ``manifest.parquet``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["EpisodeChunkDataset", "collate_fn", "crop_mix_seed"]


def crop_mix_seed(base_seed: int, epoch: int, idx: int) -> int:
    """Deterministic RNG seed for window crops; stable across workers when ``epoch`` is synced."""
    return int((int(base_seed) + int(epoch) * 1_000_003 + int(idx) * 917_521) & 0x7FFFFFFF)


def _norm_gap_length(raw: object) -> int:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _norm_probe(raw: object, episode_type: str) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return str(episode_type) if episode_type else "unknown"
    return str(raw)


class EpisodeChunkDataset(Dataset):
    """Random crops of length ``seq_len`` from episodes (train/val/test)."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        seq_len: int = 16,
        split: str = "train",
        exclude_manual: bool = True,
        scenes: tuple[str, ...] | None = None,
        max_rows: int | None = None,
        seed: int = 0,
        epoch_shared: Any = None,
        return_meta: bool = False,
    ):
        super().__init__()
        self.root = Path(data_root)
        self.seq_len = int(seq_len)
        self.split = split
        self._base_seed = int(seed)
        self._epoch_local = 0
        self._epoch_shared = epoch_shared
        self.return_meta = bool(return_meta)

        df = pd.read_parquet(self.root / "manifest.parquet")
        df = df[df["split"] == split]
        # Drop rows for episodes whose collection failed: empty/non-.h5 relative_path
        # (manifest keeps a placeholder row even when the writer never produced a file).
        df = df[df["relative_path"].fillna("").str.endswith(".h5")]
        if exclude_manual:
            df = df[df["episode_type"] != "manual"]
        if scenes is not None:
            df = df[df["scene"].isin(scenes)]
        if max_rows is not None:
            df = df.head(max_rows)
        if len(df) == 0:
            raise ValueError(f"No episodes for split={split!r} under {self.root}")
        self._df = df.reset_index(drop=True)

    def set_epoch(self, epoch: int) -> None:
        """Sync crop RNG across DDP + DataLoader workers (use with ``epoch_shared``)."""
        e = int(epoch)
        self._epoch_local = e
        if self._epoch_shared is not None:
            self._epoch_shared.value = e

    def _epoch_for_crop(self) -> int:
        if self._epoch_shared is not None:
            return int(self._epoch_shared.value)
        return self._epoch_local

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx]
        path = self.root / row["relative_path"]
        rng = np.random.default_rng(crop_mix_seed(self._base_seed, self._epoch_for_crop(), idx))
        with h5py.File(path, "r") as f:
            T = int(f["rgb"].shape[0])
            if T < self.seq_len:
                start = 0
                pad = self.seq_len - T
            else:
                start = int(rng.integers(0, T - self.seq_len + 1))
                pad = 0
            sl = slice(start, start + self.seq_len)
            rgb = np.asarray(f["rgb"][sl], dtype=np.float32) / 255.0
            if rgb.shape[0] < self.seq_len:
                rgb = np.pad(
                    rgb,
                    ((0, pad), (0, 0), (0, 0), (0, 0)),
                    mode="edge",
                )
            act = np.asarray(f["action_id"][sl], dtype=np.int64)
            succ = np.asarray(f["action_success"][sl], dtype=np.bool_)
            phase = np.asarray(f["phase"][sl], dtype=np.int64)
            if act.shape[0] < self.seq_len:
                act = np.pad(act, (0, pad), mode="edge")
                succ = np.pad(succ, (0, pad), mode="edge")
                phase = np.pad(phase, (0, pad), mode="edge")

        rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
        episode_type = str(row.get("episode_type", "") or "")
        out: dict[str, Any] = {
            "obs": rgb_t,
            "actions": torch.from_numpy(act).long(),
            "action_success": torch.from_numpy(succ),
            "phase": torch.from_numpy(phase).long(),
            "episode_id": row["episode_id"],
        }
        if self.return_meta:
            gl = _norm_gap_length(row.get("gap_length"))
            out["probe_name"] = _norm_probe(row.get("probe_name"), episode_type)
            out["gap_length"] = gl
            out["scene"] = str(row.get("scene", "") or "")
            out["episode_type"] = episode_type
        return out


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["obs"] = torch.stack([b["obs"] for b in batch], dim=0)
    out["actions"] = torch.stack([b["actions"] for b in batch], dim=0)
    out["action_success"] = torch.stack([b["action_success"] for b in batch], dim=0)
    out["phase"] = torch.stack([b["phase"] for b in batch], dim=0)
    out["episode_id"] = [b["episode_id"] for b in batch]
    if "probe_name" in batch[0]:
        out["probe_name"] = [b["probe_name"] for b in batch]
        out["gap_length"] = torch.tensor([b["gap_length"] for b in batch], dtype=torch.long)
        out["scene"] = [b["scene"] for b in batch]
        out["episode_type"] = [b["episode_type"] for b in batch]
    return out
