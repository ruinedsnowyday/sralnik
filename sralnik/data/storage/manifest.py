"""Build a parquet manifest describing all collected episodes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .writer import EpisodeRecord


class ManifestBuilder:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.records: list[EpisodeRecord] = []

    def add(self, record: EpisodeRecord) -> None:
        self.records.append(record)

    def write(self) -> Path:
        out = self.root / "manifest.parquet"
        if not self.records:
            df = pd.DataFrame(
                columns=[
                    "episode_id", "scene", "episode_type", "probe_name",
                    "gap_length", "seed", "split", "num_steps", "success",
                    "failure_reason", "target_object_id", "target_receptacle_id",
                    "relative_path",
                ]
            )
        else:
            df = pd.DataFrame([asdict(r) for r in self.records])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out
