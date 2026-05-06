"""HDF5 writer for collected episodes.

One file per episode keeps the on-disk layout simple and shardable, lets us
inspect single trajectories quickly, and avoids HDF5 lock contention when we
later parallelise collection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from ..actions import ACTION_NAMES
from ..controller import StepResult


@dataclass
class EpisodeRecord:
    """Metadata describing one finished episode."""

    episode_id: str
    scene: str
    episode_type: str  # "exploration" | probe name
    probe_name: str | None
    gap_length: int | None
    seed: int
    split: str  # "train" | "val" | "test"
    num_steps: int
    success: bool
    failure_reason: str | None
    target_object_id: str | None
    target_receptacle_id: str | None
    relative_path: str  # relative to the manifest dir


class EpisodeWriter:
    """Writes a sequence of ``StepResult`` to a single HDF5 file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        steps: list[StepResult],
        phases: list[str],
        attrs: dict,
    ) -> None:
        if len(steps) != len(phases):
            raise ValueError("steps and phases length mismatch")
        if not steps:
            raise ValueError("Refusing to write empty episode")

        T = len(steps)
        H, W = steps[0].rgb.shape[:2]

        rgb = np.stack([s.rgb for s in steps], axis=0)
        pose = np.stack([s.pose for s in steps], axis=0)
        action_id = np.asarray([s.action_id for s in steps], dtype=np.int16)
        action_success = np.asarray([s.action_success for s in steps], dtype=bool)

        # Optional segmentation / depth.
        seg_arr: np.ndarray | None = None
        if steps[0].instance_seg is not None:
            seg_arr = np.stack([
                (s.instance_seg if s.instance_seg is not None
                 else np.zeros((H, W), dtype=np.uint16))
                for s in steps
            ], axis=0)
        depth_arr: np.ndarray | None = None
        if steps[0].depth is not None:
            depth_arr = np.stack([
                (s.depth if s.depth is not None
                 else np.zeros((H, W), dtype=np.float16))
                for s in steps
            ], axis=0)

        phase_codes = np.asarray(
            [{"A": 0, "B": 1, "C": 2}.get(p, 1) for p in phases], dtype=np.int8
        )
        action_object_ids = [s.action_object_id or "" for s in steps]
        tracked_blobs = [json.dumps(s.tracked_object_states) for s in steps]
        # Encode action object ids as fixed-length S-strings for h5 storage.
        max_obj_len = max((len(x) for x in action_object_ids), default=1)
        action_obj_arr = np.asarray(action_object_ids, dtype=f"S{max(1, max_obj_len)}")

        with h5py.File(self.path, "w") as f:
            for k, v in attrs.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    f.attrs[k] = v
                else:
                    f.attrs[k] = json.dumps(v)
            f.attrs["num_steps"] = T
            f.attrs["height"] = H
            f.attrs["width"] = W
            f.attrs["action_names"] = json.dumps(list(ACTION_NAMES))

            f.create_dataset(
                "rgb",
                data=rgb,
                chunks=(1, H, W, 3),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset("pose", data=pose)
            f.create_dataset("action_id", data=action_id)
            f.create_dataset("action_success", data=action_success)
            f.create_dataset("phase", data=phase_codes)
            f.create_dataset("action_object_id", data=action_obj_arr)
            tracked_ds = f.create_dataset(
                "tracked_objects_json",
                shape=(T,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            tracked_ds[...] = tracked_blobs

            if seg_arr is not None:
                f.create_dataset(
                    "instance_seg",
                    data=seg_arr,
                    chunks=(1, H, W),
                    compression="gzip",
                    compression_opts=4,
                )
            if depth_arr is not None:
                f.create_dataset(
                    "depth",
                    data=depth_arr,
                    chunks=(1, H, W),
                    compression="gzip",
                    compression_opts=4,
                )
