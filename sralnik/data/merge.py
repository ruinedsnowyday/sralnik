"""Merge two episode roots into one unified dataset.

Use when one scene was collected into a separate directory (e.g. FP403 in
``data/ithor_v2`` and FP1 in ``data/ithor_v2_fp1``) and you want a single
``manifest.parquet`` spanning both.

For each row in ``add``'s manifest where ``success`` is True and
``scene`` matches ``--scene`` (or all scenes if omitted), copy the HDF5 file
from ``add`` into ``base`` (if a non-identical file already exists at the
destination path, it is overwritten). Any conflicting ``success=True`` rows in
``base`` for the same ``episode_id`` are dropped from the merged manifest so
the supplemental run wins (useful when the base manifest has a corrupt scene
from a half-finished Unity crash).

Then write ``base/manifest.parquet`` as the row-wise concatenation of all
remaining ``base`` rows plus all imported ``add`` rows.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def merge_datasets(
    base: Path,
    add: Path,
    *,
    scenes: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> Path:
    base = Path(base).resolve()
    add = Path(add).resolve()
    if base == add:
        raise ValueError("base and add must differ")

    base_m = base / "manifest.parquet"
    add_m = add / "manifest.parquet"
    if not base_m.exists():
        raise FileNotFoundError(base_m)
    if not add_m.exists():
        raise FileNotFoundError(add_m)

    df_base = pd.read_parquet(base_m)
    df_add = pd.read_parquet(add_m)

    ok_add = df_add[df_add["success"]]
    if scenes:
        ok_add = ok_add[ok_add["scene"].isin(scenes)]

    imported_ids = set(ok_add["episode_id"])
    if imported_ids:
        # Drop duplicate rows in base (supplemental replaces them).
        mask = df_base["episode_id"].isin(imported_ids)
        if mask.any():
            df_base = df_base[~mask]

    for _, row in ok_add.iterrows():
        rel = row["relative_path"]
        if not rel or pd.isna(rel):
            continue
        src = add / str(rel)
        dst = base / str(rel)
        if not src.exists():
            raise FileNotFoundError(f"add is missing {src}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    df_merged = pd.concat([df_base, ok_add], ignore_index=True)
    if not dry_run:
        df_merged.to_parquet(base_m, index=False)

    return base_m
