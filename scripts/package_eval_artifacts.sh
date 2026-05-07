#!/usr/bin/env bash
# Package compact reusable outputs for S3/Colab. Does not copy raw HDF5 data.
#
# Usage:
#   bash scripts/package_eval_artifacts.sh runs/m_gated_v2fid_<ts> runs/m_none_v2fid_<ts>

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
cd "$REPO_DIR"

TS="$(date -u +%Y%m%dT%H%M%S)"
OUT="artifacts/eval_${TS}"
mkdir -p "$OUT/runs" "$OUT/eval"

{
  echo "created_utc=$(date -u --iso-8601=seconds)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "runs=$*"
  echo "host=$(hostname)"
} > "$OUT/artifact_meta.txt"

for run in "$@"; do
  name="$(basename "$run")"
  mkdir -p "$OUT/runs/$name"
  for f in last.pt run_meta.txt metrics.jsonl stdout.log rollout_eval.log; do
    if [[ -f "$run/$f" ]]; then
      cp "$run/$f" "$OUT/runs/$name/"
    fi
  done
done

if [[ -d eval ]]; then
  cp -R eval/. "$OUT/eval/"
fi

find "$OUT" -type f | sort > "$OUT/inventory.txt"
echo "packaged artifacts -> $OUT"
