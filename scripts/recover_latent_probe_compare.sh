#!/usr/bin/env bash
# Recover missing latent caches and latent probe comparisons for existing M0/M3 evals.
#
# This is intended for the current AWS run after run_memory_evals.sh reached
# latent-probe comparison but one split cache was missing.
#
# Usage:
#   bash scripts/recover_latent_probe_compare.sh
#   SPLITS="expert_eval" DEVICE=cuda:1 bash scripts/recover_latent_probe_compare.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
DATA_DIR="${DATA_DIR:-/mnt/data/sralnik/data/ithor_v2}"
DEVICE="${DEVICE:-cuda:1}"
SPLITS="${SPLITS:-val test expert_eval}"
M3_RUN="${M3_RUN:-}"
M0_RUN="${M0_RUN:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

cd "$REPO_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "$M3_RUN" ]]; then
  M3_RUN="$(ls -td runs/m_gated_v2fid_* | head -1)"
fi
if [[ -z "$M0_RUN" ]]; then
  M0_RUN="$(ls -td runs/m_none_v2fid_* | head -1)"
fi

M3_NAME="$(basename "$M3_RUN")"
M0_NAME="$(basename "$M0_RUN")"

echo "M3_RUN=$M3_RUN"
echo "M0_RUN=$M0_RUN"
echo "DEVICE=$DEVICE"
echo "PYTHON_BIN=$PYTHON_BIN"

ensure_cache() {
  local run="$1"
  local name="$2"
  local split="$3"
  local cache="eval/$name/$split/latent_cache"
  if [[ -f "$cache/index.parquet" && -f "$cache/features.npz" ]]; then
    echo "OK cache exists: $cache"
    return 0
  fi
  echo "MISSING cache; creating: $cache"
  mkdir -p "eval/$name/$split"
  "$PYTHON_BIN" -m sralnik.training eval-latent-cache \
    --checkpoint "$run/last.pt" \
    --data "$DATA_DIR" \
    --device "$DEVICE" \
    --split "$split" \
    --out-dir "$cache" \
    2>&1 | tee "eval/$name/$split/latent_cache.log"
}

mkdir -p eval/latent_probe_compare

for split in $SPLITS; do
  echo
  echo "================================================================"
  echo "  latent probe recovery split=$split"
  echo "================================================================"

  ensure_cache "$M3_RUN" "$M3_NAME" "$split"
  ensure_cache "$M0_RUN" "$M0_NAME" "$split"

  "$PYTHON_BIN" -m sralnik.training eval-latent-probes \
    --m0-cache "eval/$M0_NAME/$split/latent_cache" \
    --m3-cache "eval/$M3_NAME/$split/latent_cache" \
    --out-dir "eval/latent_probe_compare/$split" \
    2>&1 | tee "eval/latent_probe_compare/$split.log"
done

echo
echo "latent probe recovery complete"
