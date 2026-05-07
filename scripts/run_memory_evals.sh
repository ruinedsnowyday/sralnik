#!/usr/bin/env bash
# Run one-GPU memory diagnostics after training finishes.
#
# Usage:
#   M3_RUN=runs/m_gated_v2fid_<ts> M0_RUN=runs/m_none_v2fid_<ts> \
#   DATA_DIR=/mnt/data/sralnik/data/ithor_v2 DEVICE=cuda:0 \
#   SPLITS="val test expert_eval" bash scripts/run_memory_evals.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
DATA_DIR="${DATA_DIR:-/mnt/data/sralnik/data/ithor_v2}"
DEVICE="${DEVICE:-cuda:0}"
SPLITS="${SPLITS:-val test expert_eval}"
K_IMAGINE="${K_IMAGINE:-8}"
MAX_ROWS="${MAX_ROWS:-}"
M3_RUN="${M3_RUN:?set M3_RUN=runs/<m3_run>}"
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

echo "==> git pull (ff-only)"
git fetch --quiet origin main 2>&1 | tail -3 || true
git pull --ff-only 2>&1 | tail -3 || true
echo

max_rows_args=()
if [[ -n "$MAX_ROWS" ]]; then
  max_rows_args=(--max-rows "$MAX_ROWS")
fi

run_name() {
  basename "$1"
}

run_existing_eval() {
  local run="$1"
  local split="$2"
  local name
  name="$(run_name "$run")"
  mkdir -p "eval/$name/$split"

  # The existing teacher-forced eval excludes manual episodes internally, so it
  # is useful for val/test but not for the expert_eval manual split.
  if [[ "$split" != "expert_eval" ]]; then
    "$PYTHON_BIN" -m sralnik.training eval \
      --checkpoint "$run/last.pt" \
      --data "$DATA_DIR" \
      --device "$DEVICE" \
      --split "$split" \
      --batch 8 \
      --seq 16 \
      --out-parquet "eval/$name/$split/phase_c.parquet" \
      "${max_rows_args[@]}" \
      2>&1 | tee "eval/$name/$split/phase_c.log"
  fi

  "$PYTHON_BIN" -m sralnik.training eval-rollout \
    --checkpoint "$run/last.pt" \
    --data "$DATA_DIR" \
    --device "$DEVICE" \
    --split "$split" \
    --k-imagine "$K_IMAGINE" \
    --out-dir "eval/$name/$split/rollout" \
    "${max_rows_args[@]}" \
    2>&1 | tee "eval/$name/$split/rollout.log"
}

run_latent_cache() {
  local run="$1"
  local split="$2"
  local name
  name="$(run_name "$run")"
  mkdir -p "eval/$name/$split"
  "$PYTHON_BIN" -m sralnik.training eval-latent-cache \
    --checkpoint "$run/last.pt" \
    --data "$DATA_DIR" \
    --device "$DEVICE" \
    --split "$split" \
    --out-dir "eval/$name/$split/latent_cache" \
    "${max_rows_args[@]}" \
    2>&1 | tee "eval/$name/$split/latent_cache.log"
}

for split in $SPLITS; do
  echo
  echo "================================================================"
  echo "  split=$split"
  echo "================================================================"

  run_existing_eval "$M3_RUN" "$split"

  m3_name="$(run_name "$M3_RUN")"
  "$PYTHON_BIN" -m sralnik.training eval-memory-trace \
    --checkpoint "$M3_RUN/last.pt" \
    --data "$DATA_DIR" \
    --device "$DEVICE" \
    --split "$split" \
    --out-dir "eval/$m3_name/$split/memory_trace" \
    "${max_rows_args[@]}" \
    2>&1 | tee "eval/$m3_name/$split/memory_trace.log"

  "$PYTHON_BIN" -m sralnik.training eval-memory-intervention \
    --checkpoint "$M3_RUN/last.pt" \
    --data "$DATA_DIR" \
    --device "$DEVICE" \
    --split "$split" \
    --k-imagine "$K_IMAGINE" \
    --out-dir "eval/$m3_name/$split/memory_interventions" \
    "${max_rows_args[@]}" \
    2>&1 | tee "eval/$m3_name/$split/memory_interventions.log"

  run_latent_cache "$M3_RUN" "$split"

  if [[ -n "$M0_RUN" ]]; then
    run_existing_eval "$M0_RUN" "$split"
    run_latent_cache "$M0_RUN" "$split"
    m0_name="$(run_name "$M0_RUN")"
    mkdir -p "eval/latent_probe_compare"
    "$PYTHON_BIN" -m sralnik.training eval-latent-probes \
      --m0-cache "eval/$m0_name/$split/latent_cache" \
      --m3-cache "eval/$m3_name/$split/latent_cache" \
      --out-dir "eval/latent_probe_compare/$split" \
      2>&1 | tee "eval/latent_probe_compare/$split.log"
  fi
done

echo
echo "==> memory evals complete under eval/"
