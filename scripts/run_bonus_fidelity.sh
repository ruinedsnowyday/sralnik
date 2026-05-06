#!/usr/bin/env bash
# Bonus v2 fidelity run: take a memory mode (default M3 = gated) and stack the
# three "cheap fidelity wins" identified in the audit:
#   1. LPIPS perceptual loss (sharper outputs in feature space)
#   2. PixelShuffle decoder upsampling (no checkerboard artefacts)
#   3. KL relaxation (free_bits=0) + Dreamer-correct kl_balance direction (=0.2 in
#      this code's formula, which puts 80%% of gradient on the prior network)
#
# All three flags are opt-in CLI overrides; the underlying training script is the
# same as run_condition.sh, just with extra args. Running this does NOT affect
# the M0-M3 ablation runs (those stay matched at the original config).
#
# Usage:
#   bash scripts/run_bonus_fidelity.sh                                  # M3 + 50k steps + all three fixes
#   bash scripts/run_bonus_fidelity.sh gated 50000                      # explicit
#   bash scripts/run_bonus_fidelity.sh attention 75000                  # use M2 instead of M3
#   LPIPS_WEIGHT=0.3 bash scripts/run_bonus_fidelity.sh                 # tune the LPIPS weight
#   FREE_BITS=0.5 KL_BALANCE=0.2 bash scripts/run_bonus_fidelity.sh     # less aggressive KL
#
# Auto-runs eval-rollout on expert_eval at the end (same as run_condition.sh).

set -euo pipefail

MODE="${1:-gated}"
MAX_STEPS="${2:-50000}"

case "$MODE" in
    none|concat|attention|gated) ;;
    *) echo "error: memory_mode must be one of: none concat attention gated (got: $MODE)" >&2; exit 2 ;;
esac

LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.5}"
FREE_BITS="${FREE_BITS:-0.0}"
KL_BALANCE="${KL_BALANCE:-0.2}"

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
DATA_DIR="${DATA_DIR:-/mnt/data/sralnik/data/ithor_v2}"
S3_RUNS="${S3_RUNS:-s3://sralnik-runs-213128717646/runs}"

cd "$REPO_DIR"

# Pull any code fixes pushed since the queue started.
echo "==> git pull (ff-only)"
git fetch --quiet origin main 2>&1 | tail -3 || true
git pull --ff-only 2>&1 | tail -3 || true
echo

# Mandatory: aws-ofi-nccl plugin segfaults on this instance (no EFA attached).
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TS=$(date -u +%Y%m%dT%H%M%S)
RUN="runs/m_${MODE}_v2fid_${TS}"
mkdir -p "$RUN"

cat > "$RUN/run_meta.txt" <<EOF
git_sha=$(git rev-parse HEAD)
memory=$MODE
max_steps=$MAX_STEPS
diffusion=off
resume=fresh
v2_fidelity=on
  lpips=on
  lpips_weight=$LPIPS_WEIGHT
  pixel_shuffle=on
  free_bits=$FREE_BITS
  kl_balance=$KL_BALANCE
batch=4
seq=16
ckpt_every=2500
data=$DATA_DIR
started_utc=$(date -u --iso-8601=seconds)
host=$(hostname)
EOF
uv pip freeze >> "$RUN/run_meta.txt"

echo "================================================================"
echo "  SRALNIK v2 fidelity bonus run"
echo "  condition:    $MODE  (+lpips +pixel-shuffle +free_bits=$FREE_BITS +kl_balance=$KL_BALANCE)"
echo "  max_steps:    $MAX_STEPS"
echo "  ckpt dir:     $RUN"
echo "  S3 dest:      $S3_RUNS/$(basename $RUN)/"
echo "  start utc:    $(date -u --iso-8601=seconds)"
echo "================================================================"
echo

uv run torchrun --redirects 1 --tee 1 --standalone --nproc_per_node=8 \
    -m sralnik.training train \
    --data "$DATA_DIR" \
    --batch 4 --seq 16 \
    --max-steps "$MAX_STEPS" \
    --memory "$MODE" --bf16 --num-workers 4 \
    --ckpt-dir "$RUN" --ckpt-every 2500 \
    --lpips --lpips-weight "$LPIPS_WEIGHT" \
    --pixel-shuffle \
    --free-bits "$FREE_BITS" --kl-balance "$KL_BALANCE" \
    2>&1 | tee "$RUN/stdout.log"

echo
echo "================================================================"
echo "  bonus training finished at $(date -u --iso-8601=seconds)"
echo "================================================================"

if [[ -f "$RUN/last.pt" ]]; then
    EVAL_OUT="$RUN/rollout_eval"
    echo
    echo "==> running eval-rollout for v2-fidelity bonus  (out: $EVAL_OUT)"
    uv run python -m sralnik.training eval-rollout \
        --checkpoint "$RUN/last.pt" \
        --data "$DATA_DIR" \
        --split expert_eval \
        --k-imagine "${K_IMAGINE:-8}" \
        --out-dir "$EVAL_OUT" \
        2>&1 | tee "$RUN/rollout_eval.log"
    echo "==> eval-rollout done; gifs in $EVAL_OUT/gifs/"
else
    echo "WARN: $RUN/last.pt missing, skipping eval-rollout."
fi

echo
echo "================================================================"
echo "  bonus complete (train + eval) at $(date -u --iso-8601=seconds)"
echo "  S3 sync should land within 120s; verify with:"
echo "    aws s3 ls $S3_RUNS/$(basename $RUN)/"
echo "================================================================"

# Prune intermediate ckpts.
KEEP_LAST=$(ls -1 "$RUN"/step_*.pt 2>/dev/null | tail -1 || true)
for ck in "$RUN"/step_*.pt; do
    [[ -f "$ck" && "$ck" != "$KEEP_LAST" ]] && rm -f "$ck"
done
echo "pruned intermediate step_*.pt; kept: $(ls $RUN/{last.pt,step_*.pt} 2>/dev/null | tr '\n' ' ')"
