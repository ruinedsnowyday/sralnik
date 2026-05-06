#!/usr/bin/env bash
# v3 architecture run: LPIPS + free_bits=0 + kl_balance=0.2 + SD-VAE-decoder.
#
# Replaces the from-scratch CNN decoder with the frozen pretrained SD-VAE
# (stabilityai/sd-vae-ft-mse), adds perceptual loss, and breaks the KL
# free-bits floor that pinned z degenerate during M0/M1 matched training.
#
# Usage:
#   bash scripts/run_v3.sh                          # M3 (gated) + 50k steps + full v3 stack
#   bash scripts/run_v3.sh gated 50000              # explicit memory mode + steps
#   bash scripts/run_v3.sh none 50000               # M0+v3 fair-comparison baseline
#   bash scripts/run_v3.sh attention 75000          # M2+v3 if you want it
#
# Optional env overrides (all have sensible defaults):
#   LPIPS_WEIGHT=0.5          # weight on LPIPS term (default 0.5)
#   FREE_BITS=0.0             # KL clamp floor (default 0.0; full release)
#   KL_BALANCE=0.2            # Dreamer-correct in this code's convention (default 0.2)
#   K_IMAGINE=8               # eval-rollout last-K frames imagined open-loop
#
# Auto-runs eval-rollout on expert_eval after training. GIFs land in
# runs/m_<mode>_v3_<ts>/rollout_eval/gifs/ and sync to S3 within 120s.

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
BATCH="${BATCH:-4}"             # drop to 2 if VAE+LPIPS still OOMs after grad-checkpointing
SEQ="${SEQ:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# Drop LPIPS to recover ~30% throughput. SD-VAE + KL fixes alone still test the
# headline v3 hypothesis (pretrained image prior + un-degenerated z); LPIPS was
# the polish on top. Set NO_LPIPS=1 to disable.
NO_LPIPS="${NO_LPIPS:-0}"
LPIPS_FLAGS="--lpips --lpips-weight $LPIPS_WEIGHT"
if [[ "$NO_LPIPS" == "1" ]]; then
    LPIPS_FLAGS=""
    echo "==> LPIPS disabled (NO_LPIPS=1)"
fi

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
DATA_DIR="${DATA_DIR:-/mnt/data/sralnik/data/ithor_v2}"
S3_RUNS="${S3_RUNS:-s3://sralnik-runs-213128717646/runs}"

cd "$REPO_DIR"

echo "==> git pull (ff-only)"
git fetch --quiet origin main 2>&1 | tail -3 || true
git pull --ff-only 2>&1 | tail -3 || true
echo

# Mandatory: aws-ofi-nccl plugin segfaults on this instance (no EFA attached).
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TS=$(date -u +%Y%m%dT%H%M%S)
RUN="runs/m_${MODE}_v3_${TS}"
mkdir -p "$RUN"

cat > "$RUN/run_meta.txt" <<EOF
git_sha=$(git rev-parse HEAD)
memory=$MODE
max_steps=$MAX_STEPS
arch=v3
  lpips=$([[ "$NO_LPIPS" == "1" ]] && echo "off" || echo "on (weight=$LPIPS_WEIGHT)")
  free_bits=$FREE_BITS
  kl_balance=$KL_BALANCE
  sd_vae_decoder=on (stabilityai/sd-vae-ft-mse, frozen + grad-checkpointed)
  pixel_shuffle=off (sd_vae replaces the CNN decoder)
diffusion=off
batch=$BATCH
seq=$SEQ
num_workers=$NUM_WORKERS
ckpt_every=2500
data=$DATA_DIR
started_utc=$(date -u --iso-8601=seconds)
host=$(hostname)
EOF
uv pip freeze >> "$RUN/run_meta.txt"

echo "================================================================"
echo "  SRALNIK v3 architecture run"
echo "  condition:    $MODE  +lpips +sd_vae +free_bits=$FREE_BITS +kl_balance=$KL_BALANCE"
echo "  max_steps:    $MAX_STEPS"
echo "  ckpt dir:     $RUN"
echo "  S3 dest:      $S3_RUNS/$(basename $RUN)/"
echo "  start utc:    $(date -u --iso-8601=seconds)"
echo "================================================================"
echo

uv run torchrun --redirects 1 --tee 1 --standalone --nproc_per_node=8 \
    -m sralnik.training train \
    --data "$DATA_DIR" \
    --batch "$BATCH" --seq "$SEQ" \
    --max-steps "$MAX_STEPS" \
    --memory "$MODE" --bf16 --num-workers "$NUM_WORKERS" \
    --ckpt-dir "$RUN" --ckpt-every 2500 \
    $LPIPS_FLAGS \
    --free-bits "$FREE_BITS" --kl-balance "$KL_BALANCE" \
    --sd-vae \
    2>&1 | tee "$RUN/stdout.log"

echo
echo "================================================================"
echo "  v3 training finished at $(date -u --iso-8601=seconds)"
echo "================================================================"

if [[ -f "$RUN/last.pt" ]]; then
    EVAL_OUT="$RUN/rollout_eval"
    echo
    echo "==> running eval-rollout for v3 run  (out: $EVAL_OUT)"
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
echo "  v3 complete (train + eval) at $(date -u --iso-8601=seconds)"
echo "  S3 sync should land within 120s; verify with:"
echo "    aws s3 ls $S3_RUNS/$(basename $RUN)/"
echo "================================================================"

# Prune intermediate ckpts.
KEEP_LAST=$(ls -1 "$RUN"/step_*.pt 2>/dev/null | tail -1 || true)
for ck in "$RUN"/step_*.pt; do
    [[ -f "$ck" && "$ck" != "$KEEP_LAST" ]] && rm -f "$ck"
done
echo "pruned intermediate step_*.pt; kept: $(ls $RUN/{last.pt,step_*.pt} 2>/dev/null | tr '\n' ' ')"
