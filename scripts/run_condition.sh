#!/usr/bin/env bash
# Launch one ablation condition on the H100 cluster.
#
# Usage:
#   bash scripts/run_condition.sh <memory_mode> [max_steps] [--diffusion] [--resume <ckpt>]
#
# Examples:
#   bash scripts/run_condition.sh none                      # M0 baseline, 75k steps
#   bash scripts/run_condition.sh concat 50000              # M1 with shorter budget
#   bash scripts/run_condition.sh gated 75000 --diffusion   # M3 + latent diffusion
#   bash scripts/run_condition.sh gated 75000 --resume runs/m_gated_20260506T140000/step_010000.pt
#
# All output goes to runs/m_<mode>_<UTC-timestamp>/, which the evac watcher
# in the other tmux pane mirrors to S3 every 120s.

set -euo pipefail

# --- args ---
if [[ $# -lt 1 ]]; then
    echo "usage: $0 <memory_mode> [max_steps] [--diffusion] [--resume <ckpt>]" >&2
    echo "  memory_mode: none | concat | attention | gated" >&2
    exit 2
fi

MODE="$1"
shift
case "$MODE" in
    none|concat|attention|gated) ;;
    *) echo "error: memory_mode must be one of: none concat attention gated (got: $MODE)" >&2; exit 2 ;;
esac

MAX_STEPS=75000
DIFFUSION_FLAG=""
RESUME_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --diffusion)
            DIFFUSION_FLAG="--diffusion"; shift ;;
        --resume)
            [[ $# -ge 2 ]] || { echo "error: --resume needs a checkpoint path" >&2; exit 2; }
            RESUME_FLAG="--resume $2"; shift 2 ;;
        --help|-h)
            head -25 "$0"; exit 0 ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                MAX_STEPS="$1"; shift
            else
                echo "error: unknown arg: $1" >&2; exit 2
            fi
            ;;
    esac
done

# --- environment ---
REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
DATA_DIR="${DATA_DIR:-/mnt/data/sralnik/data/ithor_v2}"
S3_RUNS="${S3_RUNS:-s3://sralnik-runs-213128717646/runs}"

cd "$REPO_DIR"

# Pull any code fixes pushed since the queue started. Idempotent if already up-to-date.
# Critical for picking up the memory.py SDPA fix before M2/M3 launch.
echo "==> git pull (ff-only)"
git fetch --quiet origin main 2>&1 | tail -3 || true
git pull --ff-only 2>&1 | tail -3 || true
echo

# Mandatory: aws-ofi-nccl plugin segfaults on this instance (no EFA attached).
# Disable it; NVLink handles intra-node NCCL traffic for our single-node 8-GPU run.
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# --- run dir + metadata ---
TS=$(date -u +%Y%m%dT%H%M%S)
RUN="runs/m_${MODE}_${TS}"
mkdir -p "$RUN"

cat > "$RUN/run_meta.txt" <<EOF
git_sha=$(git rev-parse HEAD)
memory=$MODE
max_steps=$MAX_STEPS
diffusion=${DIFFUSION_FLAG:-off}
resume=${RESUME_FLAG:-fresh}
batch=4
seq=16
ckpt_every=2500
data=$DATA_DIR
started_utc=$(date -u --iso-8601=seconds)
host=$(hostname)
EOF
uv pip freeze >> "$RUN/run_meta.txt"

# --- banner ---
echo "================================================================"
echo "  SRALNIK ablation run"
echo "  condition:    $MODE"
echo "  max_steps:    $MAX_STEPS"
echo "  diffusion:    ${DIFFUSION_FLAG:-(off)}"
echo "  resume:       ${RESUME_FLAG:-(fresh)}"
echo "  ckpt dir:     $RUN"
echo "  data:         $DATA_DIR"
echo "  S3 dest:      $S3_RUNS/$(basename $RUN)/"
echo "  start utc:    $(date -u --iso-8601=seconds)"
echo "================================================================"
echo

# --- launch ---
# `--redirects 1 --tee 1` makes child stdout/stderr visible on rank 0 (otherwise
# you only see the torchrun supervisor's lines and tracebacks are buried).
uv run torchrun --redirects 1 --tee 1 --standalone --nproc_per_node=8 \
    -m sralnik.training train \
    --data "$DATA_DIR" \
    --batch 4 --seq 16 \
    --max-steps "$MAX_STEPS" \
    --memory "$MODE" --bf16 --num-workers 4 \
    --ckpt-dir "$RUN" --ckpt-every 2500 \
    $DIFFUSION_FLAG $RESUME_FLAG \
    2>&1 | tee "$RUN/stdout.log"

echo
echo "================================================================"
echo "  $MODE training finished at $(date -u --iso-8601=seconds)"
echo "  see: $RUN/stdout.log, $RUN/metrics.jsonl, $RUN/last.pt"
echo "================================================================"

# --- open-loop rollout eval on expert_eval split (paper qualitative figure) ---
# Runs immediately after training so each condition's GIFs land before the next
# training start. Single-GPU, deterministic z=prior_mu, last K=8 frames imagined.
if [[ -f "$RUN/last.pt" ]]; then
    EVAL_OUT="$RUN/rollout_eval"
    echo
    echo "==> running eval-rollout for $MODE  (out: $EVAL_OUT)"
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
echo "  $MODE complete (train + eval) at $(date -u --iso-8601=seconds)"
echo "  S3 sync should land within 120s; verify with:"
echo "    aws s3 ls $S3_RUNS/$(basename $RUN)/"
echo "================================================================"

# --- prune intermediate ckpts so the next condition has NVMe room ---
# Keeps last.pt + the very last numbered step_*.pt (for resume safety).
KEEP_LAST=$(ls -1 "$RUN"/step_*.pt 2>/dev/null | tail -1 || true)
for ck in "$RUN"/step_*.pt; do
    [[ -f "$ck" && "$ck" != "$KEEP_LAST" ]] && rm -f "$ck"
done
echo "pruned intermediate step_*.pt; kept: $(ls $RUN/{last.pt,step_*.pt} 2>/dev/null | tr '\n' ' ')"
