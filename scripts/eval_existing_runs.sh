#!/usr/bin/env bash
# Run open-loop rollout eval on every condition under runs/ that has a last.pt
# but no rollout_eval/ subdir yet. Useful to backfill evals for conditions
# that finished BEFORE the auto-eval was wired into run_condition.sh.
#
# Usage:
#   bash scripts/eval_existing_runs.sh                           # all missing evals
#   bash scripts/eval_existing_runs.sh runs/m_none_<ts>          # one specific run
#   FORCE=1 bash scripts/eval_existing_runs.sh                   # re-eval even if rollout_eval exists
#   K_IMAGINE=4 bash scripts/eval_existing_runs.sh               # override last-K
#   DEVICE=cpu bash scripts/eval_existing_runs.sh                # CPU instead of cuda:0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
DATA_DIR="${DATA_DIR:-/mnt/data/sralnik/data/ithor_v2}"
K_IMAGINE="${K_IMAGINE:-8}"
DEVICE="${DEVICE:-cuda:0}"
FORCE="${FORCE:-0}"

cd "$REPO_DIR"

# Pull any eval-related fixes pushed since last invocation.
echo "==> git pull (ff-only)"
git fetch --quiet origin main 2>&1 | tail -3 || true
git pull --ff-only 2>&1 | tail -3 || true
echo

if [[ $# -gt 0 ]]; then
    RUNS=("$@")
else
    mapfile -t RUNS < <(ls -d runs/m_*_*/ 2>/dev/null | sed 's:/$::' | sort)
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "no condition run dirs found under runs/m_*_*"
    exit 0
fi

echo "==> candidates:"
for run in "${RUNS[@]}"; do echo "    $run"; done
echo

DONE=0
SKIPPED=0
FAILED=0
for run in "${RUNS[@]}"; do
    if [[ ! -f "$run/last.pt" ]]; then
        echo "SKIP $run  (no last.pt yet)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if [[ -d "$run/rollout_eval" && "$FORCE" != "1" ]]; then
        echo "SKIP $run  (rollout_eval/ exists; FORCE=1 to re-run)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo
    echo "================================================================"
    echo "  eval-rollout: $run"
    echo "  device:       $DEVICE"
    echo "  K_IMAGINE:    $K_IMAGINE"
    echo "================================================================"

    if uv run python -m sralnik.training eval-rollout \
        --checkpoint "$run/last.pt" \
        --data "$DATA_DIR" \
        --device "$DEVICE" \
        --split expert_eval \
        --k-imagine "$K_IMAGINE" \
        --out-dir "$run/rollout_eval" \
        2>&1 | tee "$run/rollout_eval.log"; then
        DONE=$((DONE + 1))
        echo "OK   $run -> $run/rollout_eval/gifs/"
    else
        FAILED=$((FAILED + 1))
        echo "FAIL $run  (see $run/rollout_eval.log)"
    fi
done

echo
echo "================================================================"
echo "  summary: $DONE done, $SKIPPED skipped, $FAILED failed"
echo "================================================================"
[[ $FAILED -eq 0 ]] || exit 1
