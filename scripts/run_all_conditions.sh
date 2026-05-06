#!/usr/bin/env bash
# Sequentially run all four memory conditions (M0 -> M1 -> M2 -> M3).
# Each condition fires only if the previous one wrote its last.pt successfully.
# Run this from the repo root; expects scripts/run_condition.sh in the same tree.
#
# Usage:
#   bash scripts/run_all_conditions.sh                # 75k steps each (default)
#   MAX_STEPS=50000 bash scripts/run_all_conditions.sh
#
# Logs live under runs/m_<mode>_<ts>/ and are auto-synced to S3 by the evac watcher.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
MAX_STEPS="${MAX_STEPS:-75000}"

cd "$REPO_DIR"

ORDER=(none concat attention gated)
T0=$(date -u --iso-8601=seconds)
echo "== START $T0  conditions=${ORDER[*]}  max_steps=$MAX_STEPS =="

for MODE in "${ORDER[@]}"; do
    echo
    echo "== launching $MODE =="
    bash scripts/run_condition.sh "$MODE" "$MAX_STEPS"

    # Sanity: did this condition actually finish?
    LAST_RUN=$(ls -1d runs/m_${MODE}_* 2>/dev/null | tail -1 || true)
    if [[ -z "$LAST_RUN" || ! -f "$LAST_RUN/last.pt" ]]; then
        echo "FATAL: $MODE did not produce $LAST_RUN/last.pt; halting before next condition." >&2
        exit 1
    fi
    echo "== $MODE done; last.pt at $LAST_RUN/last.pt =="
done

echo
echo "== ALL CONDITIONS DONE at $(date -u --iso-8601=seconds) (started $T0) =="
echo "Run eval next:  bash scripts/run_eval_all.sh   (or step 4 of the runbook)"
