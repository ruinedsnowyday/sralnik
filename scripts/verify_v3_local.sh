#!/usr/bin/env bash
# Local laptop-side syntax verification for the v3 architecture changes.
# No torch needed — uses system Python's ast module + bash -n.
#
# Usage:
#   bash scripts/verify_v3_local.sh
#
# Run this BEFORE committing/pushing the v3 work to catch typos and
# indentation errors that would otherwise surface as crashes mid-training
# on the H100 instance.

set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}

PY_FILES=(
    sralnik/models/config.py
    sralnik/models/decoder.py
    sralnik/models/encoder.py
    sralnik/models/lpips_loss.py
    sralnik/models/memory.py
    sralnik/models/sd_vae_decoder.py
    sralnik/models/world_model.py
    sralnik/training/dataset.py
    sralnik/training/ddp_train.py
    sralnik/training/eval_rollout.py
    sralnik/training/eval_run.py
    sralnik/training/train.py
    verify_correctness.py
    verify_v2_arch.py
)

SH_FILES=(
    scripts/eval_existing_runs.sh
    scripts/run_all_conditions.sh
    scripts/run_bonus_fidelity.sh
    scripts/run_condition.sh
    scripts/run_v3.sh
    scripts/verify_v3_local.sh
)

PY_FAIL=0
SH_FAIL=0

echo "=== Python syntax (ast.parse) ==="
for f in "${PY_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  SKIP $f  (does not exist)"
        continue
    fi
    if "$PY" -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo "  OK   $f"
    else
        echo "  FAIL $f"
        PY_FAIL=$((PY_FAIL + 1))
    fi
done

echo
echo "=== Shell syntax (bash -n) ==="
for f in "${SH_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  SKIP $f  (does not exist)"
        continue
    fi
    if bash -n "$f" 2>/dev/null; then
        echo "  OK   $f"
    else
        echo "  FAIL $f"
        SH_FAIL=$((SH_FAIL + 1))
    fi
done

echo
echo "=== Git status (what would be committed) ==="
git status --short

echo
echo "=== Diff stat for staged + unstaged ==="
git diff --stat HEAD

echo
echo "================================================================"
if [[ $PY_FAIL -eq 0 && $SH_FAIL -eq 0 ]]; then
    echo "  ALL SYNTAX OK  (Python: ${#PY_FILES[@]} files, Shell: ${#SH_FILES[@]} files)"
    echo "  Safe to commit + push."
    exit 0
else
    echo "  FAILURES: Python=$PY_FAIL, Shell=$SH_FAIL"
    exit 1
fi
