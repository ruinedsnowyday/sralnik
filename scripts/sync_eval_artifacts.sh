#!/usr/bin/env bash
# Sync compact run/eval/artifact outputs to S3.
#
# Usage:
#   S3_ROOT=s3://sralnik-runs-213128717646 bash scripts/sync_eval_artifacts.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/data/sralnik/repo}"
S3_ROOT="${S3_ROOT:-s3://sralnik-runs-213128717646}"

cd "$REPO_DIR"

mkdir -p runs eval artifacts

aws s3 sync runs/ "$S3_ROOT/runs/" \
  --size-only \
  --exclude "*" \
  --include "*/last.pt" \
  --include "*/run_meta.txt" \
  --include "*/metrics.jsonl" \
  --include "*/stdout.log" \
  --include "*/rollout_eval.log" \
  --only-show-errors

aws s3 sync eval/ "$S3_ROOT/eval/" --size-only --only-show-errors
aws s3 sync artifacts/ "$S3_ROOT/artifacts/" --size-only --only-show-errors

aws s3 ls --recursive --human-readable "$S3_ROOT/" > /tmp/sralnik_eval_inventory.txt
aws s3 cp /tmp/sralnik_eval_inventory.txt "$S3_ROOT/eval_inventory.txt" --only-show-errors

echo "synced eval artifacts to $S3_ROOT"
