#!/usr/bin/env bash
# Run every config run on every task in the config, then aggregate to summary.csv.
# Usage: bash scripts/eval/run_all.sh [limit]
# Resumable: each (run, task) cell skips if its results.json already exists.
set -euo pipefail
CONFIG=${CONFIG:-configs/eval.yaml}
LIMIT=${1:-}
TASKS=$(python - "$CONFIG" <<'PY'
import sys, yaml
print("\n".join(yaml.safe_load(open(sys.argv[1])).get("tasks", ["pope"])))
PY
)
RUNS=$(python - "$CONFIG" <<'PY'
import sys, yaml
print("\n".join(r["name"] for r in yaml.safe_load(open(sys.argv[1]))["runs"]))
PY
)
while IFS= read -r TASK; do
  [ -z "$TASK" ] && continue
  while IFS= read -r RUN; do
    [ -z "$RUN" ] && continue
    echo "==================== $RUN × $TASK ===================="
    CONFIG="$CONFIG" bash scripts/eval/run.sh "$RUN" "$TASK" "$LIMIT" \
      || echo "[warn] $RUN × $TASK failed — continuing sweep (re-run to retry this cell)"
  done <<< "$RUNS"
done <<< "$TASKS"
python -m vtp_eval.eval.report aggregate results/ --output results/summary.csv
echo "[all done] results/summary.csv"
