#!/usr/bin/env bash
# Run every config run on the default task, then aggregate to summary.csv.
set -euo pipefail
CONFIG=${CONFIG:-configs/eval.yaml}
TASK=${1:-}
LIMIT=${2:-}
RUNS=$(python - "$CONFIG" <<'PY'
import sys, yaml
print("\n".join(r["name"] for r in yaml.safe_load(open(sys.argv[1]))["runs"]))
PY
)
while IFS= read -r RUN; do
  echo "==================== $RUN ===================="
  CONFIG="$CONFIG" bash scripts/eval/run.sh "$RUN" "$TASK" "$LIMIT"
done <<< "$RUNS"
python -m vtp_eval.eval.report aggregate results/ --output results/summary.csv
echo "[all done] results/summary.csv"
