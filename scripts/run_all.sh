#!/usr/bin/env bash
# Run all 6 methods sequentially WITHOUT process restart.
# WARNING: Only works on a machine where every method's transformers/llava
# packages are isolated in separate conda envs the caller switches between,
# or where all methods happen to share compatible deps (rare).
# On Colab use notebooks/pope_eval.ipynb instead.
set -euo pipefail

CONFIG=${1:-configs/methods.yaml}
TASK=${2:-pope}
LIMIT=${3:-}

RUNS=$(python - <<PYEOF
import yaml
print("\n".join(r["name"] for r in yaml.safe_load(open("$CONFIG"))["runs"]))
PYEOF
)

while IFS= read -r RUN; do
  echo "==================== $RUN ===================="
  bash scripts/run_one.sh "$RUN" "$CONFIG" "$TASK" "$LIMIT"
done <<< "$RUNS"

python -m vtp_eval.utils.result_io aggregate results/ \
  --output results/summary.csv
echo "[all done] results/summary.csv"
