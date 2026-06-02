#!/usr/bin/env bash
# Run ONE config run on ONE task. Usage: bash scripts/eval/run.sh <run_name> [task] [limit]
set -euo pipefail
RUN_NAME=${1:?usage: $0 <run_name> [task] [limit]}
CONFIG=${CONFIG:-configs/eval.yaml}
TASK=${2:-}
LIMIT=${3:-}

# Resolve model/batch/task/model_args from YAML first (need the resolved task
# id for the output dir). Unit-separator join lets values contain commas.
IFS=$'\x1f' read -r MODEL BATCH TASK_CFG MODEL_ARGS < <(python - "$CONFIG" "$RUN_NAME" "$TASK" <<'PY'
import sys, yaml
cfg_path, run_name, task_cli = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(cfg_path))
run = next(r for r in cfg["runs"] if r["name"] == run_name)
args = {**cfg.get("common_args", {}), **run["model_args"], "pretrained": cfg["model_base"]}
task = task_cli or (cfg.get("tasks") or ["pope"])[0]
batch = run.get("batch_size", cfg.get("batch_size", 1))
print("\x1f".join([run["model"], str(batch), task,
                   ",".join(f"{k}={v}" for k, v in args.items())]))
PY
)

OUT_DIR="results/$RUN_NAME/$TASK_CFG"          # nested: one cell per (run, task)
mkdir -p "$OUT_DIR"
if [ -f "$OUT_DIR/results.json" ]; then
  echo "[skip] $OUT_DIR/results.json exists. Delete to re-run."; exit 0
fi

MODEL_ARGS="$MODEL_ARGS,timing_sidecar=$OUT_DIR/timing_raw.json"
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "[run] $RUN_NAME — $MODEL (bs=$BATCH) on $TASK_CFG"
python -m vtp_eval.eval.run_lmms \
  --model "$MODEL" --model_args "$MODEL_ARGS" \
  --tasks "$TASK_CFG" --batch_size "$BATCH" \
  --log_samples --log_samples_suffix "${RUN_NAME}_${TASK_CFG}" \
  --output_path "$OUT_DIR" $LIMIT_ARG 2>&1 | tee "$OUT_DIR/run.log"

# lmms-eval v0.5 nests <model>/<timestamp>_results.json under --output_path.
if [ ! -f "$OUT_DIR/results.json" ]; then
  FOUND=$(find "$OUT_DIR" -name '*results.json' | sort | head -1 || true)
  [ -n "$FOUND" ] && cp "$FOUND" "$OUT_DIR/results.json"
fi

python -m vtp_eval.eval.report parse-sidecar \
  --sidecar "$OUT_DIR/timing_raw.json" --output "$OUT_DIR/timing.json"
echo "[done] $RUN_NAME × $TASK_CFG — see $OUT_DIR"
