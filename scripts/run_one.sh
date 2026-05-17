#!/usr/bin/env bash
# Run one method on POPE.
#
# Usage:
#   bash scripts/run_one.sh <run_name> [config_path] [task] [limit]
#
# Example:
#   bash scripts/run_one.sh baseline configs/methods.yaml pope 100
set -euo pipefail

RUN_NAME=${1:?usage: $0 <run_name> [config] [task] [limit]}
CONFIG=${2:-configs/methods.yaml}
TASK=${3:-pope}
LIMIT=${4:-}

OUT_DIR="results/$RUN_NAME"
mkdir -p "$OUT_DIR"

# Skip if already complete
if [ -f "$OUT_DIR/results.json" ]; then
  echo "[skip] $OUT_DIR/results.json exists. Delete to re-run."
  exit 0
fi

# Extract model, per-run batch_size override (if any), and model_args from YAML.
# Use ASCII unit-separator (0x1F) so model_args containing commas doesn't break read.
IFS=$'\x1f' read -r MODEL RUN_BATCH_SIZE MODEL_ARGS_RAW < <(python - <<PYEOF
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
run = next(r for r in cfg["runs"] if r["name"] == "$RUN_NAME")
args = run["model_args"]
common = cfg.get("common_args", {})
all_args = {**common, **args, "pretrained": cfg["model_base"]}
batch = run.get("batch_size", "")  # empty string → use VTP_BATCH_SIZE fallback
parts = [run["model"], str(batch), ",".join(f"{k}={v}" for k, v in all_args.items())]
print("\x1f".join(parts))
PYEOF
)

MODEL_ARGS="$MODEL_ARGS_RAW,timing_sidecar=$OUT_DIR/timing_raw.json"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
# Per-run batch_size wins over env override wins over default (2)
BATCH_SIZE=${RUN_BATCH_SIZE:-${VTP_BATCH_SIZE:-2}}

echo "[run] $RUN_NAME — $MODEL(batch=$BATCH_SIZE; $MODEL_ARGS) on $TASK"
# Use vtp_eval.run_lmms wrapper instead of `python -m lmms_eval` directly:
# the wrapper imports vtp_eval.adapters first, which mutates lmms-eval's
# AVAILABLE_SIMPLE_MODELS so our adapter names resolve. (--include_path alone
# is insufficient; lmms-eval doesn't auto-import packages from that path.)
python -m vtp_eval.run_lmms \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size "$BATCH_SIZE" \
  --log_samples --log_samples_suffix "$RUN_NAME" \
  --output_path "$OUT_DIR" \
  $LIMIT_ARG \
  2>&1 | tee "$OUT_DIR/run.log"

# Build timing.json from sidecar (pruning_meta is bundled inside)
python -m vtp_eval.utils.timing parse-sidecar \
  --sidecar "$OUT_DIR/timing_raw.json" \
  --output "$OUT_DIR/timing.json"

echo "[done] $RUN_NAME — see $OUT_DIR"
