# scripts/eval/run_sparsevlm_native.sh
#!/usr/bin/env bash
# Run SparseVLM in its own repo at a given RETAIN_TOKN on one task, then copy its
# results.json into our results/ tree so report.py can aggregate it.
# Usage: bash scripts/eval/run_sparsevlm_native.sh <retain_tokn> <task> <run_name>
set -euo pipefail
RETAIN=${1:?retain_tokn}; TASK=${2:?task}; RUN=${3:?run_name}
SPARSE_DIR="${SPARSE_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)/SparseVLMs}"
OUT_DIR="results/$RUN/$TASK"; mkdir -p "$OUT_DIR"
( cd "$SPARSE_DIR" && RETAIN_TOKN="$RETAIN" bash scripts/v1_5/eval/"$TASK".sh ) \
    | tee "$OUT_DIR/run.log"
FOUND=$(find "$SPARSE_DIR" -name '*.json' -newer "$OUT_DIR/run.log" | head -1 || true)
[ -n "$FOUND" ] && cp "$FOUND" "$OUT_DIR/results.json"
echo "[sparsevlm-native] $RUN x $TASK -> $OUT_DIR (flag manually as != harness)"
