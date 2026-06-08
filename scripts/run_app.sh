#!/usr/bin/env bash
# Launch ONE vtp-eval Gradio app, freeing the GPU first.
#
# Only one LLaVA-1.5-7B (~14 GB) fits on the 22 GB GPU, so any running app is
# killed before the requested one starts (run them one at a time / alternate).
#
# Usage (on the Vast.ai box, from anywhere):
#   bash scripts/run_app.sh demo        # proposed-method demo (public --share link)
#   bash scripts/run_app.sh prune_viz   # token-pruning visualization (public --share)
#   bash scripts/run_app.sh stop        # stop everything, free the GPU
#
# `tva` (standalone text-visual-attention) runs on the LEGACY llava-hf /
# transformers-4.49 stack and does NOT work on this 4.37.2 instance — its
# functionality is being folded into the demo as the unified 3-section UI.
#
# The venv is auto-detected from /workspace/.venv (written by the on-start), with
# a fallback search. Override with `VENV=/path bash scripts/run_app.sh demo`.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/vtp-eval}"
LOG="$WORKSPACE/app_ui.log"

VENV="${VENV:-}"
[ -z "$VENV" ] && [ -f /workspace/.venv ] && VENV="$(cat /workspace/.venv)"
if [ -z "$VENV" ] || [ ! -f "$VENV/bin/activate" ]; then
    for c in /workspace/venv /venv/main /workspace/sparsevenv; do
        [ -f "$c/bin/activate" ] && { VENV="$c"; break; }
    done
fi
[ -f "$VENV/bin/activate" ] || { echo "No venv found (set VENV=...)"; exit 3; }

stop_all() {
    pkill -9 -f "vtp_eval.demo.app"                          2>/dev/null || true
    pkill -9 -f "vtp_eval.insight.prune_viz.ui"             2>/dev/null || true
    pkill -9 -f "vtp_eval.insight.text_visual_attention.ui" 2>/dev/null || true
    sleep 3
}

start() {   # $1 = python module   $2 = port
    : > "$LOG"
    setsid bash -c "cd '$REPO' && source '$VENV/bin/activate' && \
        python -m $1 --port $2 --share > '$LOG' 2>&1" < /dev/null &
    echo "Starting $1 on port $2 (venv $VENV) — waiting for the public link..."
    for _ in $(seq 1 90); do
        if grep -q "gradio.live" "$LOG" 2>/dev/null; then
            echo "READY: $(grep -m1 -o 'https://[a-z0-9]*\.gradio\.live' "$LOG")"
            return 0
        fi
        if grep -qiE "traceback|out of memory|address already in use" "$LOG" 2>/dev/null; then
            echo "ERROR starting $1 — tail of $LOG:"; tail -n 8 "$LOG"; return 1
        fi
        sleep 2
    done
    echo "Timed out waiting for the link; tail of $LOG:"; tail -n 8 "$LOG"; return 1
}

case "${1:-}" in
    demo)      stop_all; start "vtp_eval.demo.app"            7860 ;;
    prune_viz) stop_all; start "vtp_eval.insight.prune_viz.ui" 7861 ;;
    tva)       echo "tva standalone needs the legacy llava-hf/transformers-4.49 stack;"
               echo "it does not run on this 4.37.2 instance. TVA is folded into the"
               echo "demo via the unified UI (Spec B)." ;;
    stop)      stop_all; echo "All apps stopped; GPU freed." ;;
    *) echo "Usage: bash scripts/run_app.sh {demo|prune_viz|stop}"; exit 2 ;;
esac
