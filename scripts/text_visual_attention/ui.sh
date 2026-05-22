#!/usr/bin/env bash
# Launch the Gradio UI on port 7860. Binds 0.0.0.0 so SSH tunnels reach it.
#
# Access from your laptop:
#     ssh -p <VAST_PORT> -L 7860:localhost:7860 root@<HOST>
#     # then open http://localhost:7860 in your browser
#
# Forward additional args to the underlying CLI:
#     bash scripts/figure3_ui.sh --port 8000
set -euo pipefail
python -m vtp_eval.figure3.ui "$@"
