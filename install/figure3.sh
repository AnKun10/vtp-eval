#!/usr/bin/env bash
# Install deps for Figure 3 of LearnPruner reproduction.
# Idempotent: pip skips satisfied requirements.
#
# Run from the repo root:
#     bash install/figure3.sh
set -euo pipefail

pip install --no-cache-dir \
    "transformers==4.49.0" \
    "accelerate>=0.30" \
    "pillow>=10" \
    "matplotlib>=3.8" \
    "pandas>=2.0" \
    "pyyaml" \
    "jupyterlab>=4"

# Install vtp-eval as editable so `python -m vtp_eval.figure3` resolves.
# --no-deps because the install above already pinned the heavy ones; we don't
# want pip resolving the broader vtp-eval extras here.
pip install -e . --no-deps
