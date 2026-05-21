#!/usr/bin/env bash
# Install deps for Figure 3 of LearnPruner reproduction.
# Idempotent: pip skips satisfied requirements.
#
# Run from the repo root:
#     bash install/figure3.sh
set -euo pipefail

# 1. Some PyTorch base images (vastai/pytorch:latest) ship only `python3`.
#    Our wrappers (figure3_ui.sh, figure3_run.sh, figure3_list_samples.sh,
#    figure3_vast_onstart.sh) all call `python`. Create the symlink if absent.
if ! command -v python >/dev/null 2>&1; then
    ln -sf "$(command -v python3)" /usr/local/bin/python
    echo "[install/figure3] Created python -> python3 symlink"
fi

# 2. The base image may ship torch built for a CUDA toolkit newer than the
#    host driver supports (e.g. torch+cu130 on a CUDA-12.8 driver), which makes
#    torch.cuda.is_available() return False. Force the broadly-compatible
#    cu121 build that works on any host with driver >= 525.60 (CUDA 12.0+).
pip uninstall -y torch torchvision triton 2>/dev/null || true
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.1" "torchvision==0.19.1"

# 3. Everything else. gradio>=5 (not 4.x) because 4.44.x trips on a
#    pydantic 2.13+ schema bug ("argument of type 'bool' is not iterable")
#    that prevents demo.launch() from starting.
pip install --no-cache-dir \
    "transformers==4.49.0" \
    "accelerate>=0.30" \
    "pillow>=10" \
    "matplotlib>=3.8" \
    "pandas>=2.0" \
    "pyyaml" \
    "jupyterlab>=4" \
    "gradio>=5.0,<6"

# 4. Install vtp-eval as editable so `python -m vtp_eval.figure3` resolves.
# --no-deps because the install above already pinned the heavy ones; we don't
# want pip resolving the broader vtp-eval extras here.
pip install -e . --no-deps
