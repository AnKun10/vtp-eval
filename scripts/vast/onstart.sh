#!/bin/bash
# Vast.ai onstart for LearnPruner Figure 3 reproduction via vtp-eval.
#
# Paste the ENTIRE contents of this file into the "On-start Script" box on
# https://cloud.vast.ai/create/. Idempotent: safe to re-run on every boot.
#
# What it does:
#   1. Clones (or pulls) AnKun10/vtp-eval into /workspace/vtp-eval
#   2. Runs install/figure3.sh (pip deps + pip install -e .)
#   3. Persists HF cache + auto-cd-on-login for SSH sessions
#   4. Sanity-checks the GPU
set -euo pipefail

LOG=/workspace/onstart.log
mkdir -p /workspace
exec > >(tee -a "$LOG") 2>&1
echo "=== onstart started: $(date -Iseconds) ==="

export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"

if [ ! -d /workspace/vtp-eval ]; then
    git clone https://github.com/AnKun10/vtp-eval.git /workspace/vtp-eval
fi
cd /workspace/vtp-eval
git pull --ff-only origin master || echo "[warn] git pull skipped"

bash install/figure3.sh

grep -q 'HF_HOME=/workspace/.cache/huggingface' /root/.bashrc 2>/dev/null \
    || echo 'export HF_HOME=/workspace/.cache/huggingface' >> /root/.bashrc
grep -q 'cd /workspace/vtp-eval' /root/.bashrc 2>/dev/null \
    || echo 'cd /workspace/vtp-eval' >> /root/.bashrc

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"
python -c "import vtp_eval.insight.text_visual_attention as tva; print('text_visual_attention package OK')"

echo "=== onstart finished: $(date -Iseconds) ==="
