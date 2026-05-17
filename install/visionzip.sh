#!/usr/bin/env bash
# VisionZip install — pre-LLM token pruning via CLIP attention.
set -euo pipefail
bash install/_common.sh

if [ ! -d /content/VisionZip ]; then
  git clone --depth 1 https://github.com/dvlab-research/VisionZip.git /content/VisionZip
fi

# VisionZip ships its own LLaVA fork at /content/VisionZip/LLaVA (newer
# upstream layout). Fall back to haotian-liu LLaVA if not present.
if [ -d /content/VisionZip/LLaVA ]; then
  sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' /content/VisionZip/LLaVA/pyproject.toml || true
  pip install -e /content/VisionZip/LLaVA --no-deps
elif [ ! -d /content/LLaVA ]; then
  git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
  sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' /content/LLaVA/pyproject.toml
  pip install -e /content/LLaVA --no-deps
fi

# VisionZip itself
pip install -e /content/VisionZip --no-deps

pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Force-pin transformers/tokenizers — see baseline.sh rationale
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==1.11.0
