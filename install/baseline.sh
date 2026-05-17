#!/usr/bin/env bash
# Baseline LLaVA-1.5 install (no pruning).
set -euo pipefail
bash install/_common.sh

if [ ! -d /content/LLaVA ]; then
  git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
fi

# LLaVA's pyproject pins torch==2.1.2 / torchvision==0.16.2 / deepspeed==0.12.6
# which conflict with Colab's torch 2.10+cu128. Relax the pins in-place and
# install with --no-deps to avoid pip pulling incompatible accelerate/gradio/
# sentencepiece/timm from the fork's metadata.
sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' /content/LLaVA/pyproject.toml
pip install -e /content/LLaVA --no-deps

pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# lmms-eval install may bump transformers/tokenizers/huggingface-hub to versions
# that break LLaVA 1.5. Force-pin to the verified-working set.
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==1.11.0
