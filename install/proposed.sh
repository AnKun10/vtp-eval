#!/usr/bin/env bash
# Proposed method install — patches the original LLaVA repo at runtime via
# vtp_eval.proposed_method.proposed_prune (no extra third-party package).
# Same base environment as VisionZip: original LLaVA + transformers 4.37.2.
set -euo pipefail
bash install/_common.sh

# Drop any prior method's editable `llava` finder, then (re)install LLaVA.
pip uninstall -y llava || true
[ -d /content/LLaVA ] || git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' /content/LLaVA/pyproject.toml || true
pip install -e /content/LLaVA --no-deps

pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Same pin as VisionZip — Stage 2's LlamaModel.forward port targets 4.37.2.
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==0.24.7
