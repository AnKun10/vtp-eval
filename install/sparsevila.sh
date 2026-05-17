#!/usr/bin/env bash
# SparseVILA install — encoder-side prune + decode-time retrieval.
set -euo pipefail
bash install/_common.sh

# Drop any prior method's editable `llava` finder.
pip uninstall -y llava || true

if [ ! -d /content/SparseVILA ]; then
  git clone --depth 1 https://github.com/AnKun10/sparsevila-implementation.git /content/SparseVILA
fi

cd /content/SparseVILA
git submodule update --init --recursive

# SparseVILA bundles a LLaVA submodule at third_party/LLaVA that its loader
# (`from llava.model.builder import load_pretrained_model`) requires at
# import time. Install it explicitly — submodule fetch alone doesn't register
# the package.
LLAVA_DIR=/content/SparseVILA/third_party/LLaVA
if [ -f $LLAVA_DIR/pyproject.toml ]; then
  sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' $LLAVA_DIR/pyproject.toml
  pip install -e $LLAVA_DIR --no-deps
fi

# --no-deps everywhere to avoid pulling conflicting accelerate/timm/gradio
# pins from the fork metadata. SparseVILA's own pyproject pins transformers
# 4.37.2 and tokenizers 0.15.1, which we want anyway.
pip install -e ./flash-colreduce --no-deps
pip install -e . --no-deps

cd /content
pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Force-pin to the verified-working transformers stack
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==0.24.7
