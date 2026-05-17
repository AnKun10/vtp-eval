#!/usr/bin/env bash
# SparseVILA install — encoder-side prune + decode-time retrieval.
set -euo pipefail
bash install/_common.sh

if [ ! -d /content/SparseVILA ]; then
  git clone --depth 1 https://github.com/AnKun10/sparsevila-implementation.git /content/SparseVILA
fi

cd /content/SparseVILA
git submodule update --init --recursive

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
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==1.11.0
