#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/SparseVILA ]; then
  git clone --depth 1 https://github.com/AnKun10/sparsevila-implementation.git /content/SparseVILA
fi
cd /content/SparseVILA
git submodule update --init --recursive
pip install -e ./flash-colreduce
pip install -e .
cd /content
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
