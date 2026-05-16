#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/SparseVLMs ]; then
  git clone --depth 1 https://github.com/Gumpest/SparseVLMs.git /content/SparseVLMs
fi
pip install -e /content/SparseVLMs   # installs SparseVLMs' LLaVA fork
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
