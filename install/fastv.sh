#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/FastV ]; then
  git clone --depth 1 https://github.com/pkunlp-icler/FastV.git /content/FastV
fi
python install/_patch_fastv.py
pip install -e /content/FastV/src/LLaVA
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
