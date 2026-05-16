#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/LLaVA ]; then
  git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
fi
pip install -e /content/LLaVA
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
