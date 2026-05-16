#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.40.0 tokenizers==0.19.1
if [ ! -d /content/VisionZip ]; then
  git clone --depth 1 https://github.com/dvlab-research/VisionZip.git /content/VisionZip
fi
pip install -e /content/VisionZip
# VisionZip's LLaVA fork (if shipped) or standalone llava
if [ -d /content/VisionZip/LLaVA ]; then
  pip install -e /content/VisionZip/LLaVA
elif [ ! -d /content/LLaVA ]; then
  git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
  pip install -e /content/LLaVA
fi
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
