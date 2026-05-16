#!/usr/bin/env bash
# Shared setup run by every per-method install script.
# Idempotent: skips git clone if dir exists; pip install -q is safe to re-run.
set -euo pipefail

cd /content

if [ ! -d lmms-eval ]; then
  git clone --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
fi

pip install -q accelerate==0.34.2 datasets==2.21.0 \
               sentencepiece protobuf==3.20.3 \
               pyyaml pandas matplotlib
