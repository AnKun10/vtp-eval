#!/usr/bin/env bash
# Shared setup run by every per-method install script.
# Idempotent: skips git clone if dir exists; pip install -q is safe to re-run.
set -euo pipefail

cd /content

# Pin lmms-eval to v0.5 (2025-10-07) — last tag before MODEL_REGISTRY_V2
# (commit b83e3ec4, 2026-02-10) replaced the AVAILABLE_SIMPLE_MODELS path
# our adapters depend on.
if [ ! -d lmms-eval ]; then
  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
  (cd lmms-eval && git checkout v0.5)
fi

# protobuf 3.20.3 pin was removed: it breaks transformers 4.37.2 CLIP image
# processor ("cannot import name 'runtime_version' from 'google.protobuf'").
# Let transformers pick its own 4.x+ version (tested with 5.29.6).
# sqlitedict is required by lmms-eval v0.5.
pip install -q accelerate==0.34.2 datasets==2.21.0 \
               sentencepiece sqlitedict \
               pyyaml pandas matplotlib
