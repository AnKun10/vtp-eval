#!/usr/bin/env bash
# FastV install — token pruning AFTER layer K based on attention rank.
#
# ⚠️ STATUS: BLOCKED on Colab Python 3.12 (as of 2026-05-17).
# FastV's bundled transformers fork is pinned to 4.31.0, whose
# dependency_versions_check requires tokenizers >=0.11.1, !=0.11.3, <0.14.
# No pre-built wheel of tokenizers <0.14 exists for Python 3.12, and source
# build fails on Colab. Stock transformers 4.37.2 doesn't work either: FastV's
# modeling_llama.py patch assumes the older eager-attention output is a
# tensor, but it's a tuple in 4.37.2 — the patch can't be ported cleanly.
#
# This script is kept for documentation; it will fail on Colab today.
# To run FastV you'd need a Python 3.10 kernel or a maintained FastV branch.
# See docs/COLAB_FIXES.md root cause #18.
#
# Plan-of-record: install FastV's bundled transformers fork at
# /content/FastV/src/transformers/ AS the transformers package. This is
# isolating: FastV must be in its own kernel restart (the notebook already
# enforces this between methods).
set -euo pipefail
bash install/_common.sh

# Drop any prior method's editable `llava` finder.
pip uninstall -y llava || true

if [ ! -d /content/FastV ]; then
  git clone --depth 1 https://github.com/pkunlp-icler/FastV.git /content/FastV
fi

# Install FastV's bundled transformers fork (replaces stock transformers
# in this kernel). FastV's modeling_llama.py inside this fork is consistent
# with the rest of the transformers package; the single-file copy approach
# could not be made consistent.
if [ -d /content/FastV/src/transformers ]; then
  pip install -q --no-deps /content/FastV/src/transformers
else
  echo "ERROR: FastV bundled transformers not at /content/FastV/src/transformers" >&2
  echo "Repo layout may have changed; check https://github.com/pkunlp-icler/FastV" >&2
  exit 1
fi

# Install FastV's LLaVA fork.
LLAVA_DIR=/content/FastV/src/LLaVA
if [ -f $LLAVA_DIR/pyproject.toml ]; then
  sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' $LLAVA_DIR/pyproject.toml
  pip install -e $LLAVA_DIR --no-deps
fi

pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Do NOT force-reinstall transformers here — we want FastV's bundled fork
# to stay in place. tokenizers/huggingface_hub still pinned for LLaVA 1.5.
pip install -q --force-reinstall --no-deps \
    tokenizers==0.15.1 huggingface_hub==0.24.7
