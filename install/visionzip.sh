#!/usr/bin/env bash
# VisionZip install — pre-LLM token pruning via CLIP attention.
set -euo pipefail
bash install/_common.sh

# Drop any prior method's editable `llava` finder. After this, llava is
# UN-installed regardless of whether /content/LLaVA still exists on disk —
# the conditional below must always reinstall.
pip uninstall -y llava || true

if [ ! -d /content/VisionZip ]; then
  git clone --depth 1 https://github.com/dvlab-research/VisionZip.git /content/VisionZip
fi

# Pick which LLaVA fork to use. VisionZip ships /content/VisionZip/LLaVA on
# some commits and not others; fall back to haotian-liu when missing.
# IMPORTANT — decouple "which dir" from "install":  the install step must run
# unconditionally because `pip uninstall -y llava` above removed the previous
# method's editable record. The earlier `elif [ ! -d /content/LLaVA ]` form
# silently skipped the install when /content/LLaVA was left on disk by a
# previous method, surfacing as `NameError: get_model_name_from_path is not
# defined` inside lmms-eval. See docs/COLAB_FIXES.md root cause #19.
if [ -d /content/VisionZip/LLaVA ]; then
  LLAVA_DIR=/content/VisionZip/LLaVA
else
  [ -d /content/LLaVA ] || git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
  LLAVA_DIR=/content/LLaVA
fi
sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' $LLAVA_DIR/pyproject.toml || true
pip install -e $LLAVA_DIR --no-deps

# VisionZip itself
pip install -e /content/VisionZip --no-deps

pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Force-pin transformers/tokenizers — see baseline.sh rationale
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==0.24.7
