#!/usr/bin/env bash
# Proposed-method install (Vast.ai). Run from the repo root:
#     bash install/proposed.sh
#
# Sets up the ORIGINAL LLaVA repo (haotian-liu) + transformers 4.37.2 so that
# vtp_eval.proposed_method.proposed_prune can patch CLIPVisionTower.forward
# (Stage 1) and LlamaModel.forward (Stage 2) at runtime. No extra third-party
# pruning package is needed. Idempotent: pip skips satisfied requirements and
# the LLaVA clone is skipped if it already exists.
#
# NOTE: lmms-eval is intentionally NOT installed here — it is only needed for
# benchmarking via the adapter, which depends on the archived eval harness
# (separate `eval/` concern). This script supports the smoke test + ad-hoc
# inference path.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
LLAVA_DIR="$WORKSPACE/LLaVA"

# 1. Some PyTorch base images (vastai/pytorch) ship only `python3`; our scripts
#    call `python`. Create the symlink if absent.
if ! command -v python >/dev/null 2>&1; then
    ln -sf "$(command -v python3)" /usr/local/bin/python
    echo "[install/proposed] Created python -> python3 symlink"
fi

# 2. Force a torch build matching the host driver (cu121 works on driver >= 525,
#    i.e. CUDA 12.0+). torch 2.1.2 is the version the original LLaVA repo targets.
pip uninstall -y torch torchvision triton 2>/dev/null || true
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.2" "torchvision==0.16.2"

# 3. Clone the original LLaVA repo and install it WITHOUT deps (we pin
#    torch/transformers ourselves). Relax LLaVA's hard version pins first.
[ -d "$LLAVA_DIR" ] || git clone --depth 1 https://github.com/haotian-liu/LLaVA.git "$LLAVA_DIR"
sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' \
    "$LLAVA_DIR/pyproject.toml" || true
pip install -e "$LLAVA_DIR" --no-deps

# 4. Pin transformers 4.37.2 (Stage 2's LlamaModel.forward port targets it) and
#    the LLaVA runtime deps that --no-deps skipped.
pip install --no-cache-dir --force-reinstall --no-deps \
    "transformers==4.37.2" "tokenizers==0.15.1" "huggingface_hub==0.24.7"
pip install --no-cache-dir \
    "accelerate>=0.21,<0.27" "sentencepiece" "protobuf" "pillow>=10" "einops" "pyyaml"

# 5. Install vtp-eval editable so vtp_eval.proposed_method resolves.
pip install -e . --no-deps

echo "[install/proposed] done. transformers=$(python -c 'import transformers;print(transformers.__version__)')"
