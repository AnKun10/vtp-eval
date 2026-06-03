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

# 2. Force a torch build matching the host driver (cu121 works on any driver >=
#    525 / CUDA 12.0+, including CUDA 13 hosts — NVIDIA drivers are backward
#    compatible). Pick the torch version by the active Python: 2.1.2 (LLaVA's
#    target) for py<=3.11, 2.2.2 for py3.12 (2.1.2 has no cp312 wheels).
PYV=$(python -c 'import sys;print(sys.version_info.major*100+sys.version_info.minor)')
if [ "$PYV" -ge 312 ]; then
    TORCH_PKG="torch==2.2.2"; TV_PKG="torchvision==0.17.2"
else
    TORCH_PKG="torch==2.1.2"; TV_PKG="torchvision==0.16.2"
fi
# Skip the slow (~757 MB) reinstall if a compatible CUDA torch is already in
# this env — e.g. on a Stop/Start reboot where the venv persisted, or on a base
# image that already ships torch 2.1/2.2 + CUDA. Only download when needed.
if python -c "import torch,sys; sys.exit(0 if (torch.__version__.startswith(('2.1.','2.2.')) and torch.version.cuda) else 1)" 2>/dev/null; then
    echo "[install/proposed] compatible torch present ($(python -c 'import torch;print(torch.__version__)')) — skipping torch reinstall"
else
    echo "[install/proposed] python=$PYV -> installing $TORCH_PKG (cu121)"
    pip uninstall -y torch torchvision triton 2>/dev/null || true
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        "$TORCH_PKG" "$TV_PKG"
fi

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
    "accelerate>=0.21,<0.27" "sentencepiece" "protobuf" "pillow>=10" "einops" "pyyaml" \
    "regex" "safetensors"   # regex/safetensors are transformers runtime deps (--no-deps skipped them)

# 5. Install vtp-eval editable so vtp_eval.proposed_method resolves.
pip install -e . --no-deps

# 6. lmms-eval benchmark harness (pin v0.5 — compatible with transformers 4.37.2).
#    --no-deps so it doesn't move our torch/transformers pins; add its runtime deps.
#    Toggle off with INSTALL_LMMS=0 when only the method (not eval) is needed.
if [ "${INSTALL_LMMS:-1}" = "1" ]; then
    LMMS_DIR="$WORKSPACE/lmms-eval"
    [ -d "$LMMS_DIR" ] || git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "$LMMS_DIR"
    # Pin to v0.5. A shallow on-start clone may not have the v0.5 tag as a local
    # ref (its default branch already IS the v0.5 release commit), so a failed
    # checkout must NOT abort the whole install under `set -e` — fetch tags, then
    # checkout, and fall through if the tag still isn't resolvable.
    ( cd "$LMMS_DIR" && git fetch --tags --quiet 2>/dev/null || true
      git checkout v0.5 2>/dev/null \
        || echo "[install/proposed] v0.5 tag not a local ref — assuming clone is already at the v0.5 commit" )
    # These tasks hardcode an HF auth token, forcing datasets to require a cached
    # token even though the lmms-lab/* datasets are public. It appears two ways:
    # in the task yaml (`token: True`) AND in Python (`token=True` in utils.py,
    # e.g. gqa loads its images config there). Flip both to False so everything
    # loads anonymously (no HF login needed on a fresh instance).
    for d in pope gqa textvqa mme mmbench scienceqa vizwiz_vqa ocrbench; do
        sed -i 's/token: True/token: False/g; s/token: true/token: false/g' \
            "$LMMS_DIR"/lmms_eval/tasks/$d/*.yaml 2>/dev/null || true
        sed -i 's/token=True/token=False/g' \
            "$LMMS_DIR"/lmms_eval/tasks/$d/*.py 2>/dev/null || true
    done
    pip install -e "$LMMS_DIR" --no-deps
    # lmms-eval v0.5 pulls transformers>=4.39 / accelerate>=0.29 — those would break
    # our 4.37.2 Stage-2 port, so we install its runtime deps under a constraints
    # file that pins the versions the proposed method needs.
    CONSTRAINTS=$(mktemp)
    cat > "$CONSTRAINTS" <<C
torch==2.1.2
torchvision==0.16.2
transformers==4.37.2
tokenizers==0.15.1
accelerate==0.26.1
numpy==1.26.4
huggingface_hub==0.36.2
datasets==2.20.0
C
    # huggingface_hub MUST stay <1.0 (transformers 4.37.2 / tokenizers 0.15.1
    # hard-require it) and datasets <2.21 — newer `datasets`/`evaluate` otherwise
    # drag in huggingface_hub>=1.0 and break the import. Pin both explicitly.
    pip install --no-cache-dir -c "$CONSTRAINTS" \
        "huggingface_hub==0.36.2" "datasets==2.20.0" \
        sqlitedict tenacity pytablewriter sacrebleu evaluate \
        hf_transfer loguru openai jsonlines numexpr peft scikit-learn ftfy \
        opencv-python-headless nltk tqdm-multiprocess zstandard sympy mpmath \
        openpyxl tiktoken pydantic python-dotenv timm jinja2 protobuf
    # Belt-and-suspenders: if a transitive dep still bumped it, force it back.
    python -c "import importlib.metadata as m,sys; sys.exit(0 if m.version('huggingface_hub').startswith('0.') else 1)" \
        || pip install --no-cache-dir --force-reinstall --no-deps "huggingface_hub==0.36.2"
fi

echo "[install/proposed] done. transformers=$(python -c 'import transformers;print(transformers.__version__)')"
