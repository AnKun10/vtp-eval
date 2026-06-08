#!/bin/bash
# Vast.ai on-start for the vtp-eval proposed-method DEMO + insight tools.
#
# Paste the ENTIRE contents of this file into the template's "On-start Script"
# box on https://cloud.vast.ai/create/ (see docs/vast_proposed_method.md).
# Idempotent: safe to re-run on every boot. On-start runs ONCE at first boot;
# editing it on a running instance does not re-run — Destroy and rent fresh, or
# re-run this file manually: `bash /workspace/vtp-eval/scripts/vast/onstart_proposed.sh`.
#
# Builds ONE venv (liuhaotian LLaVA + transformers 4.37.2) that runs the demo AND
# prune_viz on the same stack. After boot, launch an app:
#     bash scripts/run_app.sh demo        # or: prune_viz
set -uo pipefail   # NOT -e: a single non-fatal step shouldn't abort the boot
mkdir -p /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1
echo "=== onstart (proposed-method demo) $(date -Iseconds) ==="

export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"

REPO=/workspace/vtp-eval
[ -d "$REPO" ] || git clone -b proposed-method https://github.com/AnKun10/vtp-eval.git "$REPO"
cd "$REPO"
git fetch origin --prune && git checkout proposed-method \
    && git pull --ff-only origin proposed-method || echo "[warn] git update skipped"

# Pick the Python env: reuse the base image's torch env if it already ships a
# compatible CUDA torch (2.1/2.2) — e.g. the 2.1.2-cuda-12.1.1-py310 tag — so
# install skips the slow ~757 MB torch download; otherwise build a clean venv.
if /venv/main/bin/python -c "import torch,sys; sys.exit(0 if (torch.__version__.startswith(('2.1.','2.2.')) and torch.version.cuda) else 1)" 2>/dev/null; then
    VENV=/venv/main; echo "[onstart] reusing base image torch env: $VENV"
else
    python3 -m venv /workspace/venv; VENV=/workspace/venv; echo "[onstart] built fresh venv: $VENV"
fi
echo "$VENV" > /workspace/.venv          # scripts/run_app.sh reads this
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel setuptools

# Core method stack (llava + transformers 4.37.2). install/proposed.sh's lmms
# block also installs gradio + datasets + matplotlib under a constraints file
# that keeps huggingface_hub/datasets compatible with transformers 4.37.2 — the
# demo (UI + benchmark streaming + figures) needs all of them, so keep it on.
WORKSPACE=/workspace bash install/proposed.sh
pip install -q pytest "numpy<2" "pandas>=2.0" "matplotlib>=3.7"

# Auto-activate the env + cd on interactive SSH login.
grep -q "source $VENV/bin/activate" /root/.bashrc 2>/dev/null || {
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> /root/.bashrc
    echo "source $VENV/bin/activate"                     >> /root/.bashrc
    echo 'cd /workspace/vtp-eval'                        >> /root/.bashrc
}

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"
python -c "import transformers; print('transformers', transformers.__version__)"   # expect 4.37.2
python -c "import gradio, datasets, pandas, matplotlib; print('UI/data deps OK; gradio', gradio.__version__)"
python -c "import llava; from vtp_eval.proposed_method import proposed_prune; print('proposed_method import OK')"
python -c "import vtp_eval.demo.tva; from vtp_eval.demo.engine import load_engine; print('unified demo import OK')"
echo "[onstart] ready. Launch an app:  bash scripts/run_app.sh demo   (or prune_viz)"
echo "=== onstart finished: $(date -Iseconds) ==="
