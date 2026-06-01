# Running the proposed pruning method on Vast.ai

Operator runbook for renting a Vast.ai GPU instance, building the **VTP-Eval**
template, and running the proposed two-stage visual-token pruning method
(`vtp_eval/proposed_method/`) on **LLaVA-1.5-7B** with the original LLaVA repo +
transformers 4.37.2.

> Method design: `docs/proposed_method.md`. Spec/plan under
> `docs/superpowers/`. This runbook is the GPU counterpart to `docs/vast.md`
> (which covers the Figure-3 *insight*, a different transformers pin).

**Estimated cost:** ~$0.30–0.60/h on a 24 GB card; a smoke test + a few
generations is well under **$0.50** total.

---

## What you can run after this

| Step | Needs |
|------|-------|
| Stage-1/2 unit tests (`-m "not slow"`) | nothing GPU — already pass locally |
| **Method end-to-end** (generation) | this runbook: rent with the on-start (install is automatic) |
| Benchmark across configs (`configs/proposed_method.yaml`) | the archived lmms-eval harness restored under `eval/` — **not yet done**, out of scope here |

So this runbook gets you to a **working generation + the `@slow` smoke test**.
Full benchmarking waits on the separate `eval/` harness restore.

---

## 0. Prerequisites (do these once, locally)

1. **Push the `proposed-method` branch to GitHub.** The on-start script fetches
   the code from `origin`, and the method currently lives only on your local
   `proposed-method` branch (origin has just `master` + `archive`). From the
   repo root:
   ```bash
   git push -u origin proposed-method
   ```
   (If you prefer not to push a WIP branch, merge to `master` first and skip the
   `git checkout proposed-method` line in the on-start script below.)

2. **Add your SSH public key to Vast:** Account → Keys → **+ Add SSH Key**,
   paste `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`).

3. **Have a Hugging Face token** (for the LLaVA-1.5-7B download). You'll set it
   as `HF_TOKEN` in the template.
   > ⚠️ **Security:** the `HF_TOKEN` value is a secret. The token visible in your
   > `Vast.ai _ Console (New).pdf` screenshot is exposed — **rotate it** at
   > <https://huggingface.co/settings/tokens> and never commit it to git.

---

## 1. Create the `VTP-Eval` template

On <https://cloud.vast.ai/create/> → **Edit/Create Template**. These are the
fields from your console screenshot, kept as-is:

| Field | Value |
|-------|-------|
| **Template Name** | `VTP-Eval` |
| **Template Description** | `PyTorch for Nvidia CUDA` |
| **Image Path:Tag** | `vastai/pytorch` |
| **Version Tag** | **`2.1.2-cuda-12.1.1-py310-ipv2`** (recommended) — ships torch 2.1.2 + CUDA 12.1 + Python 3.10, exactly our stack, so the on-start **skips the torch download**. (`[Automatic]`/py3.12 also works but re-downloads torch.) |
| **Docker Options** | `-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -e OPEN_BUTTON_PORT=1111` |
| **Ports** | `1111`, `6006`, `8080`, `8384`, `72299` (all TCP) |
| **Launch Mode** | **Jupyter-python notebook + SSH** |

**Environment Variables:**

| Key | Value |
|-----|-------|
| `OPEN_BUTTON_PORT` | `1111` |
| `OPEN_BUTTON_TOKEN` | `1` |
| `JUPYTER_DIR` | `/` |
| `DATA_DIRECTORY` | `/workspace/` |
| `PORTAL_CONFIG` | `localhost:1111:11111:/:Instance Portal` |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `HF_TOKEN` | `<your-hf-token>` |

**Extra Filters (CLI format):** `cpu_arch in ['arm64', 'amd64']`

**On-start Script:** paste the block from §2 below.

---

## 2. On-start script (paste into the template's "On-start Script" box)

**This is the on-start WRAPPER — do NOT paste `install/proposed.sh` here.**
`install/proposed.sh` is the *install step* (run from the repo root); on its own
it never clones the repo and aborts on a Python-3.12 image. This wrapper clones
the repo (checkout `proposed-method`), builds a **dedicated venv** (so versions
are deterministic and the base image's `/venv/main` is left untouched), runs
`install/proposed.sh` inside it (which auto-picks torch 2.2.2 on py3.12), and
auto-activates the venv on SSH login. **Idempotent.** The transformers-4.37.2
Stage-2 forward is already committed in the repo (§5), so no manual sync is
needed — after the on-start finishes the method is ready to run.

> On-start runs **once at first boot**. If you change it on a running instance
> it will not re-run — Destroy and rent a fresh instance.

```bash
#!/bin/bash
set -uo pipefail   # NOT -e: a single non-fatal step shouldn't abort the boot
mkdir -p /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1
echo "=== onstart (proposed-method) $(date -Iseconds) ==="

export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"

REPO=/workspace/vtp-eval
[ -d "$REPO" ] || git clone -b proposed-method https://github.com/AnKun10/vtp-eval.git "$REPO"
cd "$REPO"
git fetch origin --prune && git checkout proposed-method \
    && git pull --ff-only origin proposed-method || echo "[warn] git update skipped"

# Pick the Python env. If the base image already ships a compatible CUDA torch
# (2.1/2.2) — e.g. the 2.1.2-cuda-12.1.1-py310 tag — reuse it so install skips
# the slow ~757 MB torch download. Otherwise build a clean venv.
if /venv/main/bin/python -c "import torch,sys; sys.exit(0 if (torch.__version__.startswith(('2.1.','2.2.')) and torch.version.cuda) else 1)" 2>/dev/null; then
    VENV=/venv/main; echo "[onstart] reusing base image torch env: $VENV"
else
    python3 -m venv /workspace/venv; VENV=/workspace/venv; echo "[onstart] built fresh venv: $VENV"
fi
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel setuptools

WORKSPACE=/workspace bash install/proposed.sh   # skips torch if already compatible
pip install -q pytest "numpy<2"

# Auto-activate the chosen env + cd on interactive SSH login.
grep -q "source $VENV/bin/activate" /root/.bashrc 2>/dev/null || {
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> /root/.bashrc
    echo "source $VENV/bin/activate"                     >> /root/.bashrc
    echo 'cd /workspace/vtp-eval'                        >> /root/.bashrc
}

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"
python -c "import transformers; print('transformers', transformers.__version__)"  # expect 4.37.2
python -c "import llava; from vtp_eval.proposed_method import proposed_prune; print('proposed_method import OK')"
echo "=== onstart finished: $(date -Iseconds) ==="
```

> After SSH login the venv auto-activates (via `.bashrc`). For non-interactive
> commands, prefix with `source /workspace/venv/bin/activate &&`.

---

## 3. Rent the instance

1. **Search filters** (left pane): GPU VRAM ≥ **24 GB** (LLaVA-1.5-7B fp16 peaks
   ~16 GB), **Verified** hosts. Disk ≥ **50 GB** (LLaVA weights ~14 GB + buffer;
   the template's 32 GB container size is tight — bump the disk slider to 50 GB).
2. **GPU pick:** any 24 GB card — RTX 3090 / 4090 / A5000 / L4 / RTX 6000 /
   A10. From your template's candidate list the **RTX 3090** and **RTX A5000**
   (24 GB) are the safe picks; the L4/A10/RTX 6000 at 22–23 GB will work but
   leave less headroom.
3. Select the **VTP-Eval** template, set disk to **50 GB**, click **RENT**.
4. The on-start runs once at first boot. LLaVA installs lazily on first model
   load, so on-start itself is ~1–2 min.

---

## 4. SSH in and verify the on-start

```bash
ssh -p <PORT> root@<HOST>           # from the instance's "Direct SSH" button
tail -n 30 /workspace/onstart.log   # should end with "onstart finished" + GPU OK + transformers 4.37.2
```
If it halted mid-stream, re-run (idempotent):
```bash
bash /workspace/vtp-eval/scripts/vast/onstart.sh   # or paste the §2 block into a file and run it
```

---

## 5. (Already done) The transformers-4.37.2 forward is committed

Stage 2 replaces `LlamaModel.forward` with transformers **4.37.2**'s forward +
the prune splice. This is **already implemented and committed** in
`vtp_eval/proposed_method/stage2_llm.py::_llama_forward_437` (validated on a 22 GB
RTX 2080 Ti) — no manual sync needed. It handles the mid-layer subtleties:
contiguous position re-indexing + KV-cache slicing (so RoPE positions stay in
range), and rebuilding the stale `attention_mask`/`position_ids` that `generate()`
passes after the cache shrinks.

> **Version caveat:** the body is pinned to transformers 4.37.2 (matched by the
> recommended image tag). If you run on a different transformers version and hit
> a forward mismatch, re-sync from `inspect.getsource(LlamaModel.forward)` and
> re-apply the splice (force `output_attentions` at layer `k`; call
> `prune_after_layer_k` right after `hidden_states = layer_outputs[0]`). The
> procedure is documented in the `_llama_forward_437` docstring.

---

## 6. Verify with the smoke test

```bash
cd /workspace/vtp-eval
python -m pytest tests/test_proposed_smoke.py -m slow -v
```
Expected: the model loads (~14 GB download on first run, ~5 min), one generation
runs, and the test asserts **non-empty output** plus
`model._proposed_spans["vision_len"] == cfg.r1` (== 64 for the default config).
A green smoke test means both stages fired correctly.

---

## 7. Ad-hoc generation

```python
import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from vtp_eval.proposed_method import proposed_prune, ProposedConfig

path = "liuhaotian/llava-v1.5-7b"
tok, model, image_processor, _ = load_pretrained_model(
    path, None, get_model_name_from_path(path), attn_implementation="eager")  # eager required

model = proposed_prune(model, ProposedConfig(
    dominant_k=54, diversity_m=10, pruned_layer=12, llm_keep_r2=37))  # R1=64

img = Image.open("your.jpg").convert("RGB")
prompt = "USER: <image>\nWhat is in this image? ASSISTANT:"
input_ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX,
                                  return_tensors="pt").unsqueeze(0).to(model.device)
image_tensor = process_images([img], image_processor, model.config)[0]
with torch.inference_mode():
    out = model.generate(input_ids, images=image_tensor.unsqueeze(0).half().to(model.device),
                         max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

**Sweeping settings:** change the four `ProposedConfig` knobs (or read them from
`configs/proposed_method.yaml`). The yaml ships `proposed_R1-64_R2-37_k12`,
`proposed_R1-128_R2-37_k12`, `proposed_k8`, and `proposed_div0` (ablation:
diversity off).

> ⚠️ **Batch size:** `prune_after_layer_k` assumes the vision block starts at the
> same position across the batch (true at `batch_size=1`). For batched eval with
> left-padding, run `batch_size=1` (as several archived methods do) until the
> per-sample span path is added. Single-image generation above is unaffected.

---

## 8. Benchmarking (not yet wired)

The `llava_proposed` lmms-eval adapter (`vtp_eval/proposed_method/adapter.py`)
needs `LlavaPruningBase` and the lmms-eval harness, which live on
`archive/lmms-eval-pre-cleanup` and are restored under `vtp_eval/eval/` as a
separate task. Once that's done, `install/proposed.sh` gains an
`pip install -e <lmms-eval> --no-deps` line and you run benchmarks (POPE, GQA,
TextVQA, …) per run in `configs/proposed_method.yaml`.

---

## 9. Retrieve outputs / pause / destroy

- **Files:** instance page → *Files* tab → `/workspace/`, or
  `scp -P <PORT> -r root@<HOST>:/workspace/outputs ./out`.
- **Pause billing, keep the volume (model cache + your synced forward):** use
  **Stop**, not Destroy. Re-running the on-start on next boot is a no-op.
- **Destroy** only when done — it wipes the volume, so the next rent re-downloads
  LLaVA (~14 GB) and you redo §5.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Forward shape/arg error in `_llama_forward_437` | transformers != 4.37.2. Re-sync the forward per §5's version caveat, or pin the recommended image tag. |
| `AssertionError: layer-k attention is None … eager` | Model wasn't loaded with `attn_implementation="eager"`. Reload with it. |
| `onstart.log` ends before "onstart finished" | Re-run `bash install/proposed.sh`; it's idempotent. |
| `git checkout proposed-method` fails on the box | You didn't push the branch (Prereq 0.1). Push it, then re-run on-start. |
| `torch.cuda.is_available()` is `False` | Base-image torch/CUDA mismatch — `install/proposed.sh` already reinstalls cu121 torch; if it still fails, the host driver is too old, pick another host. |
| LLaVA `pip install` dependency conflict | The original LLaVA repo pins are finicky; loosen the offending pin in `install/proposed.sh` step 4 and re-run. Verify with `python -c "import llava"`. |
| CUDA OOM on model load | < 24 GB VRAM. Rent a bigger card. |
| Splice shape mismatch at layer k for batch>1 | Run `batch_size=1` (see §7 note) or add the per-sample span path. |
