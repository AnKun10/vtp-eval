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
| **Method end-to-end** (generation) | this runbook: install + the `_llama_forward_437` sync (§5) |
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
| **Version Tag** | `[Automatic]` (`@vastai-automatic-tag`) |
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

This is the proposed-method counterpart of `scripts/vast/onstart.sh`. It clones
the repo, checks out `proposed-method`, runs `install/proposed.sh` (original
LLaVA + transformers 4.37.2), and sanity-checks. **Idempotent** — safe on every
boot. It does **not** auto-sync `_llama_forward_437` (that is a deliberate step,
§5).

```bash
#!/bin/bash
# Vast.ai onstart for the proposed two-stage pruning method via vtp-eval.
set -euo pipefail

LOG=/workspace/onstart.log
mkdir -p /workspace
exec > >(tee -a "$LOG") 2>&1
echo "=== onstart (proposed-method) started: $(date -Iseconds) ==="

export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"

REPO=/workspace/vtp-eval
BRANCH=proposed-method
if [ ! -d "$REPO" ]; then
    git clone https://github.com/AnKun10/vtp-eval.git "$REPO"
fi
cd "$REPO"
git fetch origin --prune || echo "[warn] git fetch skipped"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || echo "[warn] git pull skipped"

bash install/proposed.sh

# Persist HF cache + auto-cd for interactive SSH sessions.
grep -q 'HF_HOME=/workspace/.cache/huggingface' /root/.bashrc 2>/dev/null \
    || echo 'export HF_HOME=/workspace/.cache/huggingface' >> /root/.bashrc
grep -q "cd $REPO" /root/.bashrc 2>/dev/null || echo "cd $REPO" >> /root/.bashrc

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"
python -c "import transformers; print('transformers', transformers.__version__)"  # expect 4.37.2
python -c "import llava; from vtp_eval.proposed_method import proposed_prune; print('proposed_method import OK')"
echo "=== onstart finished: $(date -Iseconds) ==="
```

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

## 5. Sync `_llama_forward_437` (the one Vast-only step)

Stage 2 replaces `LlamaModel.forward` with a copy of **transformers 4.37.2**'s
forward that has the prune spliced in. That verbatim body cannot be shipped from
a dev box (it is version-pinned), so `vtp_eval/proposed_method/stage2_llm.py`
ships a stub that raises `NotImplementedError`. Sync it now, on the instance
where transformers 4.37.2 is installed.

**5.1 — print the installed forward:**
```bash
cd /workspace/vtp-eval
python - <<'PY'
import inspect
from transformers.models.llama.modeling_llama import LlamaModel
print(inspect.getsource(LlamaModel.forward))
PY
```

**5.2 — edit `vtp_eval/proposed_method/stage2_llm.py`:** replace the body of
`_llama_forward_437(self, cfg, *args, **kw)` with the printed forward, keeping
the `(self, cfg, *args, **kw)` signature (bind the original kwargs at the top:
`input_ids=kw.get("input_ids", args[0] if args else None)`, etc. — easiest is to
copy 4.37.2's exact parameter list in place of `*args, **kw`).

Then make **two** edits inside the decoder-layer loop:

1. If the loop is `for decoder_layer in self.layers:`, change it to
   `for idx, decoder_layer in enumerate(self.layers):` so `idx` is available.
2. Force attention at layer `k` — change the `output_attentions=output_attentions`
   argument of the `decoder_layer(...)` call to:
   ```python
   output_attentions=output_attentions or (idx == cfg.pruned_layer),
   ```
3. Immediately **after** `hidden_states = layer_outputs[0]`, insert (note the
   mask guard — the no-text skip returns `_mask=None` meaning "keep the existing
   mask"; overwriting `attention_mask` with `None` would drop causal masking for
   layers `k+1..L`):
   ```python
   spans = getattr(self, "_proposed_spans", None)
   if idx == cfg.pruned_layer and spans is not None and hidden_states.shape[1] > 1:
       from vtp_eval.proposed_method.stage2_llm import prune_after_layer_k
       _hs, _pos, _mask, _pkv = prune_after_layer_k(
           hidden_states, layer_outputs[1], position_ids,
           past_key_values, spans, cfg, use_cache)
       hidden_states, position_ids, past_key_values = _hs, _pos, _pkv
       if _mask is not None:
           attention_mask = _mask
   ```
   (`prune_after_layer_k`, `text_to_vision_scores`, etc. are already defined in
   the same module, so a local import is unnecessary if you splice inside that
   file — shown here only for clarity.)

The exact procedure is also written in the docstring of `_llama_forward_437`.

> If you'd rather version-control the synced forward, commit it on the
> `proposed-method` branch from the instance (or copy it back and commit
> locally) so future rents pick it up via `git pull` and skip this step.

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
| `NotImplementedError: _llama_forward_437 must be synced …` | You skipped §5. Sync the 4.37.2 forward. |
| `AssertionError: layer-k attention is None … eager` | Model wasn't loaded with `attn_implementation="eager"`. Reload with it. |
| `onstart.log` ends before "onstart finished" | Re-run `bash install/proposed.sh`; it's idempotent. |
| `git checkout proposed-method` fails on the box | You didn't push the branch (Prereq 0.1). Push it, then re-run on-start. |
| `torch.cuda.is_available()` is `False` | Base-image torch/CUDA mismatch — `install/proposed.sh` already reinstalls cu121 torch; if it still fails, the host driver is too old, pick another host. |
| LLaVA `pip install` dependency conflict | The original LLaVA repo pins are finicky; loosen the offending pin in `install/proposed.sh` step 4 and re-run. Verify with `python -c "import llava"`. |
| CUDA OOM on model load | < 24 GB VRAM. Rent a bigger card. |
| Splice shape mismatch at layer k for batch>1 | Run `batch_size=1` (see §7 note) or add the per-sample span path. |
