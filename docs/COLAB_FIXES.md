# Colab POPE smoke-test fixes (2026-05-17)

Smoke-tested `notebooks/pope_eval.ipynb` on Colab Pro **L4 (23 GB)** with
Python 3.12, CUDA 12.8, torch 2.10 (default Colab kernel).

**Re-test on fresh clone after first round of repo fixes: 5/6 methods pass.**
FastV remains blocked on a deeper Python 3.12 incompatibility. Four small
issues still need to land in the repo (see "Remaining issues after first
fix pass" below).

| Method               | Status     | POPE F1 (n=10) | Blocker                                                |
|----------------------|------------|----------------|--------------------------------------------------------|
| baseline             | ✅ PASS    | 0.67           | —                                                      |
| fastv_K2_R128        | ❌ BLOCKED | —              | FastV-patched modeling_llama.py incompatible with TF 4.37.2 attention-output API (L777 `torch.mean(tuple, dim=int)`). Needs FastV's bundled transformers fork. |
| sparsevlm_192        | ❌ BLOCKED | —              | (a) requires `attn_implementation='sdpa'` (asserts at `modelling_sparse_llama.py:727`), conflicts with `eager` set in common_args. (b) builder passes `multimodal=True` kwarg the custom `LlavaLlamaDynamicForCausalLM.__init__` doesn't accept. |
| visionzip_64         | ✅ PASS    | 0.80           | —                                                      |
| divprune_0.098       | ✅ PASS    | 0.91           | —                                                      |
| sparsevila_0.5_0.5   | ✅ PASS    | 0.80           | —                                                      |

## Root causes (apply across all install scripts)

### 1. `install/_common.sh` over-pins protobuf
`pip install -q protobuf==3.20.3` breaks transformers 4.37.2 CLIP image
processor: `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`.

**Fix:** drop the `protobuf==3.20.3` pin; let transformers pick a compatible
4.x+ version. Tested with `protobuf 5.29.6`.

### 2. `install/_common.sh` clones lmms-eval @ HEAD, which uses MODEL_REGISTRY_V2
Since commit `b83e3ec4` (2026-02-10), lmms-eval resolves models via a new
`ModelRegistryV2` (manifest-driven). The legacy `@register_model` decorator
used by vtp-eval adapters only populates the unused `MODEL_REGISTRY` dict —
adapters become invisible.

**Fix:** pin lmms-eval to a tag from before that commit. Latest pre-V2 tag is
**v0.5** (2025-10-07). Add `cd lmms-eval && git checkout v0.5` after clone.

Additional dep: v0.5 requires `sqlitedict` (added to install requirements).

### 3. Even v0.5 doesn't use `MODEL_REGISTRY` for lookup
`lmms_eval/models/__init__.py:97 get_model` consults the hardcoded
`AVAILABLE_SIMPLE_MODELS` dict, not `MODEL_REGISTRY`. The `register_model`
decorator alone is a no-op for resolution.

**Fix:** mutate `lmms_eval.models.AVAILABLE_SIMPLE_MODELS` at adapter package
import time, plus add a CLI wrapper that imports adapters before launching
the lmms_eval main.

```python
# vtp_eval/adapters/__init__.py
import lmms_eval.models as _lm
_lm.AVAILABLE_SIMPLE_MODELS.update({
    "llava_baseline":   "vtp_eval.adapters.llava_baseline.LlavaBaseline",
    "llava_fastv":      "vtp_eval.adapters.llava_fastv.LlavaFastV",
    "llava_sparsevlm":  "vtp_eval.adapters.llava_sparsevlm.LlavaSparseVLM",
    "llava_visionzip":  "vtp_eval.adapters.llava_visionzip.LlavaVisionZip",
    "llava_divprune":   "vtp_eval.adapters.llava_divprune.LlavaDivPrune",
    "llava_sparsevila": "vtp_eval.adapters.llava_sparsevila.LlavaSparseVILA",
})
```

```python
# vtp_eval/run_lmms.py  (new)
import vtp_eval.adapters
from lmms_eval.__main__ import cli_evaluate
if __name__ == "__main__":
    cli_evaluate()
```

```diff
# scripts/run_one.sh
-python -m lmms_eval \
+python -m vtp_eval.run_lmms \
```

### 4. `configs/methods.yaml` `dtype: float16` is not a valid `Llava` kwarg
lmms-eval v0.5 `Llava.__init__` ends with `assert kwargs == {}, ...` and
doesn't accept `dtype`. The class hardcodes fp16 internally.

**Fix:** remove `dtype: float16` from `common_args`.

### 5. Every per-method LLaVA fork pyproject pins `torch==2.1.2` (and friends)
Colab ships torch 2.10. `pip install -e` errors: "No matching distribution
found for torch==2.1.2".

**Fix:** in each install script, before `pip install -e`, relax the pins:

```bash
sed -i 's/"torch==2.1.2"/"torch"/g;
        s/"torchvision==0.16.2"/"torchvision"/g;
        s/"deepspeed==0.12.6"/"deepspeed"/g' "<fork>/pyproject.toml"
pip install -e <fork> --no-deps
```

Always use `--no-deps` to avoid pip pulling incompatible accelerate/gradio/
sentencepiece/timm pins from the fork's metadata.

### 6. `AutoConfig.register("llava", ...)` collides with transformers' native config
Older LLaVA forks (FastV, DivPrune) register their custom config under
`"llava"`. transformers 4.37.2 has a built-in LLaVA config under the same
name → `ValueError: 'llava' is already used by a Transformers config, pick
another name.`

**Fix:** patch the fork to use `exist_ok=True`:

```python
AutoConfig.register("llava", LlavaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, exist_ok=True)
```

haotian-liu's recent fork uses `"llava_llama"` instead, which avoids the
collision; SparseVLMs also uses `"llava_llama"` and is unaffected.

### 7. LLaVA forks unconditionally import MPT/Mistral submodules that pull dead transformers APIs
e.g. FastV `llava_mpt.py → mpt/hf_prefixlm_converter.py` imports
`_expand_mask` from `transformers.models.bloom`, which was removed.

**Fix:** wrap each language-model import in `llava/model/__init__.py` in
`try/except: pass`. LLaVA-1.5 only needs Llama.

```python
try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
except Exception as e:
    print("llava_llama import failed:", e)
try:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
except Exception:
    pass
try:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
except Exception:
    pass
```

### 8. Multiple LLaVA forks share the package name `llava`
`pip install -e <fork>` for multiple forks each register `llava` editable.
The kernel caches the first finder loaded — switching methods within one
kernel keeps importing the stale fork. `pip show llava` lies (shows newest
install location while runtime resolves to an older one).

**Fix:** between methods, **uninstall before reinstall** and **restart the
kernel** (the notebook's "Cell A: `os._exit(0)`" is mandatory):

```bash
pip uninstall -y llava
pip install -e <next-fork> --no-deps
```

### 9. `install/_patch_fastv.py` path is stale
Expects `/content/FastV/src/FastV/inference/transformers_replace/...`; the
current FastV repo layout has the file at
`/content/FastV/src/transformers/src/transformers/models/llama/modeling_llama.py`.

**Fix:** update the source path in `_patch_fastv.py`. (Or drop the script —
see note 11.)

### 10. lmms-eval Llava.generate() passes `image_sizes` that FastV's older `LlavaLlamaForCausalLM.forward` doesn't accept
Triggers `_validate_model_kwargs` → `ValueError: model_kwargs not used: ['image_sizes']`.

**Workaround applied:** monkey-patch `_model.__class__.generate` to pop
`image_sizes` from kwargs. Still fails downstream (see 11).

### 11. FastV's patched modeling_llama.py expects an older transformers attention API
At L777: `torch.mean(last_layer_attention, dim=1)[0]` — `last_layer_attention`
is now a tuple (one tensor per head/layer) under transformers 4.37.2 eager
attention, but FastV's patch assumes a tensor.

**Fix:** install FastV's bundled transformers fork at
`/content/FastV/src/transformers/` instead of patching one file into the
system transformers. This is a structural change; out of scope for a small
patch. Mark FastV as needing a dedicated install path.

### 12. SparseVLMs hard-requires `attn_implementation='sdpa'`
`modelling_sparse_llama.py:727 — assert config._attn_implementation == 'sdpa'`.
`common_args` currently sets `eager` (for FastV).

**Fix:** make `attn_implementation` per-method in `configs/methods.yaml`
(remove from `common_args`; set `eager` only for FastV).

### 13. SparseVLMs custom `LlavaLlamaDynamicForCausalLM.__init__` doesn't accept `multimodal` kwarg
lmms-eval Llava passes `multimodal=True` via `from_pretrained`. SparseVLMs'
constructor rejects it.

**Fix:** either patch SparseVLMs `sparse_llava_llama.py` to accept (and
ignore) `**kwargs`, or override `load_pretrained_model` to strip `multimodal`
before `from_pretrained`.

## Per-file change list for the PR

```
install/_common.sh                       drop protobuf pin; pin lmms-eval@v0.5; add sqlitedict
install/baseline.sh                      relax LLaVA pyproject torch pins; --no-deps; force transformers/tokenizers/huggingface-hub after lmms-eval install
install/fastv.sh                         (no fix yet — see notes 9, 10, 11)
install/sparsevlm.sh                     same scaffolding fixes + per-method sdpa attn_impl (see notes 12, 13)
install/visionzip.sh                     relax LLaVA torch pins; --no-deps
install/divprune.sh                      relax LLaVA torch pins; --no-deps; AutoConfig exist_ok patch; MPT try/except in __init__
install/sparsevila.sh                    OK (just needs --no-deps)
install/_patch_fastv.py                  fix stale src path; consider deleting + bundling transformers fork install
configs/methods.yaml                     remove dtype: float16; move attn_implementation per-run
scripts/run_one.sh                       use `python -m vtp_eval.run_lmms` (or add adapter-preload entry)
vtp_eval/adapters/__init__.py            mutate lmms_eval.models.AVAILABLE_SIMPLE_MODELS
vtp_eval/run_lmms.py        [NEW]        CLI wrapper that imports adapters first
```

## Remaining issues after first fix pass (2026-05-17 re-test)

Re-tested with a fresh `git clone` of the post-fix repo:

| Method               | Re-test status                                     |
|----------------------|----------------------------------------------------|
| baseline             | ✅ PASS (F1=0.67)                                  |
| fastv_K2_R128        | ❌ BLOCKED (Python 3.12 ↔ FastV bundled fork wall) |
| sparsevlm_192        | ✅ PASS w/ batch=1 (F1=0.80)                       |
| visionzip_64         | ✅ PASS w/ extra `pip uninstall llava` (F1=0.80)   |
| divprune_0.098       | ✅ PASS (F1=0.91)                                  |
| sparsevila_0.5_0.5   | ✅ PASS w/ extra `pip install third_party/LLaVA` (F1=0.80) |

### 14. `huggingface_hub==1.11.0` pin breaks transformers 4.37.2 (and 4.31.0)
Both transformers versions assert `huggingface-hub>=0.19.3,<1.0`. With
`huggingface_hub==1.11.0` on disk, `import transformers` raises
`ImportError: huggingface-hub>=0.19.3,<1.0 is required …` at fresh-kernel
import.

**Why the first smoke pass didn't catch this — and why the note in this
file said 1.11.0:** The first session's "force-reinstall" command actually
installed `huggingface_hub==0.24.7` on disk, but the verification cell
re-read `huggingface_hub.__version__` from a previously imported (cached)
copy — `notebook_login()` had already imported hub 1.11.0 into
`sys.modules` before the reinstall. The cached attribute showed 1.11.0
while the on-disk version was 0.24.7. The smoke run worked because the
bash subprocess spawned by `run_one.sh` does fresh imports from disk and
read 0.24.7. The 1.11.0 number was copied into this file from the cached
attribute, then into the install scripts — locking the broken version in.

Lesson: when noting verified versions, cross-check with `pip show <pkg>`
or `python -c 'import importlib.metadata; print(importlib.metadata.version("<pkg>"))'`
from a fresh subprocess, not `mod.__version__` of an already-imported module.

**Fix:** replace `huggingface_hub==1.11.0` with `huggingface_hub==0.24.7`
(or any `0.24.x`) in `install/baseline.sh`, `install/sparsevlm.sh`,
`install/visionzip.sh`, `install/divprune.sh`, `install/sparsevila.sh`,
and `install/fastv.sh`.

### 15. `install/sparsevila.sh` never installs the bundled LLaVA submodule
`sparsevila.sh` clones the submodule (`third_party/LLaVA` via
`git submodule update --init --recursive`) but never `pip install -e`s it.
At runtime SparseVILA's loader does `from llava.model.builder import
load_pretrained_model` → `ModuleNotFoundError: No module named 'llava'`.

**Fix:** add to `install/sparsevila.sh`, after the submodule update:

```bash
LLAVA_DIR=/content/SparseVILA/third_party/LLaVA
sed -i 's/"torch==2.1.2"/"torch"/g;
        s/"torchvision==0.16.2"/"torchvision"/g;
        s/"deepspeed==0.12.6"/"deepspeed"/g' $LLAVA_DIR/pyproject.toml
pip install -e $LLAVA_DIR --no-deps
```

### 16. Per-method install scripts don't `pip uninstall -y llava` first
Each method's editable `pip install -e <fork>/LLaVA` registers the package
name `llava` to a different on-disk path. The previous method's editable
finder stays in `dist-packages/__editable___llava_<ver>_finder.py` and wins
import resolution when there are multiple. `pip show llava` reports the
newest install while `import llava` resolves to the older one.

Symptom: VisionZip smoke crashed inside `SparseVLMs/llava/model/...` even
though VisionZip uses haotian-liu LLaVA.

**Fix:** start every per-method install script with

```bash
pip uninstall -y llava
```

so the prior finder is removed before installing the current fork.

### 17. SparseVLMs requires `VTP_BATCH_SIZE=1`
SparseVLMs' custom SDPA path (`SparseVLMs/llava/model/language_model/utils.py:109`)
does `attn_bias += attn_mask` where `attn_bias` is `[seq, seq]` and
`attn_mask` is `[batch, 1, seq, seq]`. With `batch=2` this raises
`RuntimeError: output with shape [seq, seq] doesn't match the broadcast
shape [2, 1, seq, seq]`. Setting `batch=1` avoids it (the broadcast collapses
to a no-op).

**Fix:** either patch SparseVLMs' `utils.py` to broadcast `attn_bias` before
the `+=`, or document the per-method batch-size constraint and add a
`batch_size: 1` per-run override knob to `configs/methods.yaml` +
`scripts/run_one.sh`.

### 18. FastV's bundled transformers (4.31.0) is incompatible with Python 3.12
The new `install/fastv.sh` installs `/content/FastV/src/transformers`
(transformers 4.31.0). That version's `dependency_versions_check` requires
`tokenizers>=0.11.1,!=0.11.3,<0.14`. The only tokenizers releases <0.14
predate Python 3.12 — there are no pre-built wheels, and source build fails
on Colab (`error: Failed building wheel for tokenizers`).

There is no clean path forward with Python 3.12:
- Pin Colab kernel to Python 3.10 → not user-controllable on Colab
- Stay with stock transformers 4.37.2 and re-patch FastV's `modeling_llama.py`
  → still hits the earlier tuple/tensor mismatch at L777 because FastV's patch
  assumes the older eager-attention output shape
- Maintain a Python-3.12-compatible FastV patch series (port their token-prune
  logic onto transformers 4.37.2's attention API) → out of scope for vtp-eval

**Recommendation:** drop FastV from the default `pope_eval.ipynb` runs and
keep it as a documented-blocked entry until Colab moves off Python 3.12 or
FastV ships a maintained branch.

## Verified working stack versions (Colab L4, 2026-05-17)

```
python                 3.12
torch                  2.10.0+cu128
transformers           4.37.2     (force-reinstalled after lmms-eval pulls 5.x)
tokenizers             0.15.1
huggingface_hub        0.24.7     (must be <1.0; 1.11.0 fails dependency_versions_check on fresh kernel)
protobuf               5.29.6
lmms-eval              v0.5 tag
sqlitedict             installed by us
llava                  1.2.2.post1 (haotian-liu, with relaxed torch pin)
```
