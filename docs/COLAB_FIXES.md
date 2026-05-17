# Colab POPE smoke-test fixes (2026-05-17)

Smoke-tested `notebooks/pope_eval.ipynb` on Colab Pro **L4 (23 GB)** with
Python 3.12, CUDA 12.8, torch 2.10 (default Colab kernel). 4/6 methods pass
on 10 POPE samples; 2 are blocked on deep transformers-API drift.

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

## Verified working stack versions (Colab L4, 2026-05-17)

```
python                 3.12
torch                  2.10.0+cu128
transformers           4.37.2     (force-reinstalled after lmms-eval pulls 5.x)
tokenizers             0.15.1
huggingface_hub        1.11.0     (compat — transformers 4.37 doesn't break)
protobuf               5.29.6
lmms-eval              v0.5 tag
sqlitedict             installed by us
llava                  1.2.2.post1 (haotian-liu, with relaxed torch pin)
```
