#!/usr/bin/env bash
# SparseVLMs install — text-guided multi-layer pruning (layers 2/6/15).
set -euo pipefail
bash install/_common.sh

# Drop any prior method's editable `llava` finder.
pip uninstall -y llava || true

if [ ! -d /content/SparseVLMs ]; then
  git clone --depth 1 https://github.com/Gumpest/SparseVLMs.git /content/SparseVLMs
fi

# Relax torch pin in pyproject.
sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' /content/SparseVLMs/pyproject.toml

# Patch custom LlavaLlamaDynamicForCausalLM.__init__ to accept (and ignore)
# **kwargs. lmms-eval Llava passes `multimodal=True` via from_pretrained,
# which the SparseVLMs custom constructor rejects with TypeError.
python - <<'PYEOF'
import re
from pathlib import Path

p = Path("/content/SparseVLMs/llava/model/language_model/sparse_llava_llama.py")
src = p.read_text()

# Find the LlavaLlamaDynamicForCausalLM.__init__ signature and add **kwargs
# if it isn't already present.
patched = re.sub(
    r'(class\s+LlavaLlamaDynamicForCausalLM[^:]*:\s*\n(?:\s*"""[\s\S]*?"""\s*\n)?\s*def\s+__init__\(self[^)]*?)\):',
    lambda m: m.group(1) + ', **kwargs):' if '**kwargs' not in m.group(1) else m.group(0),
    src,
    count=1,
)
if patched == src:
    print("WARN: did not find LlavaLlamaDynamicForCausalLM.__init__ signature to patch")
else:
    p.write_text(patched)
    print("Patched LlavaLlamaDynamicForCausalLM.__init__ to accept **kwargs")
PYEOF

# Wrap the efficiency-info logger in modelling_sparse_llama.py:371. Without
# this, certain POPE samples trigger `int(token_pool / num_forward)` where
# num_token_pool intermediate computation goes to +inf, raising
# `OverflowError: cannot convert float infinity to integer` mid-eval and
# aborting after ~340/9000 samples. The logger is non-essential profiling.
python - <<'PYEOF'
from pathlib import Path
p = Path("/content/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py")
src = p.read_text()
old = 'loggerinfo.info(f"{prefix} Equal Tokens: {int(self.num_token_pool / self.num_forward)}, Prefill Time (ms): {self.total_cuda_time:.2f}, TFLOPs:{FLOPs_avg_sample:.2f}")'
new = '''try:
                loggerinfo.info(f"{prefix} Equal Tokens: {int(self.num_token_pool / self.num_forward) if self.num_forward else 0}, Prefill Time (ms): {self.total_cuda_time:.2f}, TFLOPs:{FLOPs_avg_sample:.2f}")
            except (OverflowError, ValueError):
                pass  # skip per-sample efficiency log when intermediate values are inf/nan'''
if old in src:
    p.write_text(src.replace(old, new))
    print("Wrapped efficiency logger with try/except (OverflowError guard)")
else:
    print("WARN: efficiency-logger pattern not found (already patched?)")
PYEOF

pip install -e /content/SparseVLMs --no-deps   # installs SparseVLMs' LLaVA fork
pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Force-pin to the verified-working transformers stack. SparseVLMs asserts
# attn_implementation='sdpa', which we set per-run in configs/methods.yaml.
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==0.24.7
