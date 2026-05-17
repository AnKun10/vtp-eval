#!/usr/bin/env bash
# SparseVLMs install — text-guided multi-layer pruning (layers 2/6/15).
set -euo pipefail
bash install/_common.sh

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

pip install -e /content/SparseVLMs --no-deps   # installs SparseVLMs' LLaVA fork
pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Force-pin to the verified-working transformers stack. SparseVLMs asserts
# attn_implementation='sdpa', which we set per-run in configs/methods.yaml.
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==1.11.0
