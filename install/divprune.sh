#!/usr/bin/env bash
# DivPrune install — diversity-based projector-side pruning.
set -euo pipefail
bash install/_common.sh

# Drop any prior method's editable `llava` finder.
pip uninstall -y llava || true

if [ ! -d /content/divprune ]; then
  git clone --depth 1 https://github.com/vbdi/divprune.git /content/divprune
fi

LLAVA_DIR=/content/divprune/LLaVA

# Relax torch pin in pyproject (Colab has torch 2.10+cu128, not 2.1.2).
sed -i 's/"torch==2.1.2"/"torch"/g; s/"torchvision==0.16.2"/"torchvision"/g; s/"deepspeed==0.12.6"/"deepspeed"/g' $LLAVA_DIR/pyproject.toml

# Patch AutoConfig.register("llava", ...) collision: transformers 4.37.2 ships
# its own "llava" config name; the divprune fork registers under the same
# name → ValueError unless exist_ok=True.
python - <<'PYEOF'
import re
from pathlib import Path
p = Path("/content/divprune/LLaVA/llava/model/language_model/llava_llama.py")
src = p.read_text()
patched = re.sub(
    r'AutoConfig\.register\("llava",\s*LlavaConfig\)',
    'AutoConfig.register("llava", LlavaConfig, exist_ok=True)',
    src,
)
patched = re.sub(
    r'AutoModelForCausalLM\.register\(LlavaConfig,\s*LlavaLlamaForCausalLM\)',
    'AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, exist_ok=True)',
    patched,
)
p.write_text(patched)
print("Patched AutoConfig/AutoModel register() to exist_ok=True")
PYEOF

# Wrap MPT/Mistral imports in try/except — they import dead transformers APIs
# (e.g. _expand_mask from transformers.models.bloom, removed since 4.40+).
# LLaVA-1.5 only needs Llama; the other language-model submodules are
# unimported by us anyway.
python - <<'PYEOF'
from pathlib import Path
p = Path("/content/divprune/LLaVA/llava/model/__init__.py")
src = p.read_text()
new = []
for line in src.splitlines():
    if line.startswith("from .language_model."):
        # Wrap each in try/except
        name = line.split()[1].split(".")[-1]  # llava_llama / llava_mpt / ...
        new.append(f"try:\n    {line}\nexcept Exception as _e:\n    print(f'[divprune] skip {name}: '+str(_e))")
    else:
        new.append(line)
p.write_text("\n".join(new) + "\n")
print("Wrapped LLaVA model imports in try/except")
PYEOF

pip install -e $LLAVA_DIR --no-deps
pip install -e /content/lmms-eval --no-deps
pip install -e /content/vtp-eval --no-deps

# Force-pin to the verified-working transformers stack
pip install -q --force-reinstall --no-deps \
    transformers==4.37.2 tokenizers==0.15.1 huggingface_hub==0.24.7
