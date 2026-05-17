"""One-shot script that emits notebooks/fastv_eval.ipynb.

FastV-only evaluation notebook. Separate from pope_eval.ipynb because FastV
requires Python 3.10 (its bundled transformers fork needs tokenizers <0.14,
which has no Python 3.12 wheels). Uses condacolab to switch the Colab kernel
to Python 3.10 before installing FastV.
"""
import json
from pathlib import Path

REPO = "https://github.com/AnKun10/vtp-eval.git"

CELLS = []

def md(text): CELLS.append({"cell_type": "markdown", "metadata": {},
                             "source": text.splitlines(keepends=True)})
def code(text): CELLS.append({"cell_type": "code", "metadata": {},
                               "execution_count": None, "outputs": [],
                               "source": text.splitlines(keepends=True)})

md("""\
# vtp-eval — FastV evaluation on POPE (LLaVA-1.5 7B)

**Status:** Experimental — FastV requires **Python 3.10**, which Colab's default
runtime (Python 3.12) doesn't ship. This notebook uses
[`condacolab`](https://github.com/conda-incubator/condacolab) to install miniconda
with Python 3.10 inside the Colab VM, then runs FastV in that environment.

**Why FastV needs Python 3.10:**
- FastV ships a bundled `transformers` fork pinned at 4.31.0
  (the single-file `modeling_llama.py` patch can't be ported cleanly to 4.37.2 —
  its expected attention output is a tensor, but 4.37 returns a tuple).
- transformers 4.31.0's `dependency_versions_check` requires `tokenizers < 0.14`.
- `tokenizers` 0.13.x has prebuilt wheels for Python 3.10 but **not** Python 3.12.

**Runtime:** Colab Pro, GPU L4 (24 GB). **Total time:** ~5 min setup + ~30–40 min for 100 POPE samples.

**Workflow:**
1. Run **Cell 2** (condacolab install). The kernel will auto-restart — that's expected.
2. After restart, run cells 3 onwards in order.
3. If you want the full ~9k-sample POPE run instead of the quick 100-sample smoke,
   delete the `100` argument in Cell 6.
""")

code("""\
# Cell 2 — Install Python 3.10 via condacolab.
# This auto-restarts the kernel. After restart, continue from Cell 3.
!pip install -q condacolab
import condacolab
condacolab.install()  # restarts the kernel
""")

md("""\
---

## ⬇ AFTER KERNEL RESTART, continue from here ⬇
""")

code(f"""\
# Cell 3 — Verify Python 3.10 + clone vtp-eval repo
import sys
print(f"Python {{sys.version_info[:3]}} — expect (3, 10, ...)")
assert sys.version_info[:2] == (3, 10), (
    "Kernel didn't restart into Python 3.10. Re-run Cell 2 (condacolab.install) "
    "and wait for the auto-restart before running Cell 3 again."
)
!git clone --depth 1 {REPO} /content/vtp-eval || (cd /content/vtp-eval && git pull)
%cd /content/vtp-eval
""")

code("""\
# Cell 4 — GPU check + HuggingFace login (needed for POPE dataset + LLaVA weights)
!nvidia-smi --query-gpu=name,memory.total --format=csv
from huggingface_hub import notebook_login
notebook_login()
""")

code("""\
# Cell 5 — Install FastV stack
# This pulls in:
#   - torch (CUDA-12 wheel — fresh, since the conda env has empty site-packages)
#   - FastV's bundled transformers 4.31.0 (from /content/FastV/src/transformers)
#   - FastV's LLaVA fork (with relaxed torch pin)
#   - lmms-eval @ v0.5 tag
#   - tokenizers 0.13.3 (last <0.14 with cp310 wheels)
#   - huggingface_hub 0.24.7 (must be <1.0 for transformers 4.31)
#
# Verify GPU torch after install:
!bash install/fastv.sh
""")

code("""\
# Cell 5b — Verify the FastV stack imports cleanly + sees CUDA
import torch
print(f"torch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "GPU not visible to torch — switch runtime to GPU and re-run from Cell 2"

import transformers
print(f"transformers {transformers.__version__} — should be 4.31.0 (FastV bundled fork)")

import tokenizers
print(f"tokenizers {tokenizers.__version__} — should be 0.13.x")

import lmms_eval, vtp_eval, vtp_eval.adapters
print("lmms_eval + vtp_eval.adapters imported")
""")

code("""\
# Cell 6 — Run FastV on POPE
# Quick smoke: 100 samples. For full POPE (~9k samples), remove the trailing "100".
# Single-sample batch (FastV's attention-output handling is per-sample under
# the bundled-fork eager-attention path).
import os
os.environ["VTP_BATCH_SIZE"] = "1"
!bash scripts/run_one.sh fastv_K2_R128 configs/methods.yaml pope 100
""")

code("""\
# Cell 7 — Inspect results
import json
from pathlib import Path

results_dir = Path("results/fastv_K2_R128")
print(f"=== Files in {results_dir} ===")
for f in sorted(results_dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(results_dir)}  ({f.stat().st_size} bytes)")

results_json = next(results_dir.rglob("results.json"), None)
if results_json:
    print("\\n=== results.json (POPE metrics) ===")
    data = json.loads(results_json.read_text())
    if "results" in data and "pope" in data["results"]:
        for k, v in data["results"]["pope"].items():
            print(f"  {k}: {v}")

timing_json = results_dir / "timing.json"
if timing_json.exists():
    print("\\n=== timing.json (latency + VRAM) ===")
    print(json.dumps(json.loads(timing_json.read_text()), indent=2))
""")

code("""\
# Cell 8 — Optional: aggregate FastV row alongside main-notebook results
# Skip this cell if you haven't run notebooks/pope_eval.ipynb in this same workspace.
# When run together, the summary CSV will include all 6 methods (including FastV).
!python -m vtp_eval.utils.result_io aggregate results/ --output results/summary_with_fastv.csv
import pandas as pd
df = pd.read_csv("results/summary_with_fastv.csv")
df
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).parent / "fastv_eval.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} with {len(CELLS)} cells")
