"""One-shot script that emits notebooks/pope_eval.ipynb.

Kept in the repo so the notebook is reproducible from a text source.
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

md(f"""\
# vtp-eval — POPE benchmark on LLaVA-1.5 7B
Compares baseline + 4 visual-token-pruning methods (SparseVLMs, VisionZip, DivPrune, SparseVILA).
**Runtime:** Colab Pro, GPU L4 (24 GB). **Total time:** ~2–3 hours for full POPE.
**IMPORTANT:** You MUST manually restart the runtime between methods (marked in red below).

> _Note: FastV is excluded from the default runs — its bundled transformers fork (4.31.0)
> needs tokenizers <0.14, which has no Python 3.12 wheels. See `docs/COLAB_FIXES.md` #18._
""")

code(f"""\
# Cell 2 — Clone vtp-eval workspace
!git clone --depth 1 {REPO} /content/vtp-eval || (cd /content/vtp-eval && git pull)
%cd /content/vtp-eval
""")

code("""\
# Cell 3 — Detect GPU + set L4 config
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os
os.environ["VTP_BATCH_SIZE"] = "2"
os.environ["VTP_DTYPE"] = "float16"
print("Batch size:", os.environ["VTP_BATCH_SIZE"])
""")

code("""\
# Cell 4 — HuggingFace login (need for POPE dataset + LLaVA weights)
from huggingface_hub import notebook_login
notebook_login()
""")

# Default run blocks for Colab Pro / L4 (Python 3.12).
# FastV is omitted: its bundled transformers fork (4.31.0) needs tokenizers
# <0.14, which has no Python 3.12 wheels. Stock transformers 4.37.2's eager
# attention output is a tuple, breaking FastV's single-file modeling_llama
# patch. See docs/COLAB_FIXES.md root cause #18.
RUNS = [
    ("baseline",           "install/baseline.sh",  "baseline"),
    ("sparsevlm_192",      "install/sparsevlm.sh", "SparseVLMs (retain=192)"),
    ("visionzip_64",       "install/visionzip.sh", "VisionZip (64 tokens)"),
    ("divprune_0.098",     "install/divprune.sh",  "DivPrune (ratio=0.098)"),
    ("sparsevila_0.5_0.5", "install/sparsevila.sh", "SparseVILA (enc=0.5, dec=0.5)"),
]

for i, (run_name, install_sh, label) in enumerate(RUNS):
    n = i + 1
    if i == 0:
        md(f"## Run {n}/6 — {label}\nNo restart needed for the first run.")
    else:
        md(f"""\
## Run {n}/6 — {label}
**🔴 BEFORE running these cells: manually restart the runtime** (Runtime → Restart runtime), then run **Cell A** to force-restart, **Cell B** to install + run.

This is required because previous methods may have patched `transformers.models.llama.modeling_llama` in ways that conflict with this method.
""")
        code("""\
# Cell A — Force kernel restart (alternative to manual restart)
import os; os._exit(0)
""")
    code(f"""\
# Cell B — Install + run {label}
%cd /content/vtp-eval
!bash {install_sh}
!bash scripts/run_one.sh {run_name}
""")

md("""\
## Aggregate & visualize
After all 6 runs complete, this cell builds `summary.csv` and plots the two
canonical scatter plots (accuracy vs FLOPs, accuracy vs latency).
""")

code("""\
# Cell — Aggregate
%cd /content/vtp-eval
!python -m vtp_eval.utils.result_io aggregate results/ --output results/summary.csv
import pandas as pd
df = pd.read_csv("results/summary.csv")
df
""")

code("""\
# Cell — Plot
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("results/summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for _, row in df.iterrows():
    axes[0].scatter(row["tflops_prefill"], row["pope_f1"], s=80)
    axes[0].annotate(row["method"], (row["tflops_prefill"], row["pope_f1"]),
                     fontsize=9, xytext=(5, 5), textcoords="offset points")
    axes[1].scatter(row["latency_ms_mean"], row["pope_f1"], s=80)
    axes[1].annotate(row["method"], (row["latency_ms_mean"], row["pope_f1"]),
                     fontsize=9, xytext=(5, 5), textcoords="offset points")
axes[0].set(xlabel="Prefill TFLOPs (theoretical)", ylabel="POPE F1",
            title="Accuracy vs Compute")
axes[1].set(xlabel="Latency per sample (ms)", ylabel="POPE F1",
            title="Accuracy vs Latency")
fig.tight_layout()
plt.savefig("results/scatter.png", dpi=150)
plt.show()
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
out = Path(__file__).parent / "pope_eval.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} with {len(CELLS)} cells")
