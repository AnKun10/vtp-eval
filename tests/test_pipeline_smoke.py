"""End-to-end smoke: baseline adapter runs 5 POPE samples on GPU.

Marker: slow — skipped by default in CI, run only when LLaVA weights are
available and CUDA is present (i.e. on Colab after baseline.sh).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    import lmms_eval  # noqa: F401
    HAS_LMMS = True
except ImportError:
    HAS_LMMS = False


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA")
@pytest.mark.skipif(not HAS_LMMS, reason="requires lmms-eval")
def test_baseline_5_samples_pope(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path = repo_root / "vtp_eval" / "adapters"
    out_dir = tmp_path / "smoke"
    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--include_path", str(adapter_path),
        "--model", "llava_baseline",
        "--model_args", "pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager",
        "--tasks", "pope",
        "--limit", "5",
        "--batch_size", "1",
        "--output_path", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"lmms-eval exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-2000:]}\n"
        f"STDERR:\n{proc.stderr[-2000:]}"
    )

    results_json = next(out_dir.rglob("results.json"), None)
    assert results_json is not None, list(out_dir.rglob("*"))
    data = json.loads(results_json.read_text())
    assert "pope" in data["results"]
    assert "pope_accuracy" in data["results"]["pope"]
