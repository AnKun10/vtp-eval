# vtp-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin Python package (`vtp-eval`) that wraps 5 visual-token-pruning methods (FastV, SparseVLMs, VisionZip, DivPrune, SparseVILA) + a no-prune baseline as lmms-eval model adapters, plus a Colab L4 notebook that runs POPE benchmark on all 6 and aggregates results into a single CSV with accuracy, latency, peak VRAM, and theoretical TFLOPs.

**Architecture:** Each pruning method = one `lmms_eval.api.model.lmms` subclass (mostly inheriting from `lmms_eval.models.simple.llava.Llava`). Adapters are loaded into lmms-eval via the built-in `--include_path` flag — no fork of lmms-eval needed. Each method gets a dedicated install shell script because the 5 methods patch overlapping locations of `transformers`/`llava` packages and cannot coexist in one environment; the notebook runs them sequentially, restarting the Colab kernel between methods.

**Tech Stack:** Python 3.10+, `lmms-eval` (latest), `transformers==4.37.2` (pinned for SparseVILA/FastV/DivPrune compat), `torch>=2.4`, `pytest`, PyYAML, pandas, matplotlib. Colab Pro with L4 GPU (24 GB).

**Working directory:** `E:\Workspaces\My Projects\DATN\vtp-eval\` (does not exist yet; Task 1 creates it).

**Reference spec:** `docs/superpowers/specs/2026-05-16-vtp-eval-design.md`.

---

## Task 1: Scaffold workspace, git init, pyproject

**Files:**
- Create: `vtp-eval/.gitignore`
- Create: `vtp-eval/pyproject.toml`
- Create: `vtp-eval/README.md`
- Create: `vtp-eval/vtp_eval/__init__.py`
- Create: `vtp-eval/vtp_eval/adapters/__init__.py`
- Create: `vtp-eval/vtp_eval/utils/__init__.py`
- Create: `vtp-eval/tests/__init__.py`
- Create: `vtp-eval/results/.gitkeep`
- Create: `vtp-eval/configs/.gitkeep`
- Create: `vtp-eval/install/.gitkeep`
- Create: `vtp-eval/scripts/.gitkeep`
- Create: `vtp-eval/notebooks/.gitkeep`

- [ ] **Step 1: Create directory structure and init git**

Run from `E:\Workspaces\My Projects\DATN`:
```bash
mkdir -p vtp-eval/vtp_eval/adapters vtp-eval/vtp_eval/utils \
         vtp-eval/tests vtp-eval/install vtp-eval/configs \
         vtp-eval/scripts vtp-eval/notebooks vtp-eval/results \
         "vtp-eval/docs/superpowers/specs" "vtp-eval/docs/superpowers/plans"
cd vtp-eval
git init
```

The spec + plan files already live under `vtp-eval/docs/superpowers/` — they were written there during brainstorming. Verify they're there:
```bash
ls docs/superpowers/specs/ docs/superpowers/plans/
```

- [ ] **Step 2: Write `.gitignore`**

Create `vtp-eval/.gitignore`:
```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/
.venv/
venv/
build/
dist/

# Run outputs (everything except .gitkeep)
results/*
!results/.gitkeep

# Local data
*.jsonl
*.log
```

- [ ] **Step 3: Write `pyproject.toml`**

Create `vtp-eval/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vtp-eval"
version = "0.1.0"
description = "Visual Token Pruning Evaluation harness for LLaVA-1.5 via lmms-eval"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
    "pandas>=2.0",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
test = ["pytest>=7.4"]

[tool.setuptools.packages.find]
where = ["."]
include = ["vtp_eval*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: integration tests that require GPU + LLaVA weights"]
```

- [ ] **Step 4: Write package init files**

Create `vtp-eval/vtp_eval/__init__.py`:
```python
"""vtp-eval — Visual Token Pruning evaluation harness."""
__version__ = "0.1.0"
```

Create `vtp-eval/vtp_eval/adapters/__init__.py`:
```python
"""lmms-eval adapter modules for visual token pruning methods.

Each submodule registers a model adapter via @register_model decorator.
Import-side-effects intentional: importing this package registers all adapters.
"""
# Adapters imported as we add them in subsequent tasks
```

Create `vtp-eval/vtp_eval/utils/__init__.py`:
```python
"""Utility modules for vtp-eval."""
```

Create `vtp-eval/tests/__init__.py`:
```python
```

- [ ] **Step 5: Write minimal `README.md`**

Create `vtp-eval/README.md`:
```markdown
# vtp-eval

Visual Token Pruning evaluation harness for LLaVA-1.5 7B on POPE, via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

Compares: baseline (no prune), FastV, SparseVLMs, VisionZip, DivPrune, SparseVILA.

## Quick start

Open `notebooks/pope_eval.ipynb` in Colab Pro (L4 GPU), run cells sequentially. Between methods you must **restart runtime** (notebook markdown cells flag where).

Design + plan: `docs/superpowers/`.
```

- [ ] **Step 6: Create `.gitkeep` files for empty dirs**

```bash
touch results/.gitkeep configs/.gitkeep install/.gitkeep \
      scripts/.gitkeep notebooks/.gitkeep
```

- [ ] **Step 7: First commit**

```bash
git add .gitignore pyproject.toml README.md vtp_eval/ tests/ \
        results/.gitkeep configs/.gitkeep install/.gitkeep \
        scripts/.gitkeep notebooks/.gitkeep docs/
git commit -m "feat: scaffold vtp-eval workspace"
```

- [ ] **Step 8: Verify pip install works**

```bash
pip install -e ".[test]"
```
Expected: `Successfully installed vtp-eval-0.1.0`. If it fails because `pip` cannot find `setuptools>=68`, run `pip install -U setuptools wheel` first.

```bash
python -c "import vtp_eval; print(vtp_eval.__version__)"
```
Expected: `0.1.0`

---

## Task 2: `tflops.py` — theoretical FLOPs computation (TDD)

**Files:**
- Create: `vtp-eval/vtp_eval/utils/tflops.py`
- Create: `vtp-eval/tests/test_tflops.py`

- [ ] **Step 1: Write failing tests for `tflops.py`**

Create `vtp-eval/tests/test_tflops.py`:
```python
import pytest
from vtp_eval.utils.tflops import (
    LLaMAConfig, LLAVA_15_7B, layer_flops, total_llm_flops,
    shape_baseline, shape_fastv, shape_sparsevlm,
    shape_visionzip, shape_divprune, shape_sparsevila,
    clip_encoder_flops, compute_method_tflops,
)


def test_llama_config_defaults_match_llava_15_7b():
    assert LLAVA_15_7B.n_layers == 32
    assert LLAVA_15_7B.d_model == 4096
    assert LLAVA_15_7B.d_ffn == 11008


def test_layer_flops_positive():
    assert layer_flops(622) > 0  # 35 sys + 576 visual + 10 text_q + 1 text_a


def test_total_llm_flops_baseline_in_expected_range():
    """Baseline LLaVA-1.5 7B prefill on POPE (~622 token sequence) ≈ 5–10 TFLOPs."""
    total = total_llm_flops(shape_baseline())
    tflops = total / 1e12
    assert 5.0 < tflops < 12.0, f"baseline LLM tflops = {tflops:.2f}"


def test_shape_baseline_length():
    assert len(shape_baseline()) == 32
    assert all(v == 576 for v in shape_baseline())


def test_shape_fastv_K2_R128():
    shape = shape_fastv(K=2, R=128)
    assert len(shape) == 32
    assert shape[:3] == [576, 576, 576]
    assert shape[3:] == [128] * 29


def test_shape_sparsevlm_192_matches_source_schedule():
    """Reference: SparseVLMs/llava/model/language_model/score.py:11 comment.
       '2*576 4*300 10*200 16*110' for retain=192."""
    shape = shape_sparsevlm(192)
    assert len(shape) == 32
    assert shape[:2] == [576, 576]
    assert shape[2:6] == [300, 300, 300, 300]
    assert shape[6:16] == [200] * 10
    assert shape[16:] == [110] * 16


def test_shape_visionzip_pre_llm():
    shape = shape_visionzip(dominant=54, contextual=10)
    assert all(v == 64 for v in shape)


def test_shape_divprune_pre_llm():
    shape = shape_divprune(ratio=0.098)
    expected = int(576 * 0.098)
    assert all(v == expected for v in shape)


def test_shape_sparsevila_encoder_prune():
    shape = shape_sparsevila(enc_ratio=0.5, dec_ratio=0.5)
    expected = int(576 * 0.5)
    assert all(v == expected for v in shape)


def test_clip_encoder_flops_positive():
    assert clip_encoder_flops() > 0


def test_pruning_reduces_total_flops():
    base = compute_method_tflops("baseline")["tflops_total"]
    for method, kw in [
        ("fastv", dict(K=2, R=128)),
        ("sparsevlm", dict(retain=192)),
        ("visionzip", dict(dominant=54, contextual=10)),
        ("divprune", dict(ratio=0.098)),
        ("sparsevila", dict(enc_ratio=0.5, dec_ratio=0.5)),
    ]:
        r = compute_method_tflops(method, **kw)
        assert r["tflops_total"] < base, (
            f"{method} did not reduce flops: {r['tflops_total']} vs {base}"
        )


def test_visionzip_more_aggressive_than_fastv():
    vz = compute_method_tflops("visionzip", dominant=54, contextual=10)
    fv = compute_method_tflops("fastv", K=2, R=128)
    assert vz["tflops_total"] < fv["tflops_total"]


def test_compute_method_tflops_unknown_method_raises():
    with pytest.raises(KeyError):
        compute_method_tflops("not_a_method")
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/test_tflops.py -v
```
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'vtp_eval.utils.tflops'`.

- [ ] **Step 3: Implement `tflops.py`**

Create `vtp-eval/vtp_eval/utils/tflops.py`:
```python
"""Theoretical prefill FLOPs estimates for LLaVA-1.5 7B with visual-token pruning.

Approximation only — ignores RMSNorm/RoPE/SwiGLU activation, treats attention
as full quadratic (no causal mask discount). Sufficient for relative comparison.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class LLaMAConfig:
    n_layers: int = 32
    d_model: int = 4096
    d_ffn: int = 11008  # SwiGLU intermediate
    n_heads: int = 32


LLAVA_15_7B = LLaMAConfig()

# POPE prompt layout
SYS_LEN = 35
VISUAL_FULL = 576
TEXT_Q = 10
TEXT_A = 1


def layer_flops(seq_len: int, cfg: LLaMAConfig = LLAVA_15_7B) -> float:
    """Prefill FLOPs for one transformer layer at given seq_len."""
    d, m = cfg.d_model, cfg.d_ffn
    attn_proj = 4 * seq_len * d * d            # Q, K, V, O linear
    attn_mm   = 2 * seq_len * seq_len * d      # QK^T + softmax · V
    ffn       = 3 * seq_len * d * m            # SwiGLU: gate + up + down
    return float(attn_proj + attn_mm + ffn)


def total_llm_flops(visual_per_layer: List[int],
                    sys_len: int = SYS_LEN,
                    text_q: int = TEXT_Q,
                    text_a: int = TEXT_A,
                    cfg: LLaMAConfig = LLAVA_15_7B) -> float:
    """Sum prefill FLOPs across all LLM layers.

    visual_per_layer[i] is the visual-token count at layer i.
    """
    return sum(
        layer_flops(sys_len + v + text_q + text_a, cfg)
        for v in visual_per_layer
    )


# Visual-token-per-layer shapes per method ---------------------------------

def shape_baseline() -> List[int]:
    return [VISUAL_FULL] * 32


def shape_fastv(K: int, R: int) -> List[int]:
    """FastV prunes after layer K (1-indexed exclusive: layer K still sees all)."""
    return [VISUAL_FULL] * (K + 1) + [R] * (32 - K - 1)


def shape_sparsevlm(retain: int) -> List[int]:
    """SparseVLMs progressive schedule (verified from score.py:11-14).

    For retain=192: '2*576 4*300 10*200 16*110' (from source comment).
    """
    schedule = {
        192: (300, 200, 110),
        128: (303, 110, 36),
        96:  (238, 48,  26),
        64:  (66,  30,  17),
    }
    s = schedule[retain]
    return [VISUAL_FULL] * 2 + [s[0]] * 4 + [s[1]] * 10 + [s[2]] * 16


def shape_visionzip(dominant: int, contextual: int) -> List[int]:
    """VisionZip prunes before LLM — same count across all layers."""
    return [dominant + contextual] * 32


def shape_divprune(ratio: float) -> List[int]:
    """DivPrune prunes at projector (pre-LLM)."""
    return [int(VISUAL_FULL * ratio)] * 32


def shape_sparsevila(enc_ratio: float, dec_ratio: float) -> List[int]:
    """SparseVILA encoder prune dominates prefill cost.

    Decode-time retrieval (dec_ratio) reduces decode FLOPs but not prefill,
    so it doesn't enter this prefill formula.
    """
    return [int(VISUAL_FULL * (1 - enc_ratio))] * 32


def clip_encoder_flops(visual_keep: int = 577) -> float:
    """CLIP-ViT-L/14: 24 layers, d=1024, d_ffn=4096."""
    cfg = LLaMAConfig(n_layers=24, d_model=1024, d_ffn=4096, n_heads=16)
    return cfg.n_layers * layer_flops(visual_keep, cfg)


# Top-level facade --------------------------------------------------------

_SHAPE_FNS = {
    "baseline":   lambda **_: shape_baseline(),
    "fastv":      lambda K, R, **_: shape_fastv(K, R),
    "sparsevlm":  lambda retain, **_: shape_sparsevlm(retain),
    "visionzip":  lambda dominant, contextual, **_: shape_visionzip(dominant, contextual),
    "divprune":   lambda ratio, **_: shape_divprune(ratio),
    "sparsevila": lambda enc_ratio, dec_ratio, **_: shape_sparsevila(enc_ratio, dec_ratio),
}


def compute_method_tflops(method: str, **kwargs) -> dict:
    """Compute prefill TFLOPs for a method given its pruning args.

    Returns: dict with keys
      tflops_llm_prefill, tflops_encoder, tflops_total, visual_shape
    """
    if method not in _SHAPE_FNS:
        raise KeyError(f"Unknown method: {method!r}. "
                       f"Choices: {sorted(_SHAPE_FNS.keys())}")
    visual_shape = _SHAPE_FNS[method](**kwargs)
    llm = total_llm_flops(visual_shape)
    enc = clip_encoder_flops()
    return {
        "tflops_llm_prefill": llm / 1e12,
        "tflops_encoder":     enc / 1e12,
        "tflops_total":       (llm + enc) / 1e12,
        "visual_shape":       visual_shape,
    }
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/test_tflops.py -v
```
Expected: 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/utils/tflops.py tests/test_tflops.py
git commit -m "feat(utils): theoretical TFLOPs computation per method"
```

---

## Task 3: `result_io.py` — aggregate per-run results into summary CSV (TDD)

**Files:**
- Create: `vtp-eval/vtp_eval/utils/result_io.py`
- Create: `vtp-eval/tests/test_result_io.py`

- [ ] **Step 1: Write failing tests**

Create `vtp-eval/tests/test_result_io.py`:
```python
import json
from pathlib import Path

import pandas as pd
import pytest

from vtp_eval.utils.result_io import aggregate, compute_keep_ratio


def _write_run(run_dir: Path, results: dict, timing: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results))
    (run_dir / "timing.json").write_text(json.dumps(timing))


@pytest.fixture
def fake_results(tmp_path: Path) -> Path:
    """Two runs: baseline + fastv."""
    _write_run(
        tmp_path / "baseline",
        results={"results": {"pope": {
            "pope_accuracy": 0.88, "pope_f1_score": 0.876,
            "pope_yes_ratio": 0.51,
        }}},
        timing={
            "method": "baseline", "n_samples": 5,
            "latency_per_sample_ms": {"mean": 287.0, "p95": 389.0},
            "peak_gpu_mem_mb": 14210,
            "pruning_meta": {"method": "baseline", "keep_ratio": 1.0},
        },
    )
    _write_run(
        tmp_path / "fastv_K2_R128",
        results={"results": {"pope": {
            "pope_accuracy": 0.87, "pope_f1_score": 0.867,
            "pope_yes_ratio": 0.50,
        }}},
        timing={
            "method": "fastv", "n_samples": 5,
            "latency_per_sample_ms": {"mean": 205.0, "p95": 287.0},
            "peak_gpu_mem_mb": 13420,
            "pruning_meta": {"method": "fastv", "K": 2, "R": 128},
        },
    )
    return tmp_path


def test_aggregate_writes_all_expected_columns(fake_results, tmp_path):
    out = fake_results / "summary.csv"
    aggregate(fake_results, out)
    df = pd.read_csv(out)
    expected_cols = {
        "method", "pope_acc", "pope_f1", "pope_yes_ratio",
        "latency_ms_mean", "latency_ms_p95", "peak_mem_mb",
        "keep_ratio_pct", "tflops_prefill", "tflops_reduction_pct",
    }
    assert expected_cols.issubset(df.columns), df.columns.tolist()


def test_aggregate_no_nan(fake_results, tmp_path):
    out = fake_results / "summary.csv"
    aggregate(fake_results, out)
    df = pd.read_csv(out)
    assert not df.isna().any().any(), df


def test_aggregate_baseline_tflops_reduction_is_zero(fake_results, tmp_path):
    out = fake_results / "summary.csv"
    aggregate(fake_results, out)
    df = pd.read_csv(out).set_index("method")
    assert df.loc["baseline", "tflops_reduction_pct"] == pytest.approx(0.0)


def test_aggregate_fastv_has_positive_tflops_reduction(fake_results, tmp_path):
    out = fake_results / "summary.csv"
    aggregate(fake_results, out)
    df = pd.read_csv(out).set_index("method")
    assert df.loc["fastv_K2_R128", "tflops_reduction_pct"] > 0


def test_aggregate_skips_runs_missing_results_json(tmp_path):
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "timing.json").write_text("{}")  # no results.json
    out = tmp_path / "summary.csv"
    aggregate(tmp_path, out)
    df = pd.read_csv(out)
    assert "partial" not in df["method"].values


def test_compute_keep_ratio_baseline():
    assert compute_keep_ratio({"method": "baseline"}) == pytest.approx(100.0)


def test_compute_keep_ratio_fastv():
    # FastV keeps R out of 576 tokens after the prune layer
    r = compute_keep_ratio({"method": "fastv", "K": 2, "R": 128})
    assert 20.0 < r < 25.0  # 128/576 ≈ 22.2%


def test_compute_keep_ratio_visionzip():
    r = compute_keep_ratio({"method": "visionzip", "dominant": 54, "contextual": 10})
    assert 10.0 < r < 12.0  # 64/576 ≈ 11.1%


def test_compute_keep_ratio_divprune():
    r = compute_keep_ratio({"method": "divprune", "ratio": 0.098})
    assert r == pytest.approx(9.8)


def test_compute_keep_ratio_sparsevila():
    r = compute_keep_ratio({"method": "sparsevila",
                             "enc_ratio": 0.5, "dec_ratio": 0.5})
    assert r == pytest.approx(50.0)  # encoder prune determines visible tokens


def test_compute_keep_ratio_sparsevlm():
    r = compute_keep_ratio({"method": "sparsevlm", "retain": 192})
    # Average across schedule [576]*2 + [300]*4 + [200]*10 + [110]*16
    # = (2*576 + 4*300 + 10*200 + 16*110) / 32 / 576 * 100
    # = (1152 + 1200 + 2000 + 1760) / 32 / 576 * 100
    # = 6112 / 32 / 576 * 100 ≈ 33.2
    assert 30.0 < r < 36.0
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/test_result_io.py -v
```
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `result_io.py`**

Create `vtp-eval/vtp_eval/utils/result_io.py`:
```python
"""Aggregate per-run results.json + timing.json into a single summary.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from vtp_eval.utils.tflops import (
    compute_method_tflops, shape_sparsevlm, VISUAL_FULL,
)


def compute_keep_ratio(meta: Dict) -> float:
    """Percent of visual tokens retained, averaged across layers if dynamic."""
    method = meta["method"]
    if method == "baseline":
        return 100.0
    if method == "fastv":
        # After layer K only R of 576 tokens survive.
        K, R = int(meta["K"]), int(meta["R"])
        kept = ((K + 1) * VISUAL_FULL + (32 - K - 1) * R) / 32
        return 100.0 * kept / VISUAL_FULL
    if method == "sparsevlm":
        shape = shape_sparsevlm(int(meta["retain"]))
        return 100.0 * (sum(shape) / len(shape)) / VISUAL_FULL
    if method == "visionzip":
        kept = int(meta["dominant"]) + int(meta["contextual"])
        return 100.0 * kept / VISUAL_FULL
    if method == "divprune":
        return 100.0 * float(meta["ratio"])
    if method == "sparsevila":
        return 100.0 * (1 - float(meta["enc_ratio"]))
    raise KeyError(f"Unknown method for keep_ratio: {method!r}")


def _tflops_kwargs_from_meta(meta: Dict) -> Dict:
    """Map pruning_meta dict to compute_method_tflops kwargs."""
    method = meta["method"]
    if method == "baseline":
        return {}
    if method == "fastv":
        return {"K": int(meta["K"]), "R": int(meta["R"])}
    if method == "sparsevlm":
        return {"retain": int(meta["retain"])}
    if method == "visionzip":
        return {"dominant": int(meta["dominant"]),
                "contextual": int(meta["contextual"])}
    if method == "divprune":
        return {"ratio": float(meta["ratio"])}
    if method == "sparsevila":
        return {"enc_ratio": float(meta["enc_ratio"]),
                "dec_ratio": float(meta["dec_ratio"])}
    raise KeyError(f"Unknown method: {method!r}")


def _row_from_run(run_dir: Path) -> Optional[Dict]:
    res_p = run_dir / "results.json"
    tim_p = run_dir / "timing.json"
    if not res_p.exists() or not tim_p.exists():
        return None
    results = json.loads(res_p.read_text())
    timing = json.loads(tim_p.read_text())
    pope = results["results"]["pope"]
    meta = timing["pruning_meta"]
    tf = compute_method_tflops(meta["method"], **_tflops_kwargs_from_meta(meta))
    return {
        "method": run_dir.name,
        "pope_acc": pope["pope_accuracy"],
        "pope_f1": pope["pope_f1_score"],
        "pope_yes_ratio": pope["pope_yes_ratio"],
        "latency_ms_mean": timing["latency_per_sample_ms"]["mean"],
        "latency_ms_p95": timing["latency_per_sample_ms"]["p95"],
        "peak_mem_mb": timing["peak_gpu_mem_mb"],
        "keep_ratio_pct": compute_keep_ratio(meta),
        "tflops_prefill": tf["tflops_total"],
        "_method_key": meta["method"],
    }


def aggregate(results_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Walk results_dir/*/results.json+timing.json → write summary.csv."""
    results_dir = Path(results_dir)
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        row = _row_from_run(run_dir)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    # Add tflops_reduction_pct relative to baseline if present
    baseline = df[df["_method_key"] == "baseline"]
    if not baseline.empty:
        base_tf = float(baseline["tflops_prefill"].iloc[0])
        df["tflops_reduction_pct"] = (1 - df["tflops_prefill"] / base_tf) * 100
    else:
        df["tflops_reduction_pct"] = float("nan")
    df = df.drop(columns=["_method_key"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _cli() -> None:
    p = argparse.ArgumentParser(prog="vtp_eval.utils.result_io")
    sub = p.add_subparsers(dest="cmd", required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("results_dir", type=Path)
    agg.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "aggregate":
        df = aggregate(args.results_dir, args.output)
        print(df.to_string(index=False))


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/test_result_io.py -v
```
Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/utils/result_io.py tests/test_result_io.py
git commit -m "feat(utils): aggregate per-run results to summary CSV"
```

---

## Task 4: `timing.py` — latency / VRAM measurement + log parser

**Files:**
- Create: `vtp-eval/vtp_eval/utils/timing.py`
- Create: `vtp-eval/tests/test_timing.py`

- [ ] **Step 1: Write failing tests**

Create `vtp-eval/tests/test_timing.py`:
```python
import json
from pathlib import Path

import pytest

from vtp_eval.utils.timing import TimingHook, parse_log_to_timing_json


def test_timing_hook_records_zero_when_empty():
    h = TimingHook()
    summary = h.summary()
    assert summary["n_samples"] == 0
    assert summary["latency_per_sample_ms"]["mean"] == 0.0


def test_timing_hook_records_after_measure():
    """We can't measure real CUDA events on CPU. Test the API contract:
    record() accepts (latency_ms, peak_mem_mb) and the summary aggregates."""
    h = TimingHook()
    h.record(latency_ms=100.0, peak_mem_mb=12000)
    h.record(latency_ms=200.0, peak_mem_mb=12500)
    s = h.summary()
    assert s["n_samples"] == 2
    assert s["latency_per_sample_ms"]["mean"] == pytest.approx(150.0)
    assert s["latency_per_sample_ms"]["p50"] == pytest.approx(150.0)
    assert s["peak_gpu_mem_mb"] == 12500


def test_parse_log_to_timing_json_writes_file(tmp_path):
    """Parser reads a single sidecar JSON containing both per-sample latencies
    and pruning_meta — produced by TimingHook.dump() during a run."""
    sidecar = tmp_path / "timing_raw.json"
    sidecar.write_text(json.dumps({
        "samples": [{"latency_ms": 200.0, "peak_mem_mb": 13000},
                    {"latency_ms": 210.0, "peak_mem_mb": 13100}],
        "pruning_meta": {"method": "fastv", "K": 2, "R": 128},
    }))
    out = tmp_path / "timing.json"
    parse_log_to_timing_json(sidecar, out)
    data = json.loads(out.read_text())
    assert data["method"] == "fastv"
    assert data["n_samples"] == 2
    assert data["pruning_meta"]["K"] == 2
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/test_timing.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `timing.py`**

Create `vtp-eval/vtp_eval/utils/timing.py`:
```python
"""Latency + peak VRAM measurement for adapter generate_until() calls.

The adapter creates a TimingHook in __init__ and calls .measure() around each
generate batch. After the run, the hook dumps a raw sidecar JSON; a separate
post-run step (`parse_log_to_timing_json`) combines that with the lmms-eval
results.json to produce the final timing.json that result_io expects.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List


class TimingHook:
    """Records per-sample latency and peak memory.

    On GPU, .measure() uses torch.cuda.Event + max_memory_allocated().
    Without CUDA available, falls back to time.perf_counter() and skips memory.
    """

    def __init__(self) -> None:
        self._samples: List[Dict] = []
        self.pruning_meta: Dict = {}

    @contextmanager
    def measure(self):
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            self.record(
                latency_ms=start.elapsed_time(end),
                peak_mem_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
            )
        else:
            t0 = time.perf_counter()
            yield
            self.record(
                latency_ms=(time.perf_counter() - t0) * 1000,
                peak_mem_mb=0,
            )

    def record(self, latency_ms: float, peak_mem_mb: float) -> None:
        self._samples.append({
            "latency_ms": float(latency_ms),
            "peak_mem_mb": float(peak_mem_mb),
        })

    def summary(self) -> Dict:
        if not self._samples:
            return {
                "n_samples": 0,
                "latency_per_sample_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
                "peak_gpu_mem_mb": 0.0,
            }
        lats = sorted(s["latency_ms"] for s in self._samples)
        n = len(lats)
        p50 = lats[n // 2]
        p95 = lats[min(int(n * 0.95), n - 1)]
        peak = max(s["peak_mem_mb"] for s in self._samples)
        return {
            "n_samples": n,
            "latency_per_sample_ms": {
                "mean": statistics.fmean(lats),
                "p50": p50,
                "p95": p95,
            },
            "peak_gpu_mem_mb": peak,
        }

    def dump(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "samples": self._samples,
            "pruning_meta": self.pruning_meta,
        }, indent=2))


def parse_log_to_timing_json(sidecar: Path, output: Path) -> None:
    """Convert a TimingHook sidecar (samples + pruning_meta) to final timing.json."""
    raw = json.loads(Path(sidecar).read_text())
    samples = raw.get("samples", [])
    meta = raw.get("pruning_meta", {"method": "unknown"})
    if samples:
        lats = sorted(s["latency_ms"] for s in samples)
        n = len(lats)
        out = {
            "method": meta.get("method", "unknown"),
            "n_samples": n,
            "total_wall_s": sum(lats) / 1000,
            "latency_per_sample_ms": {
                "mean": statistics.fmean(lats),
                "p50": lats[n // 2],
                "p95": lats[min(int(n * 0.95), n - 1)],
            },
            "peak_gpu_mem_mb": max(s["peak_mem_mb"] for s in samples),
            "pruning_meta": meta,
        }
    else:
        out = {
            "method": meta.get("method", "unknown"),
            "n_samples": 0,
            "total_wall_s": 0.0,
            "latency_per_sample_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
            "peak_gpu_mem_mb": 0.0,
            "pruning_meta": meta,
        }
    Path(output).write_text(json.dumps(out, indent=2))


def _cli() -> None:
    p = argparse.ArgumentParser(prog="vtp_eval.utils.timing")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("parse-sidecar")
    pl.add_argument("--sidecar", type=Path, required=True)
    pl.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "parse-sidecar":
        parse_log_to_timing_json(args.sidecar, args.output)


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/test_timing.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/utils/timing.py tests/test_timing.py
git commit -m "feat(utils): per-sample latency + peak VRAM hook"
```

---

## Task 5: Adapter base class `_base.py`

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/_base.py`

**Note:** Cannot TDD this directly because importing `lmms_eval.models.simple.llava` requires the `llava` package + a real model checkpoint at module load time. We'll smoke-test via `test_adapters_load.py` in Task 6.

- [ ] **Step 1: Implement base class**

Create `vtp-eval/vtp_eval/adapters/_base.py`:
```python
"""Shared base class for all visual-token-pruning adapters.

Inherits from lmms-eval's `Llava` adapter — overrides only generate_until to
inject the timing hook. Subclasses add pruning-specific config in their own
__init__ AFTER calling super().__init__.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

from lmms_eval.models.simple.llava import Llava

from vtp_eval.utils.timing import TimingHook


class LlavaPruningBase(Llava):
    """Llava adapter + per-batch timing + pruning_meta plumbing."""

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        log_timing: bool = True,
        timing_sidecar: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
        self.timing_hook: Optional[TimingHook] = (
            TimingHook() if log_timing else None
        )
        self.timing_sidecar_path = (
            Path(timing_sidecar) if timing_sidecar else None
        )
        self.pruning_meta: Dict = {}

    def generate_until(self, requests):
        ctx = (self.timing_hook.measure()
               if self.timing_hook is not None else nullcontext())
        with ctx:
            out = super().generate_until(requests)
        # Dump sidecar after each batch (cheap, ensures partial results survive crash)
        if self.timing_hook is not None and self.timing_sidecar_path is not None:
            self.timing_hook.pruning_meta = self.pruning_meta
            self.timing_hook.dump(self.timing_sidecar_path)
        return out
```

- [ ] **Step 2: Verify import works (lmms-eval must be importable; skip if not yet installed)**

```bash
python -c "
try:
    from vtp_eval.adapters._base import LlavaPruningBase
    print('OK')
except ModuleNotFoundError as e:
    print(f'lmms-eval not installed yet (expected if before notebook): {e}')
"
```
Either output is acceptable at this stage — lmms-eval is installed on Colab in the notebook, not locally.

- [ ] **Step 3: Commit**

```bash
git add vtp_eval/adapters/_base.py
git commit -m "feat(adapters): base class with timing hook"
```

---

## Task 6: Baseline adapter + `test_adapters_load.py` skeleton

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/llava_baseline.py`
- Modify: `vtp-eval/vtp_eval/adapters/__init__.py`
- Create: `vtp-eval/tests/test_adapters_load.py`

- [ ] **Step 1: Write the adapter loading test (starts empty, grows per task)**

Create `vtp-eval/tests/test_adapters_load.py`:
```python
"""Smoke tests: each adapter must register into lmms-eval and expose
a `pretrained` kwarg. Adapters are added to ADAPTERS as we implement them.
"""
import importlib
import inspect

import pytest

pytest.importorskip("lmms_eval", reason="lmms-eval not installed locally")

ADAPTERS = [
    "llava_baseline",
    # Added in later tasks:
    # "llava_fastv", "llava_sparsevlm", "llava_visionzip",
    # "llava_divprune", "llava_sparsevila",
]


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_registered(name):
    """After importing vtp_eval.adapters, the adapter is in lmms-eval registry."""
    importlib.import_module("vtp_eval.adapters")
    from lmms_eval.api.registry import MODEL_REGISTRY
    assert name in MODEL_REGISTRY, (
        f"adapter {name!r} not in MODEL_REGISTRY. "
        f"Present: {sorted(MODEL_REGISTRY.keys())[:20]}..."
    )


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_signature_has_pretrained(name):
    importlib.import_module("vtp_eval.adapters")
    from lmms_eval.api.registry import MODEL_REGISTRY
    cls = MODEL_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    assert "pretrained" in sig.parameters
```

- [ ] **Step 2: Run tests, verify they fail or skip**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED (lmms-eval not installed locally) **OR** FAIL with `KeyError: 'llava_baseline'`.

Either outcome is fine — this test runs for real on Colab/CI, locally we just want it to not crash.

- [ ] **Step 3: Implement baseline adapter**

Create `vtp-eval/vtp_eval/adapters/llava_baseline.py`:
```python
"""Baseline LLaVA-1.5 adapter — no pruning. Reference point for comparison."""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_baseline")
class LlavaBaseline(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b", **kw):
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "baseline", "keep_ratio": 1.0}
```

- [ ] **Step 4: Wire it into the adapters package**

Modify `vtp-eval/vtp_eval/adapters/__init__.py` to import baseline (so registration fires on package import):
```python
"""lmms-eval adapter modules for visual token pruning methods.

Each submodule registers a model adapter via @register_model decorator.
Importing this package registers all adapters in lmms-eval's MODEL_REGISTRY.
"""
from vtp_eval.adapters import llava_baseline  # noqa: F401  (registers adapter)
```

- [ ] **Step 5: Run tests (skip locally is fine)**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED locally, PASS on Colab (where lmms-eval is installed).

- [ ] **Step 6: Commit**

```bash
git add vtp_eval/adapters/llava_baseline.py vtp_eval/adapters/__init__.py \
        tests/test_adapters_load.py
git commit -m "feat(adapters): baseline adapter + registration test scaffold"
```

---

## Task 7: FastV adapter

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/llava_fastv.py`
- Modify: `vtp-eval/vtp_eval/adapters/__init__.py`
- Modify: `vtp-eval/tests/test_adapters_load.py`

- [ ] **Step 1: Add `llava_fastv` to the test ADAPTERS list**

Modify `vtp-eval/tests/test_adapters_load.py` — find the line:
```python
    "llava_baseline",
    # Added in later tasks:
    # "llava_fastv", "llava_sparsevlm", "llava_visionzip",
    # "llava_divprune", "llava_sparsevila",
```
Replace with:
```python
    "llava_baseline",
    "llava_fastv",
    # Added in later tasks:
    # "llava_sparsevlm", "llava_visionzip",
    # "llava_divprune", "llava_sparsevila",
```

- [ ] **Step 2: Implement the adapter**

Create `vtp-eval/vtp_eval/adapters/llava_fastv.py`:
```python
"""FastV adapter — token pruning AFTER layer K based on attention rank.

Reference: FastV (https://github.com/pkunlp-icler/FastV).
Requires the FastV-patched modeling_llama.py in the active transformers install,
applied via install/_patch_fastv.py. With patch, the LlamaModel reads these
fields on model.config and runs the pruning forward path.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_fastv")
class LlavaFastV(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        fast_v_agg_layer: int = 2,
        fast_v_attention_rank: int = 128,
        fast_v_image_token_length: int = 576,
        **kw,
    ):
        super().__init__(pretrained=pretrained, **kw)
        self._model.config.use_fast_v = True
        self._model.config.fast_v_agg_layer = int(fast_v_agg_layer)
        self._model.config.fast_v_attention_rank = int(fast_v_attention_rank)
        self._model.config.fast_v_sys_length = None
        self._model.config.fast_v_image_token_length = int(fast_v_image_token_length)
        # Reset internal state; method comes from the FastV patch
        if hasattr(self._model.model, "reset_fastv"):
            self._model.model.reset_fastv()
        self.pruning_meta = {
            "method": "fastv",
            "K": int(fast_v_agg_layer),
            "R": int(fast_v_attention_rank),
        }
```

- [ ] **Step 3: Wire into package init**

Modify `vtp-eval/vtp_eval/adapters/__init__.py` — append after the baseline import:
```python
from vtp_eval.adapters import llava_fastv      # noqa: F401
```
Full file now reads:
```python
"""lmms-eval adapter modules for visual token pruning methods.

Each submodule registers a model adapter via @register_model decorator.
Importing this package registers all adapters in lmms-eval's MODEL_REGISTRY.
"""
from vtp_eval.adapters import llava_baseline   # noqa: F401
from vtp_eval.adapters import llava_fastv      # noqa: F401
```

- [ ] **Step 4: Run tests (skip locally OK)**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED locally; PASS on Colab.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/adapters/llava_fastv.py vtp_eval/adapters/__init__.py \
        tests/test_adapters_load.py
git commit -m "feat(adapters): FastV adapter"
```

---

## Task 8: SparseVLM adapter

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/llava_sparsevlm.py`
- Modify: `vtp-eval/vtp_eval/adapters/__init__.py`
- Modify: `vtp-eval/tests/test_adapters_load.py`

- [ ] **Step 1: Add to ADAPTERS list**

Modify `vtp-eval/tests/test_adapters_load.py` — find:
```python
    "llava_baseline",
    "llava_fastv",
    # Added in later tasks:
    # "llava_sparsevlm", "llava_visionzip",
    # "llava_divprune", "llava_sparsevila",
```
Replace with:
```python
    "llava_baseline",
    "llava_fastv",
    "llava_sparsevlm",
    # Added in later tasks:
    # "llava_visionzip", "llava_divprune", "llava_sparsevila",
```

- [ ] **Step 2: Implement adapter**

Create `vtp-eval/vtp_eval/adapters/llava_sparsevlm.py`:
```python
"""SparseVLMs adapter — text-guided multi-layer pruning at layers 2/6/15.

Reference: SparseVLMs (https://github.com/Gumpest/SparseVLMs).
Pruning is controlled by env var RETAIN_TOKN read on import inside
SparseVLMs/llava/model/language_model/score.py. We set it BEFORE super()
init triggers the LLaVA import chain.
"""
import os

from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_sparsevlm")
class LlavaSparseVLM(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        retain_token: int = 192,
        **kw,
    ):
        if int(retain_token) not in {64, 96, 128, 192}:
            raise ValueError(
                f"retain_token must be one of 64/96/128/192 (per SparseVLMs "
                f"sparse_token_dict), got {retain_token}"
            )
        os.environ["RETAIN_TOKN"] = str(int(retain_token))
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "sparsevlm", "retain": int(retain_token)}
```

- [ ] **Step 3: Wire into package**

Modify `vtp-eval/vtp_eval/adapters/__init__.py` — append:
```python
from vtp_eval.adapters import llava_sparsevlm  # noqa: F401
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED locally, PASS on Colab.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/adapters/llava_sparsevlm.py vtp_eval/adapters/__init__.py \
        tests/test_adapters_load.py
git commit -m "feat(adapters): SparseVLMs adapter"
```

---

## Task 9: VisionZip adapter

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/llava_visionzip.py`
- Modify: `vtp-eval/vtp_eval/adapters/__init__.py`
- Modify: `vtp-eval/tests/test_adapters_load.py`

- [ ] **Step 1: Add to ADAPTERS list**

Modify `vtp-eval/tests/test_adapters_load.py` — find:
```python
    "llava_sparsevlm",
    # Added in later tasks:
    # "llava_visionzip", "llava_divprune", "llava_sparsevila",
```
Replace with:
```python
    "llava_sparsevlm",
    "llava_visionzip",
    # Added in later tasks:
    # "llava_divprune", "llava_sparsevila",
```

- [ ] **Step 2: Implement adapter**

Create `vtp-eval/vtp_eval/adapters/llava_visionzip.py`:
```python
"""VisionZip adapter — pre-LLM pruning via CLIP CLS-attention dominant +
contextual stride sampling.

Reference: VisionZip (https://github.com/dvlab-research/VisionZip).
Pruning is applied by calling visionzip(model, dominant, contextual) after
load — replaces several module forwards at runtime.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_visionzip")
class LlavaVisionZip(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        dominant: int = 54,
        contextual: int = 10,
        **kw,
    ):
        super().__init__(pretrained=pretrained, **kw)
        from visionzip import visionzip  # import deferred — only on Colab
        self._model = visionzip(
            self._model,
            dominant=int(dominant),
            contextual=int(contextual),
        )
        self.pruning_meta = {
            "method": "visionzip",
            "dominant": int(dominant),
            "contextual": int(contextual),
        }
```

- [ ] **Step 3: Wire into package**

Modify `vtp-eval/vtp_eval/adapters/__init__.py` — append:
```python
from vtp_eval.adapters import llava_visionzip  # noqa: F401
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED locally.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/adapters/llava_visionzip.py vtp_eval/adapters/__init__.py \
        tests/test_adapters_load.py
git commit -m "feat(adapters): VisionZip adapter"
```

---

## Task 10: DivPrune adapter

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/llava_divprune.py`
- Modify: `vtp-eval/vtp_eval/adapters/__init__.py`
- Modify: `vtp-eval/tests/test_adapters_load.py`

- [ ] **Step 1: Add to ADAPTERS list**

Modify `vtp-eval/tests/test_adapters_load.py` — find:
```python
    "llava_visionzip",
    # Added in later tasks:
    # "llava_divprune", "llava_sparsevila",
```
Replace with:
```python
    "llava_visionzip",
    "llava_divprune",
    # Added in later task:
    # "llava_sparsevila",
```

- [ ] **Step 2: Implement adapter**

Create `vtp-eval/vtp_eval/adapters/llava_divprune.py`:
```python
"""DivPrune adapter — diversity-based pruning at projector (pre-LLM).

Reference: DivPrune (https://github.com/vbdi/divprune).
Pruning is controlled by env vars BASELINE / LAYER_INDEX / SUBSET_RATIO read in
divprune/LLaVA/llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal.
"""
import os

from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_divprune")
class LlavaDivPrune(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        subset_ratio: float = 0.098,
        layer_index: int = 0,
        **kw,
    ):
        os.environ["BASELINE"] = "OURS"
        os.environ["LAYER_INDEX"] = str(int(layer_index))
        os.environ["SUBSET_RATIO"] = str(float(subset_ratio))
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {
            "method": "divprune",
            "ratio": float(subset_ratio),
            "layer_index": int(layer_index),
        }
```

- [ ] **Step 3: Wire into package**

Modify `vtp-eval/vtp_eval/adapters/__init__.py` — append:
```python
from vtp_eval.adapters import llava_divprune   # noqa: F401
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED locally.

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/adapters/llava_divprune.py vtp_eval/adapters/__init__.py \
        tests/test_adapters_load.py
git commit -m "feat(adapters): DivPrune adapter"
```

---

## Task 11: SparseVILA adapter (custom generate)

**Files:**
- Create: `vtp-eval/vtp_eval/adapters/llava_sparsevila.py`
- Modify: `vtp-eval/vtp_eval/adapters/__init__.py`
- Modify: `vtp-eval/tests/test_adapters_load.py`

This is the most complex adapter — bypasses `Llava.__init__` because SparseVILA uses its own `load_sparse_llava()` and a custom `sparse_generate_packed()` instead of `model.generate()`.

- [ ] **Step 1: Add to ADAPTERS list (final adapter)**

Modify `vtp-eval/tests/test_adapters_load.py` — find:
```python
    "llava_divprune",
    # Added in later task:
    # "llava_sparsevila",
]
```
Replace with:
```python
    "llava_divprune",
    "llava_sparsevila",
]
```

- [ ] **Step 2: Implement adapter**

Create `vtp-eval/vtp_eval/adapters/llava_sparsevila.py`:
```python
"""SparseVILA adapter — encoder-side query-agnostic prune + decode-time
query-aware retrieval.

Reference: SparseVILA (https://github.com/AnKun10/sparsevila-implementation).
Cannot inherit Llava.__init__ because SparseVILA loads the model via its own
load_sparse_llava() and generates via sparse_generate_packed(), not
model.generate(). We bypass Llava.__init__ and reimplement the minimal lmms
contract: __init__ + generate_until.
"""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava import Llava  # noqa: F401 (sibling reference)
from lmms_eval.api.model import lmms

from vtp_eval.utils.timing import TimingHook


@register_model("llava_sparsevila")
class LlavaSparseVILA(lmms):
    """Skip Llava.__init__ — load via sparsevila API directly."""

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        encoder_prune_ratio: float = 0.5,
        decode_retrieval_ratio: float = 0.5,
        use_flash_kernel: bool = True,
        quantize_llm: str = "none",
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 1,
        log_timing: bool = True,
        timing_sidecar: Optional[str] = None,
        **kw,
    ):
        super().__init__()  # lmms.__init__ (bypasses Llava)

        from sparsevila import load_sparse_llava, SparseVILAConfig
        from transformers import AutoTokenizer

        torch_dtype = {"float16": torch.float16,
                       "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]

        cfg = SparseVILAConfig(
            encoder_prune_ratio=float(encoder_prune_ratio),
            decode_retrieval_ratio=float(decode_retrieval_ratio),
            use_flash_kernel=bool(use_flash_kernel),
            quantize_llm=str(quantize_llm),
        )
        self._model, self._image_processor = load_sparse_llava(
            pretrained, config=cfg, dtype=torch_dtype, device=device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
        self._sparse_config = cfg
        self._device = device
        self._pretrained = pretrained

        # lmms-eval contract attributes
        self.batch_size_per_gpu = int(batch_size)

        # Timing
        self.timing_hook: Optional[TimingHook] = (
            TimingHook() if log_timing else None
        )
        self.timing_sidecar_path = (
            Path(timing_sidecar) if timing_sidecar else None
        )

        self.pruning_meta = {
            "method": "sparsevila",
            "enc_ratio": float(encoder_prune_ratio),
            "dec_ratio": float(decode_retrieval_ratio),
        }

    # --- lmms required interface ----------------------------------------

    def loglikelihood(self, requests):
        raise NotImplementedError("SparseVILA only supports generate_until")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round not supported in this phase")

    def generate_until(self, requests):
        """Run sparse_generate_packed for each request."""
        from sparsevila import sparse_generate_packed
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token, process_images

        from contextlib import nullcontext
        results = []
        ctx_mgr = (self.timing_hook.measure()
                   if self.timing_hook is not None else nullcontext())
        with ctx_mgr:
            for r in requests:
                ctx, gen_kw, doc_to_visual, doc_id, task, split = r.args
                images = doc_to_visual(self.task_dict[task][split][doc_id])
                image_tensor = process_images(
                    images, self._image_processor, self._model.config,
                )[0].unsqueeze(0).to(self._device, dtype=torch.float16)

                # Build LLaVA-1.5 conv prompt
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + ctx)
                conv.append_message(conv.roles[1], None)
                input_ids = tokenizer_image_token(
                    conv.get_prompt(), self._tokenizer, IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(self._device)
                attn_mask = torch.ones_like(input_ids)

                out_ids = sparse_generate_packed(
                    self._model, self._tokenizer, input_ids,
                    image_tensor, attn_mask,
                    decode_retrieval_ratio=self._sparse_config.decode_retrieval_ratio,
                    max_new_tokens=int(gen_kw.get("max_new_tokens", 16)),
                )
                text = self._tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True,
                )[0]
                # Strip prompt prefix if echoed
                if conv.sep in text:
                    text = text.split(conv.sep)[-1].strip()
                results.append(text)

        if self.timing_hook is not None and self.timing_sidecar_path is not None:
            self.timing_hook.pruning_meta = self.pruning_meta
            self.timing_hook.dump(self.timing_sidecar_path)
        return results
```

- [ ] **Step 3: Wire into package**

Modify `vtp-eval/vtp_eval/adapters/__init__.py` — append:
```python
from vtp_eval.adapters import llava_sparsevila # noqa: F401
```
Full final file:
```python
"""lmms-eval adapter modules for visual token pruning methods.

Each submodule registers a model adapter via @register_model decorator.
Importing this package registers all adapters in lmms-eval's MODEL_REGISTRY.
"""
from vtp_eval.adapters import llava_baseline   # noqa: F401
from vtp_eval.adapters import llava_fastv      # noqa: F401
from vtp_eval.adapters import llava_sparsevlm  # noqa: F401
from vtp_eval.adapters import llava_visionzip  # noqa: F401
from vtp_eval.adapters import llava_divprune   # noqa: F401
from vtp_eval.adapters import llava_sparsevila # noqa: F401
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_adapters_load.py -v
```
Expected: SKIPPED locally (lmms-eval not installed).

- [ ] **Step 5: Commit**

```bash
git add vtp_eval/adapters/llava_sparsevila.py vtp_eval/adapters/__init__.py \
        tests/test_adapters_load.py
git commit -m "feat(adapters): SparseVILA adapter with custom generate"
```

---

## Task 12: `configs/methods.yaml`

**Files:**
- Create: `vtp-eval/configs/methods.yaml`

- [ ] **Step 1: Write the config**

Create `vtp-eval/configs/methods.yaml`:
```yaml
# Canonical pruning configs per method on POPE / LLaVA-1.5 7B.
# Values from each method's paper / reference repo default.
model_base: liuhaotian/llava-v1.5-7b

# Args appended to every --model_args
common_args:
  attn_implementation: eager   # Required for FastV/SparseVLMs (need attention map)
  dtype: float16

runs:
  - name: baseline
    model: llava_baseline
    model_args: {}

  - name: fastv_K2_R128
    model: llava_fastv
    model_args:
      fast_v_agg_layer: 2
      fast_v_attention_rank: 128

  - name: sparsevlm_192
    model: llava_sparsevlm
    model_args:
      retain_token: 192

  - name: visionzip_64
    model: llava_visionzip
    model_args:
      dominant: 54
      contextual: 10

  - name: divprune_0.098
    model: llava_divprune
    model_args:
      subset_ratio: 0.098

  - name: sparsevila_0.5_0.5
    model: llava_sparsevila
    model_args:
      encoder_prune_ratio: 0.5
      decode_retrieval_ratio: 0.5
```

- [ ] **Step 2: Verify YAML loads**

```bash
python -c "
import yaml
cfg = yaml.safe_load(open('configs/methods.yaml'))
assert len(cfg['runs']) == 6
assert all('name' in r and 'model' in r and 'model_args' in r for r in cfg['runs'])
print('OK — 6 runs configured')
"
```
Expected: `OK — 6 runs configured`

- [ ] **Step 3: Commit**

```bash
git add configs/methods.yaml
git commit -m "feat(configs): canonical method ratios for POPE eval"
```

---

## Task 13: Install scripts (all 7 shell scripts + patch helper)

**Files:**
- Create: `vtp-eval/install/_common.sh`
- Create: `vtp-eval/install/_patch_fastv.py`
- Create: `vtp-eval/install/baseline.sh`
- Create: `vtp-eval/install/fastv.sh`
- Create: `vtp-eval/install/sparsevlm.sh`
- Create: `vtp-eval/install/visionzip.sh`
- Create: `vtp-eval/install/divprune.sh`
- Create: `vtp-eval/install/sparsevila.sh`

- [ ] **Step 1: `_common.sh`**

Create `vtp-eval/install/_common.sh`:
```bash
#!/usr/bin/env bash
# Shared setup run by every per-method install script.
# Idempotent: skips git clone if dir exists; pip install -q is safe to re-run.
set -euo pipefail

cd /content

if [ ! -d lmms-eval ]; then
  git clone --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
fi

pip install -q accelerate==0.34.2 datasets==2.21.0 \
               sentencepiece protobuf==3.20.3 \
               pyyaml pandas matplotlib
```

- [ ] **Step 2: `_patch_fastv.py`**

Create `vtp-eval/install/_patch_fastv.py`:
```python
"""Copy FastV's patched modeling_llama.py into the active transformers install.

Idempotent: writes a .py.orig backup once, then overwrites every call.
"""
import shutil
import sys
from pathlib import Path

import transformers


def main() -> int:
    tf_dir = Path(transformers.__file__).parent
    target = tf_dir / "models" / "llama" / "modeling_llama.py"
    fastv_src = Path("/content/FastV/src/FastV/inference/transformers_replace"
                     "/models/llama/modeling_llama.py")
    if not fastv_src.exists():
        print(f"ERROR: FastV source not found at {fastv_src}", file=sys.stderr)
        return 1
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(target.read_text())
        print(f"Backup written to {backup}")
    shutil.copy(fastv_src, target)
    print(f"FastV patch applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: `baseline.sh`**

Create `vtp-eval/install/baseline.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/LLaVA ]; then
  git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
fi
pip install -e /content/LLaVA
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
```

- [ ] **Step 4: `fastv.sh`**

Create `vtp-eval/install/fastv.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/FastV ]; then
  git clone --depth 1 https://github.com/pkunlp-icler/FastV.git /content/FastV
fi
python install/_patch_fastv.py
pip install -e /content/FastV/src/LLaVA
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
```

- [ ] **Step 5: `sparsevlm.sh`**

Create `vtp-eval/install/sparsevlm.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/SparseVLMs ]; then
  git clone --depth 1 https://github.com/Gumpest/SparseVLMs.git /content/SparseVLMs
fi
pip install -e /content/SparseVLMs   # installs SparseVLMs' LLaVA fork
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
```

- [ ] **Step 6: `visionzip.sh`**

Create `vtp-eval/install/visionzip.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.40.0 tokenizers==0.19.1
if [ ! -d /content/VisionZip ]; then
  git clone --depth 1 https://github.com/dvlab-research/VisionZip.git /content/VisionZip
fi
pip install -e /content/VisionZip
# VisionZip's LLaVA fork (if shipped) or standalone llava
if [ -d /content/VisionZip/LLaVA ]; then
  pip install -e /content/VisionZip/LLaVA
elif [ ! -d /content/LLaVA ]; then
  git clone --depth 1 https://github.com/haotian-liu/LLaVA.git /content/LLaVA
  pip install -e /content/LLaVA
fi
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
```

- [ ] **Step 7: `divprune.sh`**

Create `vtp-eval/install/divprune.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/divprune ]; then
  git clone --depth 1 https://github.com/vbdi/divprune.git /content/divprune
fi
pip install -e /content/divprune/LLaVA
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
```

- [ ] **Step 8: `sparsevila.sh`**

Create `vtp-eval/install/sparsevila.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
bash install/_common.sh
pip install -q transformers==4.37.2 tokenizers==0.15.1
if [ ! -d /content/SparseVILA ]; then
  git clone --depth 1 https://github.com/AnKun10/sparsevila-implementation.git /content/SparseVILA
fi
cd /content/SparseVILA
git submodule update --init --recursive
pip install -e ./flash-colreduce
pip install -e .
cd /content
pip install -e /content/lmms-eval
pip install -e /content/vtp-eval
```

- [ ] **Step 9: Make scripts executable (on POSIX only — Colab is Linux)**

```bash
chmod +x install/*.sh
```
On Windows this is a no-op; bash on Colab will read the shebang regardless.

- [ ] **Step 10: Verify YAML-style sanity (no actual install — just shellcheck-style)**

```bash
for f in install/*.sh; do
  bash -n "$f" && echo "$f syntax OK"
done
```
Expected: 7 lines `install/<name>.sh syntax OK`.

- [ ] **Step 11: Commit**

```bash
git add install/
git commit -m "feat(install): per-method dependency setup scripts"
```

---

## Task 14: `scripts/run_one.sh` — single-run driver

**Files:**
- Create: `vtp-eval/scripts/run_one.sh`

- [ ] **Step 1: Implement**

Create `vtp-eval/scripts/run_one.sh`:
```bash
#!/usr/bin/env bash
# Run one method on POPE.
#
# Usage:
#   bash scripts/run_one.sh <run_name> [config_path] [task] [limit]
#
# Example:
#   bash scripts/run_one.sh baseline configs/methods.yaml pope 100
set -euo pipefail

RUN_NAME=${1:?usage: $0 <run_name> [config] [task] [limit]}
CONFIG=${2:-configs/methods.yaml}
TASK=${3:-pope}
LIMIT=${4:-}

OUT_DIR="results/$RUN_NAME"
mkdir -p "$OUT_DIR"

# Skip if already complete
if [ -f "$OUT_DIR/results.json" ]; then
  echo "[skip] $OUT_DIR/results.json exists. Delete to re-run."
  exit 0
fi

# Extract model + model_args from YAML
read -r MODEL MODEL_ARGS_RAW < <(python - <<PYEOF
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
run = next(r for r in cfg["runs"] if r["name"] == "$RUN_NAME")
args = run["model_args"]
common = cfg.get("common_args", {})
all_args = {**common, **args, "pretrained": cfg["model_base"]}
print(run["model"], ",".join(f"{k}={v}" for k, v in all_args.items()))
PYEOF
)

MODEL_ARGS="$MODEL_ARGS_RAW,timing_sidecar=$OUT_DIR/timing_raw.json"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "[run] $RUN_NAME — $MODEL($MODEL_ARGS) on $TASK"
python -m lmms_eval \
  --include_path "$PWD/vtp_eval/adapters" \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size "${VTP_BATCH_SIZE:-2}" \
  --log_samples --log_samples_suffix "$RUN_NAME" \
  --output_path "$OUT_DIR" \
  $LIMIT_ARG \
  2>&1 | tee "$OUT_DIR/run.log"

# Build timing.json from sidecar (pruning_meta is bundled inside)
python -m vtp_eval.utils.timing parse-sidecar \
  --sidecar "$OUT_DIR/timing_raw.json" \
  --output "$OUT_DIR/timing.json"

echo "[done] $RUN_NAME — see $OUT_DIR"
```

- [ ] **Step 2: Syntax check**

```bash
bash -n scripts/run_one.sh && echo "syntax OK"
```
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/run_one.sh
git add scripts/run_one.sh
git commit -m "feat(scripts): per-run driver script"
```

---

## Task 15: `scripts/run_all.sh` — local A100 sequential driver

**Files:**
- Create: `vtp-eval/scripts/run_all.sh`

- [ ] **Step 1: Implement**

Create `vtp-eval/scripts/run_all.sh`:
```bash
#!/usr/bin/env bash
# Run all 6 methods sequentially WITHOUT process restart.
# WARNING: Only works on a machine where every method's transformers/llava
# packages are isolated in separate conda envs the caller switches between,
# or where all methods happen to share compatible deps (rare).
# On Colab use notebooks/pope_eval.ipynb instead.
set -euo pipefail

CONFIG=${1:-configs/methods.yaml}
TASK=${2:-pope}
LIMIT=${3:-}

RUNS=$(python - <<PYEOF
import yaml
print("\n".join(r["name"] for r in yaml.safe_load(open("$CONFIG"))["runs"]))
PYEOF
)

while IFS= read -r RUN; do
  echo "==================== $RUN ===================="
  bash scripts/run_one.sh "$RUN" "$CONFIG" "$TASK" "$LIMIT"
done <<< "$RUNS"

python -m vtp_eval.utils.result_io aggregate results/ \
  --output results/summary.csv
echo "[all done] results/summary.csv"
```

- [ ] **Step 2: Syntax check**

```bash
bash -n scripts/run_all.sh && echo "syntax OK"
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/run_all.sh
git add scripts/run_all.sh
git commit -m "feat(scripts): sequential all-method driver (non-Colab)"
```

---

## Task 16: Smoke pipeline test (CPU-skip, GPU-only)

**Files:**
- Create: `vtp-eval/tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the test**

Create `vtp-eval/tests/test_pipeline_smoke.py`:
```python
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
```

- [ ] **Step 2: Run locally — expect skip**

```bash
pytest tests/test_pipeline_smoke.py -v
```
Expected: `SKIPPED [1] tests/test_pipeline_smoke.py:... requires CUDA` (locally).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_smoke.py
git commit -m "test: end-to-end baseline smoke on GPU"
```

---

## Task 17: Colab notebook `pope_eval.ipynb`

**Files:**
- Create: `vtp-eval/notebooks/pope_eval.ipynb`

The notebook is a JSON file. Build it via a Python script for reproducibility, then save the output as the .ipynb.

- [ ] **Step 1: Write a notebook generator script (throwaway)**

Create `vtp-eval/notebooks/_build_pope_eval.py`:
```python
"""One-shot script that emits notebooks/pope_eval.ipynb.

Kept in the repo so the notebook is reproducible from a text source.
"""
import json
from pathlib import Path

REPO = "https://github.com/<GITHUB_USER>/vtp-eval.git"  # TODO: replace with actual repo URL

CELLS = []

def md(text): CELLS.append({"cell_type": "markdown", "metadata": {},
                             "source": text.splitlines(keepends=True)})
def code(text): CELLS.append({"cell_type": "code", "metadata": {},
                               "execution_count": None, "outputs": [],
                               "source": text.splitlines(keepends=True)})

md(f"""\
# vtp-eval — POPE benchmark on LLaVA-1.5 7B
Compares baseline + 5 visual-token-pruning methods (FastV, SparseVLMs, VisionZip, DivPrune, SparseVILA).
**Runtime:** Colab Pro, GPU L4 (24 GB). **Total time:** ~3–4 hours for full POPE.
**IMPORTANT:** You MUST manually restart the runtime between methods (marked in red below).
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

# 6 run blocks
RUNS = [
    ("baseline",           "install/baseline.sh",  "baseline"),
    ("fastv_K2_R128",      "install/fastv.sh",     "FastV (K=2, R=128)"),
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
```

- [ ] **Step 2: Generate the notebook**

```bash
cd notebooks && python _build_pope_eval.py && cd ..
```
Expected: `Wrote .../pope_eval.ipynb with ~20 cells`

- [ ] **Step 3: Verify notebook is valid JSON**

```bash
python -c "
import json
nb = json.load(open('notebooks/pope_eval.ipynb'))
print(f'cells={len(nb[\"cells\"])}  format={nb[\"nbformat\"]}.{nb[\"nbformat_minor\"]}')
assert nb['nbformat'] == 4
print('OK')
"
```
Expected: `cells=20  format=4.5` then `OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/_build_pope_eval.py notebooks/pope_eval.ipynb
git commit -m "feat(notebook): Colab L4 pope_eval orchestrator"
```

---

## Task 18: Update README with usage + final commit

**Files:**
- Modify: `vtp-eval/README.md`

- [ ] **Step 1: Rewrite README with full usage**

Replace `vtp-eval/README.md` content with:
```markdown
# vtp-eval

Visual Token Pruning evaluation harness for LLaVA-1.5 7B on POPE, via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

Compares **6 configs**:
| Run | Method | Canonical config |
|---|---|---|
| `baseline` | none | full 576 visual tokens |
| `fastv_K2_R128` | FastV | prune after layer 2, keep 128 |
| `sparsevlm_192` | SparseVLMs | retain=192 multi-layer schedule |
| `visionzip_64` | VisionZip | dominant=54 + contextual=10 |
| `divprune_0.098` | DivPrune | subset ratio 9.8% |
| `sparsevila_0.5_0.5` | SparseVILA | encoder prune 0.5, decode retrieve 0.5 |

## Setup

This package is meant to run on **Colab Pro (L4 GPU, 24 GB)**. Each pruning method patches `transformers`/`llava` in incompatible ways, so the notebook restarts the kernel between methods.

### Run on Colab
1. Open `notebooks/pope_eval.ipynb` in Colab Pro.
2. Set runtime → L4 GPU.
3. Run cells sequentially. Restart runtime when prompted (markdown cells flag the spot).

### Run locally (single method, A100 / RTX 4090)
```bash
pip install -e ".[test]"
bash install/baseline.sh           # only one method per env
bash scripts/run_one.sh baseline
```
For all 6 methods locally you need separate conda envs (see `install/*.sh`).

## Output

After all runs, `notebooks/pope_eval.ipynb` aggregates into `results/summary.csv` with columns:
- `pope_acc`, `pope_f1`, `pope_yes_ratio`
- `latency_ms_mean`, `latency_ms_p95`, `peak_mem_mb`
- `keep_ratio_pct`, `tflops_prefill`, `tflops_reduction_pct`

Plus `results/scatter.png` — accuracy-vs-compute and accuracy-vs-latency scatter plots.

## Layout
```
vtp_eval/
  adapters/   # one lmms-eval adapter per method
  utils/      # tflops.py, timing.py, result_io.py
install/      # bash script per method (deps + clone + pip install -e)
configs/      # methods.yaml (canonical per-method ratios)
scripts/      # run_one.sh, run_all.sh
notebooks/    # pope_eval.ipynb (+ generator)
tests/        # pytest: TFLOPs math, result aggregation, smoke
docs/         # design spec + this plan
```

## Adding a new method
1. Add adapter at `vtp_eval/adapters/llava_<name>.py` inheriting `LlavaPruningBase`.
2. Import it in `vtp_eval/adapters/__init__.py`.
3. Add entry to `tests/test_adapters_load.py` ADAPTERS list.
4. Add a `shape_<name>()` to `vtp_eval/utils/tflops.py` + register in `_SHAPE_FNS`.
5. Add a branch in `compute_keep_ratio()` and `_tflops_kwargs_from_meta()` in `vtp_eval/utils/result_io.py`.
6. Add an `install/<name>.sh`.
7. Add an entry to `configs/methods.yaml`.
8. Add 1 run block in `notebooks/_build_pope_eval.py` and regenerate the notebook.

## Adding a new benchmark
Pass `--tasks <name>` to lmms-eval (already supported by `run_one.sh`'s third arg). No code changes required — POPE was chosen as the first benchmark; TextVQA / GQA / MME / MMBench all work the same way.

## License & references
- FastV: https://github.com/pkunlp-icler/FastV
- SparseVLMs: https://github.com/Gumpest/SparseVLMs
- VisionZip: https://github.com/dvlab-research/VisionZip
- DivPrune: https://github.com/vbdi/divprune
- SparseVILA: https://github.com/AnKun10/sparsevila-implementation
- lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
```

- [ ] **Step 2: Run full local test suite**

```bash
pytest tests/ -v
```
Expected: `tests/test_tflops.py` (13 PASS), `tests/test_result_io.py` (10 PASS), `tests/test_timing.py` (3 PASS), `tests/test_adapters_load.py` (SKIPPED — needs lmms-eval), `tests/test_pipeline_smoke.py` (SKIPPED — needs CUDA).

Total: 26 PASS + 13 SKIPPED. No failures.

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: complete README with usage + extension guide"
```

- [ ] **Step 4: Tag the repo for handoff**

```bash
git tag v0.1.0 -m "vtp-eval v0.1.0: POPE eval for 6 LLaVA-1.5 7B configs"
git log --oneline
```
Expected: ~18 commits, one per task, tag `v0.1.0` on HEAD.

---

## Post-Plan Manual Steps (user, not engineer)

1. Push `vtp-eval/` to a GitHub repo (public or private with PAT).
2. Replace `<GITHUB_USER>` placeholder in `notebooks/_build_pope_eval.py` with actual username, regenerate notebook (`python notebooks/_build_pope_eval.py`), commit, push.
3. Open the notebook on Colab Pro (L4), run cells in order, manually restart runtime when prompted.
4. After all runs complete, inspect `results/summary.csv` and `results/scatter.png`.

Verification checklist (from spec §1.4):
- [ ] All 6 runs produced `results.json` + `timing.json`
- [ ] `summary.csv` has 6 rows × 10 columns, no NaN
- [ ] Baseline `pope_f1` ∈ [0.85, 0.89]
- [ ] All methods: `tflops_prefill < baseline.tflops_prefill`
- [ ] All methods: `peak_mem_mb < baseline.peak_mem_mb`
- [ ] All methods: `pope_yes_ratio` ∈ [0.45, 0.55]
- [ ] Scatter plot generated to `results/scatter.png`
