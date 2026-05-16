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
    r = compute_keep_ratio({"method": "fastv", "K": 2, "R": 128})
    assert 20.0 < r < 25.0  # 128/576 ≈ 22.2%


def test_compute_keep_ratio_visionzip():
    r = compute_keep_ratio({"method": "visionzip", "dominant": 54, "contextual": 10})
    assert 10.0 < r < 12.0


def test_compute_keep_ratio_divprune():
    r = compute_keep_ratio({"method": "divprune", "ratio": 0.098})
    assert r == pytest.approx(9.8)


def test_compute_keep_ratio_sparsevila():
    r = compute_keep_ratio({"method": "sparsevila",
                             "enc_ratio": 0.5, "dec_ratio": 0.5})
    assert r == pytest.approx(50.0)


def test_compute_keep_ratio_sparsevlm():
    r = compute_keep_ratio({"method": "sparsevlm", "retain": 192})
    # Average across schedule [576]*2 + [300]*4 + [200]*10 + [110]*16
    # = 6112 / 32 / 576 * 100 ≈ 33.2
    assert 30.0 < r < 36.0
