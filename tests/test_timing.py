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
