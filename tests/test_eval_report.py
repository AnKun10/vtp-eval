# tests/test_eval_report.py
import json

from vtp_eval.eval import report


def test_pick_primary_metric_prefers_f1_then_acc():
    res = {"pope_accuracy,none": 0.85, "pope_f1_score,none": 0.88,
           "pope_accuracy_stderr,none": 0.01}
    name, val = report.pick_primary_metric(res)
    assert name == "pope_f1_score" and val == 0.88


def test_pick_primary_metric_skips_stderr():
    res = {"gqa_accuracy,none": 0.62, "gqa_accuracy_stderr,none": 0.01}
    name, val = report.pick_primary_metric(res)
    assert name == "gqa_accuracy" and val == 0.62


def test_estimate_tflops_monotonic_in_tokens():
    assert report.estimate_tflops(64) < report.estimate_tflops(576)
    assert report.estimate_tflops(576) > 0


def test_parse_sidecar_writes_timing_json(tmp_path):
    sidecar = tmp_path / "raw.json"
    sidecar.write_text(json.dumps({
        "records": [
            {"encoder_ms": 99, "prefill_ms": 99, "decode_ms": 99, "decode_steps": 9,
             "total_ms": 999, "batch_size": 1, "peak_mem_mb": 100},   # warm-up
            {"encoder_ms": 10, "prefill_ms": 20, "decode_ms": 30, "decode_steps": 3,
             "total_ms": 60, "batch_size": 1, "peak_mem_mb": 120},
        ],
        "pruning_meta": {"method": "proposed", "avg_tokens": 64},
    }))
    out = tmp_path / "timing.json"
    report.parse_sidecar(sidecar, out)
    d = json.loads(out.read_text())
    assert d["prefill_ms"] == 20.0          # warm-up dropped
    assert d["pruning_meta"]["method"] == "proposed"
    assert d["peak_mem_mb"] == 120.0


def test_aggregate_one_run(tmp_path):
    run = tmp_path / "proposed_x"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(
        {"results": {"pope": {"pope_f1_score,none": 0.88, "pope_accuracy,none": 0.85}}}))
    (run / "timing.json").write_text(json.dumps(
        {"encoder_ms": 5.0, "prefill_ms": 20.0, "decode_ms": 30.0,
         "total_latency_ms": 55.0, "decode_tokens": 3.0, "peak_mem_mb": 120.0,
         "pruning_meta": {"method": "proposed", "avg_tokens": 64}}))
    out_csv = tmp_path / "summary.csv"
    rows = report.aggregate(tmp_path, out_csv)
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "proposed" and r["task"] == "pope"
    assert r["metric"] == "pope_f1_score" and float(r["value"]) == 0.88
    assert float(r["keep_ratio_pct"]) == round(64 / 576 * 100, 2)
    assert float(r["prefill_ms"]) == 20.0
    assert out_csv.exists()
