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


def test_select_metrics_mapped_task():
    res = {"exact_match,none": 0.62, "exact_match_stderr,none": 0.01}
    assert report.select_metrics("gqa", res) == [("exact_match", 0.62)]


def test_select_metrics_fallback_for_unmapped_task():
    # task absent from TASK_PRIMARY -> heuristic (prefers f1)
    res = {"foo_f1_score,none": 0.70, "foo_accuracy,none": 0.60}
    assert report.select_metrics("brand_new_task", res) == [("foo_f1_score", 0.70)]


def test_select_metrics_mme_emits_perception_cognition_total():
    res = {"mme_perception_score,none": 1500.0, "mme_cognition_score,none": 350.0}
    assert report.select_metrics("mme", res) == [
        ("mme_perception", 1500.0), ("mme_cognition", 350.0), ("mme_total", 1850.0)]


def test_select_metrics_mme_partial_run_keeps_lone_subscore():
    # A --limit run can produce only one MME sub-score; the task must not vanish.
    res = {"mme_cognition_score,none": 45.0}
    assert report.select_metrics("mme", res) == [("mme_cognition", 45.0)]


def test_metric_value_ignores_bool_and_stderr():
    res = {"accuracy,none": 0.71, "accuracy_stderr,none": 0.02, "submission,none": True}
    assert report._metric_value(res, "accuracy") == 0.71


def test_aggregate_one_run(tmp_path):
    cell = tmp_path / "proposed_x" / "pope"      # results/<run>/<task>/
    cell.mkdir(parents=True)
    (cell / "results.json").write_text(json.dumps(
        {"results": {"pope": {"pope_f1_score,none": 0.88, "pope_accuracy,none": 0.85}}}))
    (cell / "timing.json").write_text(json.dumps(
        {"encoder_ms": 5.0, "prefill_ms": 20.0, "decode_ms": 30.0,
         "total_latency_ms": 55.0, "decode_tokens": 3.0, "peak_mem_mb": 120.0,
         "pruning_meta": {"method": "proposed", "avg_tokens": 64}}))
    out_csv = tmp_path / "summary.csv"
    rows = report.aggregate(tmp_path, out_csv)
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "proposed_x" and r["task"] == "pope"
    assert r["metric"] == "pope_f1_score" and float(r["value"]) == 0.88
    assert float(r["keep_ratio_pct"]) == round(64 / 576 * 100, 2)
    assert float(r["prefill_ms"]) == 20.0
    assert float(r["total_ms"]) == 55.0
    assert out_csv.exists()


def test_aggregate_two_level_multitask_and_mme(tmp_path):
    run = tmp_path / "proposed"
    timing = {"encoder_ms": 5.0, "prefill_ms": 20.0, "decode_ms": 30.0,
              "total_latency_ms": 55.0, "decode_tokens": 1.0, "peak_mem_mb": 120.0,
              "pruning_meta": {"method": "proposed", "avg_tokens": 64}}
    pope = run / "pope"
    pope.mkdir(parents=True)
    (pope / "results.json").write_text(json.dumps(
        {"results": {"pope": {"pope_f1_score,none": 0.76}}}))
    (pope / "timing.json").write_text(json.dumps(timing))
    mme = run / "mme"
    mme.mkdir(parents=True)
    (mme / "results.json").write_text(json.dumps(
        {"results": {"mme": {"mme_perception_score,none": 1500.0,
                             "mme_cognition_score,none": 350.0}}}))
    (mme / "timing.json").write_text(json.dumps(timing))
    rows = report.aggregate(tmp_path, tmp_path / "summary.csv")
    pairs = {(r["task"], r["metric"]) for r in rows}
    assert ("pope", "pope_f1_score") in pairs
    assert ("mme", "mme_perception") in pairs
    assert ("mme", "mme_total") in pairs
    mt = next(r for r in rows if r["metric"] == "mme_total")
    assert float(mt["value"]) == 1850.0
    assert mt["method"] == "proposed" and float(mt["prefill_ms"]) == 20.0
    assert all(r["method"] == "proposed" for r in rows)   # method == run-dir name
