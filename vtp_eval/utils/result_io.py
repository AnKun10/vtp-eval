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

_OUTPUT_COLUMNS = [
    "method", "pope_acc", "pope_f1", "pope_precision", "pope_recall", "pope_yes_ratio",
    "latency_ms_mean", "latency_ms_p95", "peak_mem_mb",
    "keep_ratio_pct", "tflops_llm", "tflops_encoder", "tflops_prefill", "tflops_reduction_pct",
]


# lmms-eval emits metric keys with a ",<filter>" suffix; for unfiltered tasks
# (POPE here) the filter is the literal string "none". Earlier vtp-eval expected
# bare keys (`pope_accuracy`) from an older lmms-eval; v0.5+ writes
# `pope_accuracy,none`.
def _metric(pope_dict: Dict, base_key: str):
    if base_key in pope_dict:
        return pope_dict[base_key]
    suffixed = f"{base_key},none"
    if suffixed in pope_dict:
        return pope_dict[suffixed]
    raise KeyError(
        f"metric {base_key!r} not found in pope results "
        f"(keys: {sorted(pope_dict.keys())})"
    )


def compute_keep_ratio(meta: Dict) -> float:
    """Percent of visual tokens retained, averaged across layers if dynamic."""
    method = meta["method"]
    if method == "baseline":
        return 100.0
    if method == "fastv":
        R = int(meta["R"])
        return 100.0 * R / VISUAL_FULL
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


def _find_results_json(run_dir: Path) -> Optional[Path]:
    """Return the lmms-eval results.json for this run, or None.

    lmms-eval v0.5+ writes the file under `<run_dir>/<model_name>/<timestamp>_results.json`
    (e.g. `results/baseline/liuhaotian__llava-v1.5-7b/20260517_142135_results.json`).
    For backward compat we also accept the older `<run_dir>/results.json`.
    """
    direct = run_dir / "results.json"
    if direct.exists():
        return direct
    candidates = list(run_dir.glob("*/*_results.json"))
    if not candidates:
        return None
    # Most recent if multiple
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _row_from_run(run_dir: Path) -> Optional[Dict]:
    res_p = _find_results_json(run_dir)
    tim_p = run_dir / "timing.json"
    if res_p is None or not tim_p.exists():
        return None
    results = json.loads(res_p.read_text())
    timing = json.loads(tim_p.read_text())
    pope = results["results"]["pope"]
    meta = timing["pruning_meta"]
    tf = compute_method_tflops(meta["method"], **_tflops_kwargs_from_meta(meta))
    return {
        "method": run_dir.name,
        "pope_acc": _metric(pope, "pope_accuracy"),
        "pope_f1": _metric(pope, "pope_f1_score"),
        "pope_precision": _metric(pope, "pope_precision"),
        "pope_recall": _metric(pope, "pope_recall"),
        "pope_yes_ratio": _metric(pope, "pope_yes_ratio"),
        "latency_ms_mean": timing["latency_per_sample_ms"]["mean"],
        "latency_ms_p95": timing["latency_per_sample_ms"]["p95"],
        "peak_mem_mb": timing["peak_gpu_mem_mb"],
        "keep_ratio_pct": compute_keep_ratio(meta),
        "tflops_llm": tf["tflops_llm_prefill"],
        "tflops_encoder": tf["tflops_encoder"],
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
    if not rows:
        df = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return df
    df = pd.DataFrame(rows)
    baseline = df[df["_method_key"] == "baseline"]
    if not baseline.empty:
        base_tf = float(baseline["tflops_prefill"].iloc[0])
        df = df.copy()
        df["tflops_reduction_pct"] = (1 - df["tflops_prefill"] / base_tf) * 100
    else:
        df = df.copy()
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
