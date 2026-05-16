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
    "method", "pope_acc", "pope_f1", "pope_yes_ratio",
    "latency_ms_mean", "latency_ms_p95", "peak_mem_mb",
    "keep_ratio_pct", "tflops_prefill", "tflops_reduction_pct",
]


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
