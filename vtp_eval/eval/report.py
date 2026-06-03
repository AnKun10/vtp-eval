# vtp_eval/eval/report.py
"""Turn per-run lmms-eval results.json + timing sidecar into summary.csv.

Subcommands:
  parse-sidecar --sidecar raw.json --output timing.json
  aggregate <results_dir> --output summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from vtp_eval.eval.stage_timer import summarize_stage_records

# --- theoretical prefill TFLOPs (approximate; LLaVA-1.5-7B) ---
_N_LAYERS, _D, _FFN, _FFN_MATS = 32, 4096, 11008, 3
_SYS_LEN, _TEXT = 35, 11          # POPE-ish prompt; constant offset, fine for relative


def _layer_flops(seq_len: float) -> float:
    attn_proj = 4 * seq_len * _D * _D
    attn_mm = 2 * seq_len * seq_len * _D
    ffn = _FFN_MATS * seq_len * _D * _FFN
    return float(attn_proj + attn_mm + ffn)


def estimate_tflops(avg_tokens: float) -> float:
    """Approximate prefill TFLOPs treating every layer at avg_tokens visual
    tokens (strictly monotonic in avg_tokens; for relative comparison only)."""
    seq = _SYS_LEN + float(avg_tokens) + _TEXT
    return _N_LAYERS * _layer_flops(seq) / 1e12


def pick_primary_metric(task_results: Dict) -> Tuple[str, float]:
    """Choose a task's headline metric from an lmms-eval results dict.

    Keys look like 'pope_f1_score,none'. Skip *_stderr; prefer f1, then
    accuracy/acc, else the first numeric metric. Returns (clean_name, value).
    """
    items = []
    for k, v in task_results.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        base = k.split(",")[0]
        if base.endswith("_stderr"):
            continue
        items.append((base, float(v)))
    if not items:
        raise ValueError(f"no numeric metric in {sorted(task_results)}")
    for pref in ("f1", "acc"):
        for base, val in items:
            if pref in base.lower():
                return base, val
    return items[0]


# Per-task headline metric. Values are the metric base name (before the ",none"
# suffix lmms-eval appends). "mme" is special-cased in select_metrics (two
# sub-scores -> perception + total). Tasks absent here fall back to the
# pick_primary_metric heuristic. Keys verified on-box (see plan Task 1).
TASK_PRIMARY = {
    "pope": "pope_f1_score",
    "gqa": "exact_match",
    "textvqa_val": "exact_match",
    "scienceqa_img": "exact_match",
    "vizwiz_vqa_val": "exact_match",
    "ocrbench": "ocrbench_accuracy",
    "mme": ("mme_perception_score", "mme_cognition_score"),  # special-cased
    # mmbench_en_dev intentionally absent: its metric is gpt_eval_score (needs an
    # OpenAI API key), so it is out of the default set for now.
}


def _metric_value(task_res: Dict, base: str) -> float:
    """Value of the metric whose key (before ',') == base; skip bool / non-numeric."""
    for k, v in task_res.items():
        if k.split(",")[0] == base and not isinstance(v, bool) \
                and isinstance(v, (int, float)):
            return float(v)
    raise KeyError(base)


def select_metrics(task: str, task_res: Dict) -> List[Tuple[str, float]]:
    """Headline metric(s) for a task as [(clean_name, value), ...].

    Uses TASK_PRIMARY; MME returns perception + total when both sub-scores are
    present (and a lone sub-score otherwise, so a partial/--limit run never drops
    the task). Unmapped tasks fall back to the pick_primary_metric heuristic.
    """
    spec = TASK_PRIMARY.get(task)
    if spec is None:
        return [pick_primary_metric(task_res)]
    if isinstance(spec, tuple):                       # MME: two sub-scores
        sub = {}
        for name, key in zip(("perception", "cognition"), spec):
            try:
                sub[name] = _metric_value(task_res, key)
            except KeyError:
                pass
        rows: List[Tuple[str, float]] = []
        if "perception" in sub:
            rows.append(("mme_perception", sub["perception"]))
        if {"perception", "cognition"} <= sub.keys():
            rows.append(("mme_total", sub["perception"] + sub["cognition"]))
        elif "cognition" in sub:                      # partial run: cognition only
            rows.append(("mme_cognition", sub["cognition"]))
        if not rows:
            raise KeyError(f"no MME sub-scores in {sorted(task_res)}")
        return rows
    return [(spec, _metric_value(task_res, spec))]


def parse_sidecar(sidecar: Path, output: Path) -> Dict:
    raw = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    summary = summarize_stage_records(raw.get("records", []), drop_warmup=True)
    summary["pruning_meta"] = raw.get("pruning_meta", {"method": "unknown"})
    Path(output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate(results_dir: Path, output_csv: Path) -> List[Dict]:
    results_dir = Path(results_dir)
    rows: List[Dict] = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            rj, tj = task_dir / "results.json", task_dir / "timing.json"
            if not rj.exists():
                continue
            results = json.loads(rj.read_text(encoding="utf-8")).get("results", {})
            timing = json.loads(tj.read_text(encoding="utf-8")) if tj.exists() else {}
            meta = timing.get("pruning_meta", {"method": run_dir.name})
            avg_tokens = float(meta.get("avg_tokens", 576))
            for task, task_res in results.items():
                try:
                    metrics = select_metrics(task, task_res)
                except (ValueError, KeyError):
                    continue
                for metric, value in metrics:
                    rows.append({
                        "method": run_dir.name,   # unique + self-identifying per run
                        "task": task,
                        "metric": metric,
                        "value": round(value, 4),
                        "avg_tokens": round(avg_tokens, 1),
                        "keep_ratio_pct": round(avg_tokens / 576 * 100, 2),
                        "encoder_ms": round(timing.get("encoder_ms", 0.0), 3),
                        "prefill_ms": round(timing.get("prefill_ms", 0.0), 3),
                        "decode_ms": round(timing.get("decode_ms", 0.0), 3),
                        "total_ms": round(timing.get("total_latency_ms", 0.0), 3),
                        "peak_mem_mb": round(timing.get("peak_mem_mb", 0.0), 1),
                        "tflops": round(estimate_tflops(avg_tokens), 3),
                    })
    cols = ["method", "task", "metric", "value", "avg_tokens", "keep_ratio_pct",
            "encoder_ms", "prefill_ms", "decode_ms", "total_ms", "peak_mem_mb", "tflops"]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


def _cli() -> None:
    p = argparse.ArgumentParser(prog="vtp_eval.eval.report")
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("parse-sidecar")
    ps.add_argument("--sidecar", type=Path, required=True)
    ps.add_argument("--output", type=Path, required=True)
    ag = sub.add_parser("aggregate")
    ag.add_argument("results_dir", type=Path)
    ag.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "parse-sidecar":
        parse_sidecar(args.sidecar, args.output)
    else:
        rows = aggregate(args.results_dir, args.output)
        print(f"[report] {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    _cli()
