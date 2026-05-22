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
    """Records per-call latency and peak memory.

    On GPU, .measure() uses torch.cuda.Event + max_memory_allocated().
    Without CUDA available, falls back to time.perf_counter() and skips memory.

    `batch_size` records how many samples a single .measure() call covered, so
    that downstream aggregation can compute per-sample latency = latency_ms /
    batch_size. The previous version of this hook wrapped the *entire*
    `generate_until()` call, which records the total wall-clock and reports it
    as "per-sample" — wrong by a factor of N_samples. The hook should be
    invoked once per `model.generate()` call (i.e., per batch).
    """

    def __init__(self) -> None:
        self._samples: List[Dict] = []
        self.pruning_meta: Dict = {}

    @contextmanager
    def measure(self, batch_size: int = 1):
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
                batch_size=batch_size,
            )
        else:
            t0 = time.perf_counter()
            yield
            self.record(
                latency_ms=(time.perf_counter() - t0) * 1000,
                peak_mem_mb=0,
                batch_size=batch_size,
            )

    def record(self, latency_ms: float, peak_mem_mb: float, batch_size: int = 1) -> None:
        self._samples.append({
            "latency_ms": float(latency_ms),
            "peak_mem_mb": float(peak_mem_mb),
            "batch_size": int(batch_size),
        })

    def summary(self) -> Dict:
        if not self._samples:
            return {
                "n_samples": 0,
                "latency_per_sample_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
                "peak_gpu_mem_mb": 0.0,
            }
        # Per-sample latency = call latency / batch_size; assumes the batch was
        # processed in parallel on-GPU so per-sample latency is the same.
        per_sample = sorted(
            s["latency_ms"] / max(1, s.get("batch_size", 1))
            for s in self._samples
        )
        n = len(per_sample)
        p50 = statistics.median(per_sample)
        p95 = per_sample[min(int(n * 0.95), n - 1)]
        peak = max(s["peak_mem_mb"] for s in self._samples)
        return {
            "n_samples": n,
            "latency_per_sample_ms": {
                "mean": statistics.fmean(per_sample),
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
        per_sample = sorted(
            s["latency_ms"] / max(1, s.get("batch_size", 1)) for s in samples
        )
        total_wall_s = sum(s["latency_ms"] for s in samples) / 1000
        n_calls = len(samples)
        n_samples = sum(s.get("batch_size", 1) for s in samples)
        out = {
            "method": meta.get("method", "unknown"),
            "n_calls": n_calls,
            "n_samples": n_samples,
            "total_wall_s": total_wall_s,
            "latency_per_sample_ms": {
                "mean": statistics.fmean(per_sample),
                "p50": statistics.median(per_sample),
                "p95": per_sample[min(int(n_calls * 0.95), n_calls - 1)],
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
