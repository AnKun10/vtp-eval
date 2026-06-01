# vtp_eval/eval/stage_timer.py
"""3-stage latency timing (vision encoder / LLM prefill / LLM decode) for any
LLaVA model (baseline or proposed), plus pure aggregation helpers.

GPU: torch.cuda.Event pairs per region, synchronized once per batch.
No CUDA (or force_cpu): time.perf_counter fallback so the bookkeeping is
unit-testable. The aggregation (summarize_stage_records) is pure stdlib.
"""
from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

STAGES = ("encoder", "prefill", "decode")


def summarize_stage_records(records: List[Dict], drop_warmup: bool = True) -> Dict:
    """Per-batch stage records -> per-sample averages.

    record keys: encoder_ms, prefill_ms, decode_ms, decode_steps, total_ms,
    batch_size, peak_mem_mb. The first record is dropped as warm-up when there
    is more than one. Per-sample value = stage_ms / batch_size, averaged.
    """
    recs = list(records)
    if drop_warmup and len(recs) > 1:
        recs = recs[1:]
    if not recs:
        z = {f"{s}_ms": 0.0 for s in STAGES}
        return {"n_samples": 0, "total_latency_ms": 0.0, "decode_tokens": 0.0,
                "peak_mem_mb": 0.0, **z}

    def per_sample_mean(key: str) -> float:
        return statistics.fmean(r[key] / max(1, r["batch_size"]) for r in recs)

    return {
        "n_samples": sum(r["batch_size"] for r in recs),
        "encoder_ms": per_sample_mean("encoder_ms"),
        "prefill_ms": per_sample_mean("prefill_ms"),
        "decode_ms": per_sample_mean("decode_ms"),
        "total_latency_ms": per_sample_mean("total_ms"),
        "decode_tokens": statistics.fmean(
            r["decode_steps"] / max(1, r["batch_size"]) for r in recs),
        "peak_mem_mb": max(r["peak_mem_mb"] for r in recs),
    }


class StageTimer:
    def __init__(self, force_cpu: bool = False) -> None:
        self.records: List[Dict] = []
        self._cur = None
        self._installed = False
        if force_cpu:
            self._cuda = False
        else:
            try:
                import torch
                self._cuda = torch.cuda.is_available()
            except Exception:
                self._cuda = False

    @contextmanager
    def region(self, name: str):
        if self._cur is None:                      # not inside a batch
            yield
            return
        if self._cuda:
            import torch
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            try:
                yield
            finally:
                e.record()
                self._cur[name].append((s, e))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._cur[name].append((time.perf_counter() - t0) * 1000.0)

    def start_batch(self, batch_size: int) -> None:
        self._cur = {"encoder": [], "prefill": [], "decode": [],
                     "batch_size": int(batch_size)}
        if self._cuda:
            import torch
            torch.cuda.reset_peak_memory_stats()
            ts = torch.cuda.Event(enable_timing=True)
            ts.record()
            self._cur["_ts"] = ts
        else:
            self._cur["_t0"] = time.perf_counter()

    def end_batch(self) -> None:
        if self._cur is None:
            return
        cur = self._cur
        if self._cuda:
            import torch
            te = torch.cuda.Event(enable_timing=True)
            te.record()
            torch.cuda.synchronize()
            stage_ms = {s: sum(a.elapsed_time(b) for a, b in cur[s]) for s in STAGES}
            total_ms = cur["_ts"].elapsed_time(te)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            stage_ms = {s: float(sum(cur[s])) for s in STAGES}
            total_ms = (time.perf_counter() - cur["_t0"]) * 1000.0
            peak = 0.0
        self.records.append({
            "encoder_ms": stage_ms["encoder"],
            "prefill_ms": stage_ms["prefill"],
            "decode_ms": stage_ms["decode"],
            "decode_steps": len(cur["decode"]),
            "total_ms": total_ms,
            "batch_size": cur["batch_size"],
            "peak_mem_mb": peak,
        })
        self._cur = None

    def install(self, model) -> None:
        """Wrap the vision-tower forward (encoder) and LLM forward (prefill if
        seq_len>1 else decode). Call ONCE, lazily, after all model patching
        (e.g. proposed_prune) so the wrapped forwards are the final ones.
        """
        if self._installed:
            return
        self._installed = True
        timer = self

        vt = model.get_model().get_vision_tower()
        _vt_fwd = vt.forward

        def vt_forward(*a, **k):
            with timer.region("encoder"):
                return _vt_fwd(*a, **k)
        vt.forward = vt_forward

        lm = model.model
        _lm_fwd = lm.forward

        def lm_forward(*a, **k):
            ids = k.get("input_ids", a[0] if a else None)
            emb = k.get("inputs_embeds")
            if ids is not None and hasattr(ids, "shape"):
                seq = ids.shape[1]
            elif emb is not None and hasattr(emb, "shape"):
                seq = emb.shape[1]
            else:
                seq = 1
            with timer.region("prefill" if seq > 1 else "decode"):
                return _lm_fwd(*a, **k)
        lm.forward = lm_forward

    def dump(self, path, pruning_meta: Dict) -> None:
        Path(path).write_text(json.dumps(
            {"records": self.records, "pruning_meta": pruning_meta}, indent=2),
            encoding="utf-8")
