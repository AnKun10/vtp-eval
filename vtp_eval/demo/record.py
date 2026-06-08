# vtp_eval/demo/record.py
"""Capture the current turn's R1/R2 keep indices during the real forward.

Monkeypatches the public selection entry points where they are consumed:
- stage1_vision.select_stage1  -> R1 patch indices ([R1], sorted)
- stage2_llm.build_keep_index  -> R2 indices into the R1 block ([r2])
Index-only; never changes the returned tensors, so inference latency stays honest.
Mapping R2-local -> original patches is done by the engine (needs the per-image
R1 index set). Pure (torch only).
"""
from __future__ import annotations

import torch

from vtp_eval.proposed_method import stage1_vision, stage2_llm

_buf: dict = {"r1_idx": None, "r2_local": None}
_orig: dict = {"select_stage1": None, "build_keep_index": None}


def install_recorders() -> None:
    """Idempotent: install once; repeated calls keep the original references."""
    if _orig["select_stage1"] is None:
        _orig["select_stage1"] = stage1_vision.select_stage1
    if _orig["build_keep_index"] is None:
        _orig["build_keep_index"] = stage2_llm.build_keep_index

    orig_s1 = _orig["select_stage1"]
    orig_bk = _orig["build_keep_index"]

    def rec_s1(attn_penult, hidden_penult, dominant_k, diversity_m):
        keep = orig_s1(attn_penult, hidden_penult, dominant_k, diversity_m)
        _buf["r1_idx"] = keep[0].detach().cpu()
        return keep

    def rec_bk(scores, r2, vision_start, seq_len):
        _buf["r2_local"] = scores.topk(r2, dim=1).indices[0].detach().cpu()
        return orig_bk(scores, r2, vision_start, seq_len)

    stage1_vision.select_stage1 = rec_s1
    stage2_llm.build_keep_index = rec_bk


def uninstall_recorders() -> None:
    if _orig["select_stage1"] is not None:
        stage1_vision.select_stage1 = _orig["select_stage1"]
        _orig["select_stage1"] = None
    if _orig["build_keep_index"] is not None:
        stage2_llm.build_keep_index = _orig["build_keep_index"]
        _orig["build_keep_index"] = None


def reset() -> None:
    _buf["r1_idx"] = None
    _buf["r2_local"] = None


def last_r1_idx():
    return _buf["r1_idx"]


def last_r2_local():
    return _buf["r2_local"]
