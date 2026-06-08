# vtp_eval/demo/cache.py
"""Retain-token embedding cache for multi-turn QA on one image.

Stores the R1 retain tokens (output of the patched vision tower + projector)
keyed by image content hash, so turn 2+ on the same image skips the CLIP encoder
and Stage-1 selection entirely. Pure (torch only) so it unit-tests on CPU.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch


def image_hash(images: torch.Tensor) -> str:
    """Content hash of a (preprocessed) image tensor. Stable across dtype/device
    by canonicalizing to contiguous float32 CPU bytes."""
    t = images.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()
