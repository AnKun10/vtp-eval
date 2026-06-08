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


class RetainTokenCache:
    """LRU cache of R1 retain tokens keyed by image hash.

    Each entry: {"feats": tensor [1, R1, D], "r1_idx": tensor [R1] | None}.
    `enabled` gates lookups (toggled off for the no-cache comparison run);
    `last_was_hit` records the result of the most recent lookup for metrics.
    """

    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self.enabled = True
        self.last_was_hit = False

    def get(self, key: str):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, feats: torch.Tensor) -> None:
        self._store[key] = {"feats": feats, "r1_idx": None}
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def attach_r1(self, key: str, r1_idx: torch.Tensor) -> None:
        if key in self._store:
            self._store[key]["r1_idx"] = r1_idx

    def clear(self) -> None:
        self._store.clear()


def install_cache(model, cache: RetainTokenCache) -> RetainTokenCache:
    """Replace ``model.encode_images`` with a memoizing wrapper.

    The vision tower is already patched by ``proposed_prune`` (Stage 1), so
    encode_images returns the projected R1 retain tokens. On a hit we return the
    cached tensor and skip the CLIP encoder + R1 selection; on a miss we call the
    original and store. When ``cache.enabled`` is False we always recompute and
    store nothing (used for the no-cache comparison timing).
    """
    orig = model.encode_images   # bound method, captured once

    def wrapped(images):
        if not cache.enabled:
            cache.last_was_hit = False
            return orig(images)
        key = image_hash(images)
        hit = cache.get(key)
        if hit is not None:
            cache.last_was_hit = True
            return hit["feats"]
        cache.last_was_hit = False
        feats = orig(images)
        cache.put(key, feats)
        return feats

    model.encode_images = wrapped
    return cache
