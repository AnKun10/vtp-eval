# vtp_eval/demo/benchmarks.py
"""Benchmark image picker data layer for the demo.

Groups VQA benchmark rows by image so each unique image is shown once with ALL
its questions. Builds on prune_viz's DatasetSpec/resolve_specs + configs/
prune_viz.yaml (reused, not modified). `group_rows`/`save_groups` are pure
(CPU-testable); `fetch_grouped` is the HF-streaming glue (Vast.ai only).
"""
from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from pathlib import Path

from PIL import Image


def normalize_image(raw) -> Image.Image:
    """Coerce a dataset image field (PIL image, file path, or {'bytes': ...}
    dict) into an RGB PIL image — same handling prune_viz.fetch_samples uses."""
    if hasattr(raw, "convert"):                    # already a PIL image
        return raw.convert("RGB")
    if isinstance(raw, dict):                      # {'bytes': ...}
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    return Image.open(raw).convert("RGB")          # path-like


def image_hash(img: Image.Image) -> str:
    """Content hash of an RGB PIL image (dataset-agnostic group key)."""
    return hashlib.sha1(img.tobytes()).hexdigest()


def group_rows(rows, image_key, q_key, n_images, max_scan):
    """Group an iterable of dataset rows by image content-hash.

    Scans up to `max_scan` rows, accumulating every question per unique image,
    then returns the first `n_images` groups in first-seen order. Each group:
    {"image": PIL.Image, "questions": [str, ...]}. Empty questions are skipped.
    """
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for i, row in enumerate(rows):
        if i >= max_scan:
            break
        img = normalize_image(row[image_key])
        h = image_hash(img)
        if h not in groups:
            groups[h] = {"image": img, "questions": []}
        q = str(row.get(q_key, "") or "").strip()
        if q:
            groups[h]["questions"].append(q)
    out = []
    for g in groups.values():
        out.append(g)
        if len(out) >= n_images:
            break
    return out
