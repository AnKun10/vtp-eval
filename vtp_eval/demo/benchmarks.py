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


def save_groups(groups, dataset: str, out_dir) -> list:
    """Write one image per group to `out_dir`; return flat records
    [{"dataset", "image_path", "questions"}] in group order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for i, g in enumerate(groups):
        p = out_dir / f"{dataset}_{i}.jpg"
        g["image"].save(p)
        flat.append({"dataset": dataset, "image_path": str(p),
                     "questions": list(g["questions"])})
    return flat


def fetch_grouped(spec, out_dir, n_images: int, max_scan: int | None = None) -> list:
    """Stream `spec.hf`, group rows by image, save images, return flat records.

    `spec` is a prune_viz DatasetSpec (uses .hf/.config/.split/.image_key/.q_key/
    .name). `max_scan` defaults to min(n_images*30, 400) to keep streaming cheap.
    Heavy `datasets` import deferred. Vast.ai only (HF streaming)."""
    from datasets import load_dataset
    if max_scan is None:
        max_scan = min(int(n_images) * 30, 400)
    ds = load_dataset(spec.hf, name=spec.config, split=spec.split, streaming=True)
    groups = group_rows(ds, spec.image_key, spec.q_key, int(n_images), max_scan)
    return save_groups(groups, spec.name, out_dir)
