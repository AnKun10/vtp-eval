"""Config-driven partial fetch of benchmark samples for visualization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class DatasetSpec:
    name: str
    hf: str
    split: str
    n: int
    image_key: str = "image"
    q_key: str = "question"


def resolve_specs(cfg: dict) -> Dict[str, DatasetSpec]:
    """Turn a prune_viz.yaml dict into {name: DatasetSpec}. `n` defaults to
    samples_per_dataset; image_key/q_key default to image/question. `hf` is
    required (KeyError if missing)."""
    default_n = int(cfg.get("samples_per_dataset", 5))
    out: Dict[str, DatasetSpec] = {}
    for name, d in (cfg.get("datasets") or {}).items():
        out[name] = DatasetSpec(
            name=name,
            hf=d["hf"],                         # required
            split=d.get("split", "test"),
            n=int(d.get("n", default_n)),
            image_key=d.get("image_key", "image"),
            q_key=d.get("q_key", "question"),
        )
    return out


def fetch_samples(spec: DatasetSpec, out_dir: Path) -> list:
    """Stream the first `spec.n` samples; save images + return sample metadata.

    Uses HF datasets streaming so only `n` rows download (not the full set).
    Returns [{idx, image_path, question}]. Heavy import deferred.
    """
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(spec.hf, split=spec.split, streaming=True)
    rows = []
    for idx, row in enumerate(ds):
        if idx >= spec.n:
            break
        img = row[spec.image_key]
        if hasattr(img, "convert"):                 # PIL image
            img = img.convert("RGB")
        else:                                        # path or bytes
            from PIL import Image
            import io
            img = (Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict)
                   else Image.open(img)).convert("RGB")
        p = out_dir / f"{spec.name}_{idx}.jpg"
        img.save(p)
        rows.append({"idx": idx, "image_path": str(p),
                     "question": str(row.get(spec.q_key, ""))})
    return rows
