"""COCO val2017 annotation helpers for Figure 3 candidate selection."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path(".")
DEFAULT_ANN_DIR = DEFAULT_ROOT / "coco_ann"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017"


def ensure_coco_ann(ann_dir: Path = DEFAULT_ANN_DIR) -> Path:
    """Download + extract instances_val2017.json if missing. Returns its path."""
    ann_json = ann_dir / "annotations/instances_val2017.json"
    if ann_json.exists():
        return ann_json
    ann_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading COCO annotation zip (~241 MB) to {ann_dir}...", flush=True)
    zp = ann_dir.parent / "_coco_ann.zip"
    urllib.request.urlretrieve(COCO_ANN_URL, zp)
    with zipfile.ZipFile(zp) as zf:
        zf.extract("annotations/instances_val2017.json", ann_dir)
    zp.unlink()
    return ann_json


def pick_candidates(ann_json: Path, min_cats: int, max_cats: int, n: int):
    """Return (cands, img_info) where each cand is a dict
    {iid, size, categories: [(name, area), ...], total_area}."""
    with open(ann_json, "r") as f:
        coco = json.load(f)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {im["id"]: im for im in coco["images"]}

    per_img: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for a in coco["annotations"]:
        per_img[a["image_id"]][cat_name[a["category_id"]]] += a["area"]

    cands = []
    for iid, c2a in per_img.items():
        if not (min_cats <= len(c2a) <= max_cats):
            continue
        cands.append({
            "iid": iid,
            "size": (img_info[iid]["width"], img_info[iid]["height"]),
            "categories": sorted(c2a.items(), key=lambda kv: -kv[1]),
            "total_area": sum(c2a.values()),
        })
    cands.sort(key=lambda c: -c["total_area"])
    return cands[:n], img_info


def download_candidate_image(iid: int, img_info: dict,
                             image_dir: Path = DEFAULT_ROOT) -> Path:
    """Cache the COCO val2017 image locally; return the local path."""
    image_dir.mkdir(parents=True, exist_ok=True)
    local = image_dir / f"cand_{iid}.jpg"
    if not local.exists():
        urllib.request.urlretrieve(
            f"{COCO_IMG_URL}/{img_info[iid]['file_name']}", local)
    return local


def save_candidates_preview(cands, img_info, out_path: Path,
                            image_dir: Path = DEFAULT_ROOT) -> None:
    """3-column thumbnail grid, each tile titled with idx / id / top-5 categories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    n = len(cands)
    if n == 0:
        return
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, c in enumerate(cands):
        ax = axes[i]
        local = download_candidate_image(c["iid"], img_info, image_dir)
        ax.imshow(Image.open(local).convert("RGB"))
        cat_str = ", ".join(name for name, _ in c["categories"][:5])
        if len(c["categories"]) > 5:
            cat_str += f", +{len(c['categories']) - 5} more"
        ax.set_title(f"idx={i}  id={c['iid']}\n{cat_str}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def dump_candidates_table(cands, out_path: Path) -> None:
    """Print an ASCII table of candidates and write candidates.json."""
    rows = [{
        "idx": i, "id": c["iid"], "size": list(c["size"]),
        "categories": [{"name": n, "area": float(a)} for n, a in c["categories"]],
    } for i, c in enumerate(cands)]
    out_path.write_text(json.dumps(rows, indent=2))
    print("=" * 92)
    print(f"{'idx':<4}{'id':<10}{'size':<14}categories (area-ranked)")
    print("-" * 92)
    for i, c in enumerate(cands):
        cats = ", ".join(f"{n}({int(a)})" for n, a in c["categories"])
        print(f"{i:<4}{c['iid']:<10}{str(c['size']):<14}{cats}")
    print("=" * 92)
