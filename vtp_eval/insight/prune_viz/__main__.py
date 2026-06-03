"""CLI: python -m vtp_eval.insight.prune_viz

    --list-samples                          # fetch + list candidates per dataset
    --dataset gqa --index 2 --mode exp2 --r1 64
    --dataset textvqa --index 0 --mode exp3 --dominant-k 54 --diversity-m 10 --r2 37
    --dataset pope --index 1 --mode both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .datasets import fetch_samples, resolve_specs

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/prune_viz.yaml"
DEFAULT_ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
DEFAULT_OUT = DEFAULT_ROOT / "outputs/prune_viz"
SAMPLE_DIR = DEFAULT_OUT / "samples"


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m vtp_eval.insight.prune_viz")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--list-samples", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--mode", choices=["exp2", "exp3", "both"], default="both")
    p.add_argument("--r1", type=int, default=64)
    p.add_argument("--dominant-k", type=int, default=54)
    p.add_argument("--diversity-m", type=int, default=10)
    p.add_argument("--r2", type=int, default=37)
    p.add_argument("--pruned-layer", type=int, default=12)
    p.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return p.parse_args(argv)


def _candidates(cfg) -> dict:
    specs = resolve_specs(cfg)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, spec in specs.items():
        try:
            table[name] = fetch_samples(spec, SAMPLE_DIR / name)
        except Exception as e:        # one bad dataset shouldn't kill the rest
            print(f"[warn] {name}: {type(e).__name__}: {e}")
    (SAMPLE_DIR / "candidates.json").write_text(json.dumps(table, indent=2))
    return table


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    table = _candidates(cfg)
    if args.list_samples:
        for name, rows in table.items():
            print(f"\n=== {name} ({len(rows)}) ===")
            for r in rows:
                print(f"  [{r['idx']}] {r['question'][:70]}")
        print(f"\nSaved: {SAMPLE_DIR / 'candidates.json'}")
        return

    if not args.dataset or args.dataset not in table:
        raise SystemExit(f"--dataset must be one of {list(table)}")
    rows = table[args.dataset]
    if not (0 <= args.index < len(rows)):
        raise SystemExit(f"--index out of range [0, {len(rows) - 1}]")
    sample = rows[args.index]

    # Heavy imports deferred so --list-samples never needs torch/llava.
    from PIL import Image
    from llava.mm_utils import process_images
    from vtp_eval.proposed_method.config import ProposedConfig
    from . import extract, render

    print(f"Loading {args.model} ...", flush=True)
    tok, model, image_processor = extract.load_model(args.model)
    image = Image.open(sample["image_path"]).convert("RGB")
    it = process_images([image], image_processor, model.config)[0].unsqueeze(0).half()

    attn_p, hid_p = extract.penultimate_vision(model, it)
    sels = extract.r1_selections(attn_p, hid_p, args.r1, args.dominant_k, args.diversity_m)
    stem = f"{args.dataset}_{args.index}"

    if args.mode in ("exp2", "both"):
        render.plot_prune_row(
            image, [sels["attention"].cpu().numpy(), sels["diversity"].cpu().numpy()],
            ["Attention", "Diversity"], out_dir / f"exp2_{stem}.png",
            suptitle=f"{args.dataset} #{args.index}  R1={args.r1}")
        print(f"Saved: {out_dir / f'exp2_{stem}.png'}")

    if args.mode in ("exp3", "both"):
        cfg2 = ProposedConfig(args.dominant_k, args.diversity_m,
                              args.pruned_layer, args.r2)
        r2 = extract.r2_selection(tok, model, it, sample["question"], cfg2,
                                  sels["combined"])
        render.plot_prune_row(
            image, [sels["combined"].cpu().numpy(), r2.cpu().numpy()],
            [f"After R1 ({cfg2.r1})", f"After R2 ({args.r2})"],
            out_dir / f"exp3_{stem}.png",
            suptitle=f"{args.dataset} #{args.index}  R1={cfg2.r1} -> R2={args.r2}")
        print(f"Saved: {out_dir / f'exp3_{stem}.png'}")


if __name__ == "__main__":
    main()
