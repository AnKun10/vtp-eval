"""CLI entry: ``python -m vtp_eval.insight.text_visual_attention``.

Two modes:

    # 1) Surface candidates with their COCO category labels
    python -m vtp_eval.insight.text_visual_attention --list-samples

    # 2) Run the full pipeline picking image #N and target words from labels
    python -m vtp_eval.insight.text_visual_attention --chosen-index N --words "person" "sports ball"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

from .coco import (DEFAULT_ANN_DIR, DEFAULT_ROOT, download_candidate_image,
                   dump_candidates_table, ensure_coco_ann, pick_candidates,
                   save_candidates_preview)
from .tokens import build_default_query

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/text_visual_attention.yaml"
DEFAULT_OUT_DIR = (DEFAULT_ROOT / "outputs") if DEFAULT_ROOT.name == "workspace" \
    else (Path.cwd() / "outputs")


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m vtp_eval.insight.text_visual_attention",
        description="Reproduce Figure 3 of LearnPruner (ICLR 2026).",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--list-samples", action="store_true")
    p.add_argument("--chosen-index", type=int, default=0)
    p.add_argument("--words", nargs="+", default=None,
                   help='Target tokens, e.g. "person" "sports ball".')
    p.add_argument("--query", default=None,
                   help="Override the prompt question (must mention each --words verbatim).")
    p.add_argument("--layers", nargs=3, type=int, default=None,
                   metavar=("SHALLOW", "MIDDLE", "DEEP"))
    p.add_argument("--model-id", default=None)
    p.add_argument("--min-cats", type=int, default=None)
    p.add_argument("--max-cats", type=int, default=None)
    p.add_argument("--num-candidates", type=int, default=None)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args(argv)


def _resolve(args, cfg) -> dict:
    """CLI flag overrides config; config overrides hard-coded fallbacks."""
    return {
        "model_id":  args.model_id  or cfg.get("model_id",  "llava-hf/llava-1.5-7b-hf"),
        "layers":    args.layers    or cfg.get("layers",    [2, 12, 30]),
        "min_cats":  args.min_cats  if args.min_cats  is not None else cfg.get("min_cats",  3),
        "max_cats":  args.max_cats  if args.max_cats  is not None else cfg.get("max_cats",  6),
        "n_cand":    args.num_candidates if args.num_candidates is not None
                     else cfg.get("num_candidates", 8),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    s = _resolve(args, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_json = ensure_coco_ann(DEFAULT_ANN_DIR)
    cands, img_info = pick_candidates(ann_json, s["min_cats"], s["max_cats"], s["n_cand"])
    save_candidates_preview(cands, img_info, out_dir / "coco_candidates.png")
    dump_candidates_table(cands, out_dir / "candidates.json")
    print(f"Saved: {out_dir / 'coco_candidates.png'}")
    print(f"Saved: {out_dir / 'candidates.json'}")

    if args.list_samples:
        print()
        print("Next step: choose an idx and target words from the categories above.")
        print('Example: python -m vtp_eval.insight.text_visual_attention '
              '--chosen-index 0 --words "person" "sports ball"')
        return

    if args.words is None:
        sys.exit("Missing --words. Run --list-samples first, then pass words "
                 "from a candidate's category list.")

    if not (0 <= args.chosen_index < len(cands)):
        sys.exit(f"--chosen-index {args.chosen_index} out of range "
                 f"[0, {len(cands) - 1}].")

    chosen = cands[args.chosen_index]
    image = Image.open(download_candidate_image(chosen["iid"], img_info)).convert("RGB")
    image.save(out_dir / "chosen_image.jpg")
    print(f"\nChosen image idx={args.chosen_index}  id={chosen['iid']}  "
          f"size={image.size}")
    print(f"Categories: {[n for n, _ in chosen['categories']]}")
    print(f"Target words: {args.words}")

    query = args.query if args.query is not None else build_default_query(args.words)
    print(f'Query: "{query}"')

    # Heavy imports deferred so --list-samples is fast and lightweight.
    from .attention import extract_attention, load_llava
    from .metrics import compute_metrics
    from .visualize import plot_heatmap_grid, plot_metrics_bar

    print("\nLoading LLaVA-1.5-7B...", flush=True)
    model, proc = load_llava(s["model_id"])
    pwl, word_positions, grid, sinks, lyrs = extract_attention(
        model, proc, image, query, args.words, s["layers"])
    print("Word positions:", word_positions)
    print("Sink indices:", sorted(sinks))

    plot_heatmap_grid(pwl, image, sinks, grid, lyrs, query,
                      out_dir / "text_visual_attention_reproduction.png")
    print(f"Saved: {out_dir / 'text_visual_attention_reproduction.png'}")

    df = compute_metrics(pwl, sinks, lyrs)
    df.to_csv(out_dir / "text_visual_attention_metrics.csv", index=False)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_metrics_bar(df, lyrs, out_dir / "text_visual_attention_metrics.png")
    print(f"Saved: {out_dir / 'text_visual_attention_metrics.png'}")


if __name__ == "__main__":
    main()
