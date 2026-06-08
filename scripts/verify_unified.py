# scripts/verify_unified.py
"""Prove ONE resident LLaVA model serves demo + prune_viz + TVA (Vast.ai, GPU).

    python scripts/verify_unified.py

Loads one Engine, fetches one POPE benchmark image, runs all three tools, saves
outputs under /workspace/outputs/unified/, and prints a summary. Manual visual
check: the TVA heatmap highlights the queried region; exp2/exp3 look right.
"""
from pathlib import Path

import yaml
from PIL import Image

from vtp_eval.demo.benchmarks import fetch_grouped
from vtp_eval.demo.engine import load_engine
from vtp_eval.insight.prune_viz import render
from vtp_eval.insight.prune_viz.datasets import resolve_specs
from vtp_eval.insight.text_visual_attention import metrics, visualize

ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
OUT = ROOT / "outputs/unified"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    engine = load_engine()

    cfg = yaml.safe_load((Path("configs/prune_viz.yaml")).read_text())
    spec = resolve_specs(cfg)["pope"]
    rec = fetch_grouped(spec, OUT / "samples" / "pope", n_images=1)[0]
    img = Image.open(rec["image_path"]).convert("RGB")
    question = rec["questions"][0]
    print(f"[unified] image={rec['image_path']}  Q={question!r}")

    # 1. demo
    res = engine.run_turn(img, question, history=[])
    print(f"[demo]    {res.latency_s:.2f}s tokens={res.tokens} ans={res.answer[:80]!r}")

    # 2. prune_viz exp2 / exp3
    pv = engine.prune_viz_figures(img, question, R1=128, R2=64,
                                  dominant_k=96, diversity_m=32)
    render.plot_prune_row(
        img, [pv["attention"].cpu().numpy(), pv["diversity"].cpu().numpy()],
        ["Attention", "Diversity"], OUT / "exp2.png", suptitle="exp2  R1=128")
    render.plot_prune_row(
        img, [pv["combined"].cpu().numpy(), pv["r2"].cpu().numpy()],
        [f"After R1 ({pv['R1']})", f"After R2 ({pv['R2']})"],
        OUT / "exp3.png", suptitle="exp3  R1=128 -> R2=64")
    print(f"[pruneviz] saved {OUT/'exp2.png'} and {OUT/'exp3.png'}")

    # 3. TVA — longest content word from the benchmark question
    word = max((w.strip(".,?!\"'") for w in question.split()), key=len)
    tva = engine.tva_attention(img, question, [word], layers=(2, 12, 30))
    visualize.plot_heatmap_grid(tva["pwl"], img, tva["sinks"], tva["grid"],
                                tva["lyrs"], question, OUT / "tva_heatmap.png")
    metrics.compute_metrics(tva["pwl"], tva["sinks"], tva["lyrs"]).to_csv(
        OUT / "tva_metrics.csv", index=False)
    print(f"[tva]     word={word!r} -> saved {OUT/'tva_heatmap.png'} + metrics")

    print("[unified] OK — one resident model served demo + prune_viz + TVA")


if __name__ == "__main__":
    main()
