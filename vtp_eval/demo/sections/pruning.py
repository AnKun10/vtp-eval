# vtp_eval/demo/sections/pruning.py
"""Pruning-viz tab: budget/diversity -> exp2 (Attention|Diversity R1 sets) and
exp3 (R1 -> R2) overlays, computed via Engine.prune_viz_figures."""
from __future__ import annotations

import gradio as gr

from vtp_eval.demo.sections import OUT


def build_pruning_tab(engine, selected_image, current_question):
    """Build the Pruning-viz tab. Returns reset_fn/reset_targets."""
    from vtp_eval.insight.prune_viz.ui import resolve_knobs  # gradio-safe import

    with gr.Row():
        budget = gr.Radio([32, 64, 128], value=64,
                          label="Avg-token budget (R1:R2 = 3:1)")
        diversity = gr.Slider(0, 90, value=50, step=10,
                              label="Diversity % of R1 (rest = attention)")
        run = gr.Button("Run pruning viz", variant="primary")
    info = gr.Markdown("")
    with gr.Row():
        exp2 = gr.Image(label="exp2: Original | Attention | Diversity", type="filepath")
        exp3 = gr.Image(label="exp3: Original | After R1 | After R2", type="filepath")

    def run_fn(image, question, budget_v, diversity_v):
        if image is None:
            return "Pick a benchmark image first.", None, None
        import time

        from vtp_eval.insight.prune_viz import render
        q = (question or "").strip() or "What is in the image?"
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)          # unique names so gradio re-serves
        exp2_p, exp3_p = OUT / f"exp2_{stamp}.png", OUT / f"exp3_{stamp}.png"
        try:
            R1, R2, dom, div = resolve_knobs(budget_v, diversity_v)
            pv = engine.prune_viz_figures(image, q, R1, R2, dom, div)
            render.plot_prune_row(
                image, [pv["attention"].cpu().numpy(), pv["diversity"].cpu().numpy()],
                ["Attention", "Diversity"], str(exp2_p),
                suptitle=f"R1={R1} (dom {dom} / div {div})")
            render.plot_prune_row(
                image, [pv["combined"].cpu().numpy(), pv["r2"].cpu().numpy()],
                [f"After R1 ({R1})", f"After R2 ({R2})"], str(exp3_p),
                suptitle=f"R1={R1} -> R2={R2}")
        except Exception as e:
            return f"**Error:** {type(e).__name__}: {e}", None, None
        md = (f"**R1={R1}** (dominant {dom} + diversity {div}), **R2={R2}**, "
              f"prune@layer 12 · query: _{q[:60]}_")
        return md, str(exp2_p), str(exp3_p)

    run.click(run_fn, [selected_image, current_question, budget, diversity],
              [info, exp2, exp3])

    def reset_fn():
        return "", None, None

    return {"reset_fn": reset_fn, "reset_targets": [info, exp2, exp3]}
