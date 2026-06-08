# vtp_eval/demo/sections/tva.py
"""Text-visual-attention tab: tick target words from the question + layers ->
the Figure-3 per-word attention heatmap + concentration metrics, via
Engine.tva_attention."""
from __future__ import annotations

import gradio as gr

from vtp_eval.demo.sections import OUT


def build_tva_tab(engine, selected_image, current_question):
    """Build the TVA tab. Returns the cross-wired `words` checkbox plus
    reset_fn/reset_targets."""
    words = gr.CheckboxGroup(choices=[], label="Target words (from the question)")
    with gr.Row():
        ly_s = gr.Number(value=2, label="shallow layer", precision=0)
        ly_m = gr.Number(value=12, label="middle layer", precision=0)
        ly_d = gr.Number(value=30, label="deep layer", precision=0)
        run = gr.Button("Run text-visual attention", variant="primary")
    status = gr.Markdown("")
    with gr.Row():
        heatmap = gr.Image(label="Figure 3: per-word attention", type="filepath")
        metrics_img = gr.Image(label="Concentration (entropy / top-5% mass)",
                               type="filepath")
    df = gr.Dataframe(label="metrics", interactive=False)

    def run_fn(image, question, words_v, s, m, d):
        if image is None or not (question or "").strip():
            return "Pick a benchmark image and a question first.", None, None, None
        if not words_v:
            return "Tick at least one target word.", None, None, None
        from vtp_eval.insight.text_visual_attention import metrics, visualize
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            tva = engine.tva_attention(image, question, list(words_v),
                                       (int(s), int(m), int(d)))
            visualize.plot_heatmap_grid(tva["pwl"], image, tva["sinks"],
                                        tva["grid"], tva["lyrs"], question,
                                        str(OUT / "tva_heatmap.png"))
            mdf = metrics.compute_metrics(tva["pwl"], tva["sinks"], tva["lyrs"])
            visualize.plot_metrics_bar(mdf, tva["lyrs"], str(OUT / "tva_metrics.png"))
        except Exception as e:
            return f"**Error:** {type(e).__name__}: {e}", None, None, None
        return "Done.", str(OUT / "tva_heatmap.png"), str(OUT / "tva_metrics.png"), mdf

    run.click(run_fn, [selected_image, current_question, words, ly_s, ly_m, ly_d],
              [status, heatmap, metrics_img, df])

    def reset_fn():
        return gr.update(choices=[], value=[]), "", None, None, None

    return {"words": words, "reset_fn": reset_fn,
            "reset_targets": [words, status, heatmap, metrics_img, df]}
