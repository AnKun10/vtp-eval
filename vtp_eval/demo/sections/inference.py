# vtp_eval/demo/sections/inference.py
"""Inference tab: multi-turn chat answering with Proposed (pruned) vs Vanilla."""
from __future__ import annotations

import gradio as gr

from vtp_eval.demo.compare import format_comparison


def build_inference_tab(engine, selected_image, current_question):
    """Build the Inference tab. Returns a dict with the cross-wired `question_tb`
    plus a `reset_fn`/`reset_targets` the app clears on a new image."""
    proposed_history = gr.State([])
    vanilla_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                ov_r1 = gr.Image(label="Kept after R1 (384)")
                ov_r2 = gr.Image(label="Kept after R2 (128)")
            use_cache = gr.Checkbox(value=True, label="Enable retain-token cache")
        with gr.Column(scale=1):
            chat = gr.Chatbot(label="Proposed vs Vanilla", height=300)
            question_tb = gr.Textbox(
                label="Question", placeholder="Click a benchmark question above or type…")
            send = gr.Button("Send", variant="primary")

    def respond(image, question, chat_v, p_hist, v_hist, use_cache_v):
        if image is None or not (question or "").strip():
            return chat_v, p_hist, v_hist, None, None
        try:
            res = engine.run_turn(image, question, p_hist, use_cache=use_cache_v)
            no_cache = (engine.compare_latency(image, question, p_hist)
                        if res.cache_hit else None)
            v_answer, v_latency = engine.vanilla_generate(image, question, v_hist)
        except Exception as e:
            chat_v = chat_v + [{"role": "user", "content": question},
                               {"role": "assistant",
                                "content": f"**Error:** {type(e).__name__}: {e}"}]
            return chat_v, p_hist, v_hist, None, None
        bubble = format_comparison(res.answer, res.latency_s, res.cache_hit,
                                   res.tokens[2], no_cache, v_answer, v_latency,
                                   res.tokens[0])
        chat_v = chat_v + [{"role": "user", "content": question},
                           {"role": "assistant", "content": bubble}]
        p_hist = p_hist + [(question, res.answer)]
        v_hist = v_hist + [(question, v_answer)]
        return chat_v, p_hist, v_hist, res.overlay_r1, res.overlay_r2

    ins = [selected_image, question_tb, chat, proposed_history, vanilla_history, use_cache]
    outs = [chat, proposed_history, vanilla_history, ov_r1, ov_r2]
    send.click(respond, ins, outs)
    question_tb.submit(respond, ins, outs)

    def reset_fn():
        return [], [], [], "", None, None

    return {"question_tb": question_tb, "reset_fn": reset_fn,
            "reset_targets": [chat, proposed_history, vanilla_history,
                              question_tb, ov_r1, ov_r2]}
