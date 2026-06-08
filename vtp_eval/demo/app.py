# vtp_eval/demo/app.py
"""Gradio Blocks demo: multi-turn QA on one image with the proposed pruning
method + retain-token cache. Launch on Vast.ai:  python -m vtp_eval.demo.app
"""
from __future__ import annotations

import argparse

import gradio as gr

from vtp_eval.demo.engine import load_engine


def build_ui(engine):
    stage2 = engine.cfg.stage2_enabled

    def respond(image, question, chat, history, use_cache):
        if image is None or not (question or "").strip():
            return chat, history, "Upload an image and type a question.", None, None
        try:
            res = engine.run_turn(image, question, history, use_cache=use_cache)
            compare = engine.compare_latency(image, question, history)
        except Exception as e:  # surface errors instead of a raw stack trace
            return chat, history, f"**Error:** {type(e).__name__}: {e}", None, None
        history = history + [(question, res.answer)]
        chat = chat + [{"role": "user", "content": question},
                       {"role": "assistant", "content": res.answer}]
        s576, r1, r2 = res.tokens
        metrics = (
            f"**Latency:** {res.latency_s:.2f}s  "
            f"({'CACHE HIT' if res.cache_hit else 'cache miss'})\n\n"
            f"**Tokens:** {s576} → {r1} → {r2}\n\n"
            f"**No-cache (same turn):** {compare:.2f}s"
        )
        return chat, history, metrics, res.overlay_r1, res.overlay_r2

    def on_new_image(_image):
        # New image -> reset conversation (cache keys by image hash automatically).
        return [], [], "New image loaded — conversation reset.", None, None

    with gr.Blocks(title="LLaVA Pruning + Retain-Token Cache") as demo:
        gr.Markdown("# LLaVA-1.5 two-stage pruning + retain-token cache\n"
                    f"R1=384 (div 50%) → R2={'128' if stage2 else 'off'} · "
                    f"Stage-2: {'ON' if stage2 else 'OFF'}")
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Image")
                use_cache = gr.Checkbox(value=True, label="Enable retain-token cache")
                gr.Markdown(f"Stage-2 (R2 query-aware): **{'ON' if stage2 else 'OFF'}** "
                            "(set at launch)")
                with gr.Row():
                    ov_r1 = gr.Image(label="Kept after R1 (384)")
                    ov_r2 = gr.Image(label="Kept after R2 (128)")
            with gr.Column(scale=1):
                # Feed messages-format dicts ({"role","content"}); newer Gradio
                # (6.x) dropped the tuple format and the `type` kwarg entirely.
                chat = gr.Chatbot(label="Conversation", height=380)
                question = gr.Textbox(label="Question", placeholder="Ask about the image…")
                send = gr.Button("Send", variant="primary")
                metrics = gr.Markdown("")

        send.click(respond, [image, question, chat, history, use_cache],
                   [chat, history, metrics, ov_r1, ov_r2])
        question.submit(respond, [image, question, chat, history, use_cache],
                        [chat, history, metrics, ov_r1, ov_r2])
        image.upload(on_new_image, [image], [chat, history, metrics, ov_r1, ov_r2])
    return demo


def main():
    p = argparse.ArgumentParser(prog="python -m vtp_eval.demo.app")
    p.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    p.add_argument("--no-stage2", action="store_true",
                   help="disable Stage-2 R2 prune (fallback if decode path breaks)")
    p.add_argument("--share", action="store_true", help="Gradio public share link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    engine = load_engine(model_path=args.model, stage2_enabled=not args.no_stage2)
    ui = build_ui(engine)
    ui.queue()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
