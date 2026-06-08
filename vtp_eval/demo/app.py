# vtp_eval/demo/app.py
"""Gradio Blocks demo: multi-turn QA on one image with the proposed pruning
method + retain-token cache. The image is picked from VQA benchmarks (grouped by
image) rather than uploaded. Launch on Vast.ai:  python -m vtp_eval.demo.app
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

from vtp_eval.demo.benchmarks import fetch_grouped
from vtp_eval.demo.engine import load_engine
from vtp_eval.insight.prune_viz.datasets import resolve_specs

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/prune_viz.yaml"
DEFAULT_ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
SAMPLE_DIR = DEFAULT_ROOT / "outputs/demo_samples"


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


def _benchmark_names() -> list:
    return list((_load_cfg().get("datasets") or {}).keys())


def _load_benchmarks(selected, n_images):
    """Fetch grouped images for each selected benchmark; return
    (gallery_rows, status_md, flat_state)."""
    if not selected:
        return [], "Select at least one benchmark.", []
    specs = resolve_specs(_load_cfg())
    flat, rows, failed = [], [], []
    for name in selected:
        spec = specs.get(name)
        if spec is None:
            continue
        try:
            groups = fetch_grouped(spec, SAMPLE_DIR / name, int(n_images))
        except Exception as exc:           # one bad dataset shouldn't kill the rest
            failed.append(f"{name} ({type(exc).__name__})")
            continue
        for i, g in enumerate(groups):
            flat.append(g)
            rows.append((g["image_path"], f"{name} #{i} · {len(g['questions'])} Qs"))
    status = f"Loaded {len(flat)} images from {len(selected)} benchmark(s)."
    if failed:
        status += " Failed: " + ", ".join(failed)
    return rows, status, flat


def _on_select(flat, evt: gr.SelectData):
    """Thumbnail clicked -> set the chat image, reset conversation, list its
    benchmark questions. Returns 10 outputs (see wiring below)."""
    if not flat or evt.index >= len(flat):
        return (None, [], [], gr.update(samples=[]), [],
                "Load benchmarks first.", "", None, None, "")
    s = flat[evt.index]
    img = Image.open(s["image_path"]).convert("RGB")
    qs = list(s["questions"] or [])
    samples = [[q] for q in qs]
    note = (f"**{s['dataset']}** — {len(qs)} question(s) for this image; "
            "click one to use it." if qs else "(no questions for this image)")
    # selected_image, chat, history, questions_ds, questions_state,
    # status, metrics, ov_r1, ov_r2, question
    return img, [], [], gr.update(samples=samples), qs, note, "", None, None, ""


def _on_pick_question(qs, evt: gr.SelectData):
    """A benchmark question row clicked -> put it in the input box."""
    if qs and evt.index < len(qs):
        return qs[evt.index]
    return ""


def build_ui(engine):
    stage2 = engine.cfg.stage2_enabled

    def respond(image, question, chat, history, use_cache):
        if image is None or not (question or "").strip():
            return chat, history, "Pick a benchmark image and a question.", None, None
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

    with gr.Blocks(title="LLaVA Pruning + Retain-Token Cache") as demo:
        gr.Markdown("# LLaVA-1.5 two-stage pruning + retain-token cache\n"
                    f"R1=384 (div 50%) → R2={'128' if stage2 else 'off'} · "
                    f"Stage-2: {'ON' if stage2 else 'OFF'}")
        history = gr.State([])
        flat_state = gr.State([])
        questions_state = gr.State([])

        with gr.Row():
            ds_in = gr.CheckboxGroup(choices=_benchmark_names(),
                                     value=_benchmark_names(), label="Benchmarks")
            n_in = gr.Number(value=5, precision=0, label="Images / benchmark")
            load_btn = gr.Button("Load", variant="secondary")
        status = gr.Markdown("*Pick benchmarks → Load → click a thumbnail.*")
        gallery = gr.Gallery(label="Images (click to select)", columns=5,
                             height=320, allow_preview=True)

        with gr.Row():
            with gr.Column(scale=1):
                selected_image = gr.Image(type="pil", interactive=False,
                                          label="Selected image")
                use_cache = gr.Checkbox(value=True, label="Enable retain-token cache")
                gr.Markdown(f"Stage-2 (R2 query-aware): **{'ON' if stage2 else 'OFF'}** "
                            "(set at launch)")
                with gr.Row():
                    ov_r1 = gr.Image(label="Kept after R1 (384)")
                    ov_r2 = gr.Image(label="Kept after R2 (128)")
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Conversation", height=320)
                questions_ds = gr.Dataset(components=[gr.Textbox(visible=False)],
                                          samples=[], label="Benchmark questions "
                                          "for this image (click to use)")
                question = gr.Textbox(label="Question",
                                      placeholder="Click a question above or type your own…")
                send = gr.Button("Send", variant="primary")
                metrics = gr.Markdown("")

        load_btn.click(_load_benchmarks, [ds_in, n_in], [gallery, status, flat_state])
        gallery.select(_on_select, [flat_state],
                       [selected_image, chat, history, questions_ds, questions_state,
                        status, metrics, ov_r1, ov_r2, question])
        questions_ds.select(_on_pick_question, [questions_state], [question])
        send.click(respond, [selected_image, question, chat, history, use_cache],
                   [chat, history, metrics, ov_r1, ov_r2])
        question.submit(respond, [selected_image, question, chat, history, use_cache],
                        [chat, history, metrics, ov_r1, ov_r2])
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
