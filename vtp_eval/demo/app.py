# vtp_eval/demo/app.py
"""Unified Gradio app: a shared benchmark image picker + three tabs (Pruning viz,
Text-visual attention, Inference compare) over ONE shared Engine.
Launch on Vast.ai:  python -m vtp_eval.demo.app  (or scripts/run_app.sh demo)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

from vtp_eval.demo.benchmarks import fetch_grouped
from vtp_eval.demo.engine import load_engine
from vtp_eval.demo.sections import OUT
from vtp_eval.demo.sections.inference import build_inference_tab
from vtp_eval.demo.sections.pruning import build_pruning_tab
from vtp_eval.demo.sections.tva import build_tva_tab
from vtp_eval.demo.sections.words import content_words
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
        except Exception as exc:
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
    """Thumbnail clicked -> set the shared image, clear the shared question, list
    its benchmark questions. Returns 5 shared outputs; per-tab resets run via
    chained .then handlers."""
    if not flat or evt.index >= len(flat):
        return None, "", gr.update(samples=[]), [], "Load benchmarks first."
    s = flat[evt.index]
    img = Image.open(s["image_path"]).convert("RGB")
    qs = list(s["questions"] or [])
    note = (f"**{s['dataset']}** — {len(qs)} question(s); click one to use it."
            if qs else "(no questions for this image)")
    return img, "", gr.update(samples=[[q] for q in qs]), qs, note


def _on_pick_question(qs, evt: gr.SelectData):
    """A benchmark question clicked -> set current_question, the inference input,
    and the TVA target-word choices."""
    if not qs or evt.index >= len(qs):
        return "", "", gr.update(choices=[], value=[])
    q = qs[evt.index]
    return q, q, gr.update(choices=content_words(q), value=[])


def build_ui(engine):
    with gr.Blocks(title="VTP-Eval unified demo") as demo:
        gr.Markdown("# Visual-token pruning — unified demo\n"
                    "Pick a benchmark image + question, then explore the three "
                    "tabs (pruning viz · text-visual attention · inference).")
        flat_state = gr.State([])
        questions_state = gr.State([])
        current_question = gr.State("")

        with gr.Row():
            ds_in = gr.CheckboxGroup(choices=_benchmark_names(),
                                     value=_benchmark_names(), label="Benchmarks")
            n_in = gr.Number(value=5, precision=0, label="Images / benchmark")
            load_btn = gr.Button("Load", variant="secondary")
        status = gr.Markdown("*Pick benchmarks → Load → click a thumbnail → click a question.*")
        gallery = gr.Gallery(label="Images (click to select)", columns=6,
                             height=240, allow_preview=True)
        with gr.Row():
            selected_image = gr.Image(type="pil", interactive=False,
                                      label="Selected image", height=240)
            questions_ds = gr.Dataset(components=[gr.Textbox(visible=False)],
                                      samples=[], label="Benchmark questions "
                                      "(click to use in all tabs)")

        with gr.Tabs():
            with gr.Tab("Pruning viz"):
                prune = build_pruning_tab(engine, selected_image, current_question)
            with gr.Tab("Text-visual attention"):
                tva = build_tva_tab(engine, selected_image, current_question)
            with gr.Tab("Inference (Proposed vs Vanilla)"):
                inf = build_inference_tab(engine, selected_image, current_question)

        load_btn.click(_load_benchmarks, [ds_in, n_in], [gallery, status, flat_state])
        sel = gallery.select(
            _on_select, [flat_state],
            [selected_image, current_question, questions_ds, questions_state, status])
        sel = sel.then(inf["reset_fn"], None, inf["reset_targets"])
        sel = sel.then(prune["reset_fn"], None, prune["reset_targets"])
        sel = sel.then(tva["reset_fn"], None, tva["reset_targets"])
        questions_ds.select(_on_pick_question, [questions_state],
                            [current_question, inf["question_tb"], tva["words"]])
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
    OUT.mkdir(parents=True, exist_ok=True)
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
              allowed_paths=[str(SAMPLE_DIR), str(OUT)])


if __name__ == "__main__":
    main()
