"""Gradio interactive UI for text-visual attention reproduction.

Run:
    python -m vtp_eval.insight.text_visual_attention.ui              # binds 0.0.0.0:7860
    python -m vtp_eval.insight.text_visual_attention.ui --port 8000

Access from a laptop via SSH tunnel:
    ssh -p <VAST_PORT> -L 7860:localhost:7860 root@<HOST>
    # then open http://localhost:7860
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

from .coco import (DEFAULT_ANN_DIR, DEFAULT_ROOT,
                   download_candidate_image, ensure_coco_ann, pick_candidates)
from .tokens import build_default_query

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/text_visual_attention.yaml"
DEFAULT_OUT_DIR = (DEFAULT_ROOT / "outputs") if DEFAULT_ROOT.name == "workspace" \
    else (Path.cwd() / "outputs")

# Module-level state shared across UI events. Loaded lazily so that
# `python -m vtp_eval.figure3.ui` starts in <1 s without touching CUDA.
_CANDS: list = []
_IMG_INFO: dict = {}
_MODEL = None
_PROC = None


def _load_config(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def _load_candidates(min_cats, max_cats, n):
    """Refresh candidate gallery. Returns (gallery_rows, status_md)."""
    global _CANDS, _IMG_INFO
    try:
        ann_json = ensure_coco_ann(DEFAULT_ANN_DIR)
        _CANDS, _IMG_INFO = pick_candidates(
            ann_json, int(min_cats), int(max_cats), int(n))
        rows = []
        for i, c in enumerate(_CANDS):
            local = download_candidate_image(c["iid"], _IMG_INFO)
            cats = ", ".join(name for name, _ in c["categories"][:3])
            rows.append((str(local), f"idx={i}  {cats}"))
        return rows, f"Loaded {len(_CANDS)} candidates. Click a thumbnail to select."
    except Exception as exc:
        return [], f"**Error loading candidates:** {exc}"


def _on_select(evt: gr.SelectData):
    """Thumbnail clicked. Populate words checkbox group, clear query."""
    if not _CANDS or evt.index >= len(_CANDS):
        return ("Click *Load candidates* first.",
                gr.update(choices=[], value=[]),
                "",
                gr.update(interactive=False),
                None)
    chosen = _CANDS[evt.index]
    cat_names = [n for n, _ in chosen["categories"]]
    md = (f"**Selected:** idx={evt.index}  id={chosen['iid']}  "
          f"size={chosen['size']}")
    return (md,
            gr.update(choices=cat_names, value=[]),
            "",
            gr.update(interactive=False),
            evt.index)


def _on_words_change(words, idx):
    """Words toggled. Auto-fill query; enable Run iff image + >=1 word."""
    can_run = bool(words) and idx is not None
    if not words:
        return "", gr.update(interactive=False)
    return build_default_query(words), gr.update(interactive=can_run)


def _on_run(idx, words, query, ly_shallow, ly_middle, ly_deep, out_dir, model_id):
    """Full Figure 3 reproduction. Generator yields incremental status."""
    global _MODEL, _PROC
    if idx is None:
        yield "Select an image first.", None, None, None
        return
    if not words:
        yield "Pick at least one target word.", None, None, None
        return
    if not query:
        yield "Query is empty.", None, None, None
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        # Deferred to avoid loading torch/transformers on UI startup
        from .attention import extract_attention, load_llava
        from .metrics import compute_metrics
        from .visualize import plot_heatmap_grid, plot_metrics_bar

        if _MODEL is None:
            yield ("Loading LLaVA-1.5-7B (one-time, ~5 min on first run)...",
                   None, None, None)
            _MODEL, _PROC = load_llava(model_id)

        chosen = _CANDS[int(idx)]
        image = Image.open(
            download_candidate_image(chosen["iid"], _IMG_INFO)).convert("RGB")
        image.save(out / "chosen_image.jpg")

        yield "Extracting attention...", None, None, None
        layers = [int(ly_shallow), int(ly_middle), int(ly_deep)]
        pwl, _wp, grid, sinks, lyrs = extract_attention(
            _MODEL, _PROC, image, query, list(words), layers)

        yield "Rendering heatmap...", None, None, None
        heatmap_path = out / "figure3_reproduction.png"
        plot_heatmap_grid(pwl, image, sinks, grid, lyrs, query, heatmap_path)

        yield "Computing metrics...", None, None, None
        df = compute_metrics(pwl, sinks, lyrs)
        df.to_csv(out / "figure3_metrics.csv", index=False)
        bar_path = out / "figure3_metrics.png"
        plot_metrics_bar(df, lyrs, bar_path)

        yield (f"Done. Outputs saved to {out}/",
               str(heatmap_path), str(bar_path), df)
    except ValueError as exc:
        yield f"**Error:** {exc}", None, None, None
    except RuntimeError as exc:
        # CUDA OOM and friends - unload so a re-run can re-initialize cleanly
        _MODEL = None
        _PROC = None
        yield f"**GPU error:** {exc}. Model unloaded; restart UI.", None, None, None


def _build_ui(cfg: dict):
    nmin     = cfg.get("min_cats", 3)
    nmax     = cfg.get("max_cats", 6)
    ncand    = cfg.get("num_candidates", 25)
    layers   = cfg.get("layers", [2, 12, 30])
    model_id = cfg.get("model_id", "llava-hf/llava-1.5-7b-hf")

    with gr.Blocks(title="Figure 3 - LearnPruner") as demo:
        gr.Markdown(
            "# Figure 3 of LearnPruner - interactive reproduction\n"
            "Click *Load candidates*, pick a thumbnail, tick target words, "
            "then *Run Figure 3*. Outputs save to `/workspace/outputs/`.")
        with gr.Row():
            min_in   = gr.Number(value=nmin,  label="min_cats",       precision=0)
            max_in   = gr.Number(value=nmax,  label="max_cats",       precision=0)
            n_in     = gr.Number(value=ncand, label="num_candidates", precision=0)
            load_btn = gr.Button("Load candidates", variant="secondary")
        gallery = gr.Gallery(label="Candidates (click to select)",
                             columns=5, height=900, allow_preview=True)
        selected_md = gr.Markdown("**Selected:** *(none)*")
        idx_state   = gr.State(value=None)
        words       = gr.CheckboxGroup(label="Target words", choices=[])
        query_tb    = gr.Textbox(label="Query (auto-built, editable)",
                                 lines=1, interactive=True)
        with gr.Row():
            shallow = gr.Number(value=layers[0], label="shallow layer", precision=0)
            middle  = gr.Number(value=layers[1], label="middle layer",  precision=0)
            deep    = gr.Number(value=layers[2], label="deep layer",    precision=0)
        run_btn = gr.Button("Run Figure 3", variant="primary", interactive=False)
        status  = gr.Markdown("*Ready.*")
        with gr.Row():
            out_heatmap = gr.Image(label="figure3_reproduction.png", type="filepath")
            out_bar     = gr.Image(label="figure3_metrics.png",      type="filepath")
        out_df = gr.Dataframe(label="figure3_metrics.csv", interactive=False)
        out_dir_state   = gr.State(value=str(DEFAULT_OUT_DIR))
        model_id_state  = gr.State(value=model_id)

        load_btn.click(_load_candidates, [min_in, max_in, n_in],
                       [gallery, status])
        gallery.select(_on_select, None,
                       [selected_md, words, query_tb, run_btn, idx_state])
        words.change(_on_words_change, [words, idx_state],
                     [query_tb, run_btn])
        run_btn.click(_on_run,
                      [idx_state, words, query_tb, shallow, middle, deep,
                       out_dir_state, model_id_state],
                      [status, out_heatmap, out_bar, out_df])
    return demo


def launch(host: str = "0.0.0.0", port: int = 7860,
           config: Path | None = None) -> None:
    cfg = _load_config(Path(config) if config else DEFAULT_CONFIG)
    demo = _build_ui(cfg)
    # Gradio 5+ refuses to serve files outside cwd / /tmp unless explicitly
    # allowed. Candidate images and outputs live under DEFAULT_ROOT
    # (/workspace on Vast, ./ locally) — whitelist that whole tree.
    demo.launch(server_name=host, server_port=port, share=False,
                allowed_paths=[str(DEFAULT_ROOT)])


def main():
    p = argparse.ArgumentParser(prog="python -m vtp_eval.insight.text_visual_attention.ui")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args()
    launch(args.host, args.port, args.config)


if __name__ == "__main__":
    main()
