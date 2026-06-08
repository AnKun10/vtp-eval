"""Gradio interactive UI for the prune_viz token-pruning figures.

Run:
    python -m vtp_eval.insight.prune_viz.ui              # binds 0.0.0.0:7860
    python -m vtp_eval.insight.prune_viz.ui --port 8000

Access from a laptop via SSH tunnel:
    ssh -p <VAST_PORT> -L 7860:localhost:7860 root@<HOST>
    # then open http://localhost:7860

Flow: pick datasets + #samples -> Load -> click a thumbnail -> choose the
avg-token budget (32/64/128) + diversity ratio -> Run -> view exp2/exp3 figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import yaml

from .datasets import fetch_samples, resolve_specs

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/prune_viz.yaml"
DEFAULT_ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
OUT_DIR = DEFAULT_ROOT / "outputs/prune_viz"
SAMPLE_DIR = OUT_DIR / "samples"

# avg-token budget (across 32 LLM layers) -> (R1 after vision prune, R2 after layer 12),
# at the R1:R2 = 3:1 (prune@layer 12) setting used throughout the experiments.
BUDGETS = {32: (54, 18), 64: (105, 35), 128: (213, 71)}

# Lazily-loaded shared state (so the UI starts in <1 s without touching CUDA).
_MODEL = None
_PROC = None
_TOK = None


def _load_config(path: Path) -> dict:
    p = Path(path)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def resolve_knobs(avg_budget: int, diversity_pct: float):
    """(avg_budget, diversity %) -> (R1, R2, dominant_k, diversity_m).

    R1/R2 come from the budget (3:1). diversity_m = round(R1 * pct/100), the rest
    is dominant; dominant_k is clamped to >=1 so select_stage1 always has a seed.
    """
    R1, R2 = BUDGETS[int(avg_budget)]
    diversity_m = int(round(R1 * float(diversity_pct) / 100.0))
    diversity_m = max(0, min(diversity_m, R1 - 1))   # keep >=1 dominant
    dominant_k = R1 - diversity_m
    return R1, R2, dominant_k, diversity_m


def _load_candidates(selected, n):
    """Fetch the first `n` samples of each selected dataset; return
    (gallery_rows, status_md, flat_samples)."""
    cfg = _load_config(DEFAULT_CONFIG)
    specs = resolve_specs(cfg)
    flat, rows = [], []
    for name in selected:
        spec = specs.get(name)
        if spec is None:
            continue
        spec.n = int(n)
        try:
            samples = fetch_samples(spec, SAMPLE_DIR / name)
        except Exception as exc:                       # one bad dataset shouldn't kill the rest
            rows.append((None, f"[{name} failed: {type(exc).__name__}]"))
            continue
        for s in samples:
            flat.append({"dataset": name, **s})
            rows.append((s["image_path"], f"{name} #{s['idx']}: {s['question'][:40]}"))
    status = (f"Loaded {len(flat)} samples from {len(selected)} dataset(s). "
              "Click a thumbnail to select.")
    return rows, status, flat


def _on_select(flat, evt: gr.SelectData):
    """Thumbnail clicked -> show its question, store the flat index."""
    if not flat or evt.index >= len(flat):
        return "Load samples first.", None
    s = flat[evt.index]
    return (f"**Selected:** {s['dataset']} #{s['idx']} — {s['question'][:90]}",
            evt.index)


def _on_run(idx, flat, avg_budget, diversity_pct, mode):
    """Render the chosen figures. Generator yields incremental status."""
    global _MODEL, _PROC, _TOK
    if idx is None:
        yield "Select an image first.", None, None, ""
        return
    R1, R2, dominant_k, diversity_m = resolve_knobs(avg_budget, diversity_pct)
    info = (f"**Budget avg-{int(avg_budget)}:** R1={R1} "
            f"(dominant {dominant_k} + diversity {diversity_m}, "
            f"{round(diversity_m / R1 * 100)}%), R2={R2}, prune@layer 12")
    s = flat[int(idx)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Deferred so the UI starts without torch/llava.
        from PIL import Image
        from llava.mm_utils import process_images
        from vtp_eval.proposed_method.config import ProposedConfig
        from . import extract, render

        if _MODEL is None:
            yield "Loading LLaVA-1.5-7B (one-time)...", None, None, info
            _TOK, _MODEL, _PROC = extract.load_model()

        image = Image.open(s["image_path"]).convert("RGB")
        it = process_images([image], _PROC, _MODEL.config)[0].unsqueeze(0).half()

        yield "Extracting R1 selections...", None, None, info
        attn_p, hid_p = extract.penultimate_vision(_MODEL, it)
        sels = extract.r1_selections(attn_p, hid_p, R1, dominant_k, diversity_m)
        stem = f"{s['dataset']}_{s['idx']}"
        exp2_path = exp3_path = None

        if mode in ("exp2", "both"):
            exp2_path = str(OUT_DIR / f"exp2_{stem}.png")
            render.plot_prune_row(
                image, [sels["attention"].cpu().numpy(), sels["diversity"].cpu().numpy()],
                ["Attention", "Diversity"], exp2_path,
                suptitle=f"{stem}  R1={R1} (dom {dominant_k} / div {diversity_m})")
            yield "Rendered Exp 2...", exp2_path, None, info

        if mode in ("exp3", "both"):
            cfg2 = ProposedConfig(dominant_k, diversity_m, 12, R2)
            r2 = extract.r2_selection(_TOK, _MODEL, it, s["question"], cfg2, sels["combined"])
            exp3_path = str(OUT_DIR / f"exp3_{stem}.png")
            render.plot_prune_row(
                image, [sels["combined"].cpu().numpy(), r2.cpu().numpy()],
                [f"After R1 ({R1})", f"After R2 ({R2})"], exp3_path,
                suptitle=f"{stem}  R1={R1} -> R2={R2}")

        yield "Done.", exp2_path, exp3_path, info
    except RuntimeError as exc:                         # CUDA OOM etc. -> unload for a clean re-run
        _MODEL = _PROC = _TOK = None
        yield f"**GPU error:** {exc}. Model unloaded; re-run.", None, None, info


def _build_ui(cfg: dict):
    names = list((cfg.get("datasets") or {}).keys())
    default_n = int(cfg.get("samples_per_dataset", 5))

    with gr.Blocks(title="prune_viz — token pruning") as demo:
        gr.Markdown(
            "# Token-pruning visualization (prune_viz)\n"
            "1) Pick datasets + #samples → **Load**. 2) Click a thumbnail. "
            "3) Choose budget + diversity → **Run**. Figures save to "
            "`/workspace/outputs/prune_viz/`.")
        with gr.Row():
            ds_in = gr.CheckboxGroup(choices=names, value=names, label="Datasets")
            n_in = gr.Number(value=default_n, label="Samples / dataset", precision=0)
            load_btn = gr.Button("Load samples", variant="secondary")
        gallery = gr.Gallery(label="Samples (click to select)", columns=5,
                             height=520, allow_preview=True)
        selected_md = gr.Markdown("**Selected:** *(none)*")
        flat_state = gr.State(value=[])
        idx_state = gr.State(value=None)
        with gr.Row():
            budget_in = gr.Radio(choices=[32, 64, 128], value=64,
                                 label="Avg-token budget (R1:R2 = 3:1)")
            div_in = gr.Slider(0, 90, value=50, step=10,
                               label="Diversity % of R1 (rest = attention/dominant)")
            mode_in = gr.Radio(choices=["exp2", "exp3", "both"], value="both",
                               label="Figure")
        run_btn = gr.Button("Run", variant="primary")
        status = gr.Markdown("*Ready.*")
        info_md = gr.Markdown("")
        with gr.Row():
            out_exp2 = gr.Image(label="Exp 2: Original | Attention | Diversity",
                                type="filepath")
            out_exp3 = gr.Image(label="Exp 3: Original | After R1 | After R2",
                                type="filepath")

        load_btn.click(_load_candidates, [ds_in, n_in],
                       [gallery, status, flat_state])
        gallery.select(_on_select, [flat_state], [selected_md, idx_state])
        run_btn.click(_on_run, [idx_state, flat_state, budget_in, div_in, mode_in],
                      [status, out_exp2, out_exp3, info_md])
    return demo


def launch(host: str = "0.0.0.0", port: int = 7860,
           config: Path | None = None, share: bool = False) -> None:
    cfg = _load_config(Path(config) if config else DEFAULT_CONFIG)
    demo = _build_ui(cfg)
    demo.launch(server_name=host, server_port=port, share=share,
                allowed_paths=[str(DEFAULT_ROOT)])


def main():
    p = argparse.ArgumentParser(prog="python -m vtp_eval.insight.prune_viz.ui")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--share", action="store_true", help="Gradio public share link")
    args = p.parse_args()
    launch(args.host, args.port, args.config, args.share)


if __name__ == "__main__":
    main()
