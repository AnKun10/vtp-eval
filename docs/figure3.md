# Figure 3 of LearnPruner — Reproduction

This sub-experiment reproduces **Figure 3** of *LearnPruner: Rethinking Attention-Based Token Pruning in Vision Language Models* (ICLR 2026) — the visual evidence behind the paper's "prune in the middle layer" decision.

## What Figure 3 shows

> *"In both shallow and deep layers, uninformative regions tend to absorb most of the attention from text tokens, while in the middle layer, different text tokens are able to focus on their semantic-related regions, hence allowing for effective token selection."* — Section 3.2

For a single (image, query) pair, we visualize the attention from **specific text tokens** (e.g. `man`, `balls`) to the 576 vision-token positions across three LLM depths:

| Layer | Expected behavior |
|-------|-------------------|
| **Shallow** (layer 2)  | Diffuse — drawn to background / attention-sink tokens |
| **Middle** (layer 12)  | Focused on the semantically related region |
| **Deep** (layer 30)    | Collapses again onto a few outliers |

Middle = layer 12 is exactly the layer LearnPruner prunes at (`k = 12` in Section 3.3).

## Pipeline

```
COCO val2017 annotations
        │  filter: 3–6 distinct categories per image
        ▼
Top-N candidate images (ranked by total annotation area)
        │  user picks idx + target words from annotation labels
        ▼
LLaVA-1.5-7B + AutoProcessor (attn_implementation="eager", fp16)
        │  prompt = "USER: <image>\n<query> ASSISTANT:"
        │  forward(output_attentions=True)
        ▼
Per-(word, layer) attention vectors over the 576 vision tokens
        │  head-averaged; multi-piece words averaged across their pieces
        ▼
Sink masking (zero out global top-1 sink indices)
        │  percentile clip (99%)
        ▼
Heatmap overlay grid (rows = words, cols = depths)
Concentration metrics (entropy, max-share, top-5% mass)
```

## Layout

```
vtp_eval/figure3/
  __init__.py
  __main__.py    # CLI: python -m vtp_eval.figure3 ...
  coco.py        # ensure_coco_ann, pick_candidates, previews
  tokens.py      # find_word_positions (multi-token), build_default_query
  attention.py   # load_llava, extract_attention
  visualize.py   # overlay_from_vec, plot_heatmap_grid, plot_metrics_bar
  metrics.py     # compute_metrics

install/figure3.sh                    # transformers==4.49 + pip install -e .
scripts/figure3_vast_onstart.sh       # ~30-line Vast on-start
scripts/figure3_list_samples.sh
scripts/figure3_run.sh
configs/figure3.yaml                  # model_id, layers, candidate filters
notebooks/figure3_reproduction.ipynb  # interactive Jupyter version
```

## Usage

### On Vast.ai (recommended for the 14 GB LLaVA download)

1. Open `https://cloud.vast.ai/create/`.
2. Filters: VRAM ≥ 24 GB, Disk ≥ 50 GB, Verified. Pick RTX 3090/4090/A5000/L4/A100.
3. Disk → **50 GB**. Launch mode: *Jupyter + SSH* (default).
4. Env vars:
   - `HF_HOME` → `/workspace/.cache/huggingface`
   - `JUPYTER_DIR` → `/workspace`
   - `DATA_DIRECTORY` → `/workspace/`
5. Paste the entire content of `scripts/figure3_vast_onstart.sh` into the On-start Script box.
6. **RENT**. Wait ~30 s for onstart, then SSH in.

Inside SSH:

```bash
# Optional: confirm onstart finished cleanly
tail -n 20 /workspace/onstart.log

# 1. Surface 8 candidate images with their COCO category labels
bash scripts/figure3_list_samples.sh
# (or: python -m vtp_eval.figure3 --list-samples)
# Writes:  /workspace/outputs/coco_candidates.png  +  candidates.json
# Prints:  idx  id  size  categories(area)

# 2. Pick an idx and words from that row's category list
bash scripts/figure3_run.sh 0 person bicycle
# (or: python -m vtp_eval.figure3 --chosen-index 0 --words "person" "bicycle")
# LLaVA downloads on first call (~14 GB, ~5 min); subsequent runs ~30 s.
```

Multi-word categories (`"sports ball"`, `"tennis racket"`, `"fire hydrant"`) are supported — the script tokenizes them and averages attention across all pieces.

### Locally (24 GB+ GPU)

```bash
bash install/figure3.sh
python -m vtp_eval.figure3 --list-samples
python -m vtp_eval.figure3 --chosen-index 0 --words "person" "bicycle"
```

### Jupyter

Open `notebooks/figure3_reproduction.ipynb` and run all cells. The notebook imports the same `vtp_eval.figure3.*` modules, so behavior matches the CLI.

## Outputs

Under `/workspace/outputs/` (or `./outputs/` locally):

| File | Description |
|------|-------------|
| `coco_candidates.png` | 8-tile thumbnail grid with annotation labels |
| `candidates.json` | machine-readable candidate list |
| `chosen_image.jpg` | the image the run actually used |
| `figure3_reproduction.png` | **the Figure 3 reproduction** — heatmap grid |
| `figure3_metrics.csv` | entropy / max_share / top5pct_mass per (word, depth) |
| `figure3_metrics.png` | bar chart visualizing the same metrics |

## Interpreting the result

The paper's insight predicts, for every target word, that the **middle** column should win on all three concentration metrics:

| token  | depth   | layer | entropy ↓ | max_share ↑ | top5pct_mass ↑ |
|--------|---------|-------|-----------|-------------|----------------|
| word_A | shallow | 2     | …         | …           | …              |
| word_A | middle  | 12    | **min**   | **max**     | **max**        |
| word_A | deep    | 30    | …         | …           | …              |

Visually, the middle column of `figure3_reproduction.png` should show a heatmap that clearly aligns with where the named object actually is in the image (per the COCO annotation).

## Troubleshooting

- **`onstart.log` halts mid-stream:** re-run `bash scripts/figure3_vast_onstart.sh` — every step is idempotent.
- **`torch.cuda.is_available()` is False:** the rented instance is not exposing the GPU. Destroy and pick a different host.
- **CUDA OOM on forward pass:** the instance has < 24 GB VRAM. LLaVA-1.5-7B fp16 peaks around 16 GB; rent a 24 GB+ card.
- **"Words not found as contiguous tokens":** the auto-built query mentions each `--words` entry verbatim, but if you pass `--query` yourself it must contain them too. Multi-word categories must be quoted: `--words "sports ball"`.

## References

- Paper: *LearnPruner: Rethinking Attention-Based Token Pruning in Vision Language Models*, Takezoe et al., ICLR 2026.
- Model: [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Dataset: [COCO val2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
