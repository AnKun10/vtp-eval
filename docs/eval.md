# Benchmarking the proposed method with lmms-eval

Harness to benchmark the proposed pruning method vs a vanilla LLaVA-1.5-7B
baseline. Accuracy via lmms-eval; efficiency via a 3-stage latency breakdown
(vision encoder / LLM prefill / LLM decode), total latency, peak memory, token
budget, and a theoretical-TFLOPs estimate. Design:
`docs/superpowers/specs/2026-06-02-lmms-eval-benchmark-harness-design.md`.

## What it benchmarks
`configs/eval.yaml` runs: `baseline` (vanilla), `proposed_stage1_R1-64`
(Stage-1 vision prune only, 576→64 — clean ablation) and
`proposed_R1-64_R2-37_k12` (full Stage 1+2). Stage 2 uses the FastV no-slice
decode path (KV cache untouched, original positions + full rotary), GPU-verified
to keep decode at seq==1 (no re-feed) so decode latency measures correctly. Add
runs/benchmarks by editing the `runs` / `tasks` lists. Other methods
(VisionZip/FastV/…) are compared via their published numbers (out of scope here).

## Run on Vast.ai
1. Rent with the proposed-method template (py310/torch-2.1.2 image + the §2
   on-start from `docs/vast_proposed_method.md`). The on-start now also installs
   lmms-eval v0.5 (via `install/proposed.sh`). Disk **≥120 GB** (POPE COCO images + GQA / TextVQA / MMBench image caches + HF datasets cache).
2. SSH in. The venv auto-activates only in an interactive shell — for scripts
   run `source /venv/main/bin/activate` first. Then:
   `export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1`.
   `HF_HUB_DISABLE_XET=1` is important: recent `huggingface_hub` defaults to the
   xet backend, which on Vast stalled at 0 B/s ("connection struggling"); plain
   HTTPS + `hf_transfer` downloads the 13.5 GB model at ~12 MB/s. (The benchmark
   task yamls' `token: True` is flipped to `False` by `install/proposed.sh` so
   the public lmms-lab datasets load without an HF login.)
3. Smoke (single cell): `bash scripts/eval/run.sh baseline gqa 20`
   → check `results/baseline/gqa/{results.json,timing.json}`.
4. Full sweep: `bash scripts/eval/run_all.sh` (all runs × all 7 tasks) → `results/summary.csv`.
   Quick whole-matrix sanity first: `bash scripts/eval/run_all.sh 20` (limit 20 per cell).
5. Retrieve: `scp -P <PORT> root@<HOST>:/workspace/vtp-eval/results/summary.csv .`
6. **Stop** (not Destroy) to keep the model + dataset cache.

## summary.csv columns
`method, task, metric, value, avg_tokens, keep_ratio_pct, encoder_ms,
prefill_ms, decode_ms, total_ms, peak_mem_mb, tflops`. The measured stage
latencies are the primary efficiency numbers; `tflops` is an approximate
theoretical estimate from `avg_tokens`. One row per (method, task, metric); **MME
contributes two rows** — `mme_perception` and `mme_total` (perception + cognition).

## Notes
- POPE ≈ 9k samples; at batch=1 a full pass per run is the slow part — use
  `--limit` (3rd arg to `run.sh`) for a quick sanity number first.
- Timing drops the first batch as warm-up; per-sample latency = batch time /
  batch_size. Run at `batch_size=1` (the config default).
- lmms-eval v0.5 may nest `results.json` in a timestamped subdir of
  `--output_path`; `run.sh` copies it up to `results/<run>/<task>/results.json`.
- If `--tasks pope` errors, find the exact task id with
  `python -m vtp_eval.eval.run_lmms --tasks list`.
- The sweep is **resumable** — each `results/<run>/<task>/results.json` that exists is
  skipped, so a re-run after a crash/timeout continues where it stopped. Recommended
  flow: `run_all.sh 20` (smoke whole matrix) → inspect `summary.csv` → `run_all.sh`
  (full, overnight).

## Cross-references
- `docs/vast_proposed_method.md` — the env this installs into.
- `vtp_eval/eval/` — harness source; `vtp_eval/adapters/` — lmms-eval adapters.

## Visualizing token pruning (insight/prune_viz)
`python -m vtp_eval.insight.prune_viz --list-samples` fetches a few samples per
benchmark (streaming, partial — set counts in `configs/prune_viz.yaml`), then:
- Exp 2: `--dataset gqa --index 2 --mode exp2 --r1 64` → original | attention-only | diversity-only.
- Exp 3: `--dataset textvqa --index 0 --mode exp3 --dominant-k 54 --diversity-m 10 --r2 37` → original | R1 | R2.
PNGs land in `outputs/prune_viz/`. Viz uses its own `output_attentions` forward +
the pure `selection.py` helpers — it does not touch the eval/inference path.
