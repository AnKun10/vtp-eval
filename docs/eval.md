# Benchmarking the proposed method with lmms-eval

Harness to benchmark the proposed pruning method vs a vanilla LLaVA-1.5-7B
baseline. Accuracy via lmms-eval; efficiency via a 3-stage latency breakdown
(vision encoder / LLM prefill / LLM decode), total latency, peak memory, token
budget, and a theoretical-TFLOPs estimate. Design:
`docs/superpowers/specs/2026-06-02-lmms-eval-benchmark-harness-design.md`.

## What it benchmarks
`configs/eval.yaml` runs: `baseline` (vanilla) and `proposed_stage1_R1-64`
(Stage-1 vision prune only, 576→64). The full Stage-2 variants
(`proposed_R1-64_R2-37_k12`, `proposed_keeppos`) are present but **commented out**:
Stage 2's decode path re-feeds tokens (pruned KV cache vs `generate`'s token
bookkeeping), which corrupts decode latency; re-enable after that is reworked to
no-slice + full-rotary. Add runs/benchmarks by editing the `runs` / `tasks`
lists. Other methods (VisionZip/FastV/…) are compared via their published
numbers (out of scope here).

## Run on Vast.ai
1. Rent with the proposed-method template (py310/torch-2.1.2 image + the §2
   on-start from `docs/vast_proposed_method.md`). The on-start now also installs
   lmms-eval v0.5 (via `install/proposed.sh`). Disk **≥60 GB** (POPE COCO images
   + HF datasets cache).
2. SSH in. The venv auto-activates only in an interactive shell — for scripts
   run `source /venv/main/bin/activate` first. Then:
   `export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1`.
   `HF_HUB_DISABLE_XET=1` is important: recent `huggingface_hub` defaults to the
   xet backend, which on Vast stalled at 0 B/s ("connection struggling"); plain
   HTTPS + `hf_transfer` downloads the 13.5 GB model at ~12 MB/s. (POPE's task
   yaml `token: True` is flipped to `False` by `install/proposed.sh` so the
   public dataset loads without an HF login.)
3. Smoke (limit 20): `bash scripts/eval/run.sh baseline pope 20`
   → check `results/baseline/{results.json,timing.json}`.
4. Full sweep: `bash scripts/eval/run_all.sh` → `results/summary.csv`
   (baseline + proposed variants on POPE).
5. Retrieve: `scp -P <PORT> root@<HOST>:/workspace/vtp-eval/results/summary.csv .`
6. **Stop** (not Destroy) to keep the model + dataset cache.

## summary.csv columns
`method, task, metric, value, avg_tokens, keep_ratio_pct, encoder_ms,
prefill_ms, decode_ms, total_ms, peak_mem_mb, tflops`. The measured stage
latencies are the primary efficiency numbers; `tflops` is an approximate
theoretical estimate from `avg_tokens`.

## Notes
- POPE ≈ 9k samples; at batch=1 a full pass per run is the slow part — use
  `--limit` (3rd arg to `run.sh`) for a quick sanity number first.
- Timing drops the first batch as warm-up; per-sample latency = batch time /
  batch_size. Run at `batch_size=1` (the config default).
- lmms-eval v0.5 may nest `results.json` in a timestamped subdir of
  `--output_path`; `run.sh` copies it up to `results/<run>/results.json`.
- If `--tasks pope` errors, find the exact task id with
  `python -m vtp_eval.eval.run_lmms --tasks list`.

## Cross-references
- `docs/vast_proposed_method.md` — the env this installs into.
- `vtp_eval/eval/` — harness source; `vtp_eval/adapters/` — lmms-eval adapters.
