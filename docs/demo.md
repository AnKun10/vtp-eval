# Demo: LLaVA pruning + retain-token cache

Gradio multi-turn QA on one image with the proposed two-stage pruning method.

## Run on Vast.ai
1. Provision a GPU instance (≥24 GB) and run the environment install:
   `bash install/proposed.sh`  (installs LLaVA + transformers 4.37.2 + gradio)
2. Launch the demo:
   `python -m vtp_eval.demo.app --share`
3. Open the printed Gradio URL (use `--share` for a public link, or forward
   port 7860).

> **Single-user demo.** The retain-token cache and the index recorders are
> process-global state, so use one browser session at a time. The default
> request queue serializes turns; do not raise concurrency.

Config is fixed: R1=384 (div 50%, dominant_k=192 + diversity_m=192), R2=128 at
LLM layer 12. Fallback if the Stage-2 decode path misbehaves: add `--no-stage2`
(R1-only; the cache still works).

## Manual smoke checklist
- [ ] Pick one or more benchmarks (pope/textvqa/mme/scienceqa/vizwiz/ocrbench),
      set images/benchmark, click **Load** → a thumbnail gallery appears.
- [ ] Click a thumbnail → "Selected image" fills, both conversations reset, and
      the benchmark questions for that image are listed below the chat.
- [ ] Click a listed question (or type your own) → **Send**. The assistant bubble
      shows BOTH a **Proposed** answer (latency · 128 tok · cache miss) and a
      **Vanilla** answer (latency · 576 tok), the proposed no-cache latency, and
      the speedup. Vanilla should be slower than proposed.
- [ ] Ask a second question about the same image → the Proposed line shows
      `CACHE HIT` and a lower latency; vanilla has no cache and stays slow.
- [ ] R1 overlay (384) is identical across turns; R2 overlay (128) changes with
      the question.
- [ ] Click a different thumbnail → both conversations reset; first turn is a miss.

## Automated slow test
`pytest -m slow tests/test_demo_engine_smoke.py -v`  (Vast.ai only)
