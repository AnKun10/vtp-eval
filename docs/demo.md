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
      set images/benchmark, click **Load** → a thumbnail gallery appears; status
      shows how many images loaded (and any failed benchmarks).
- [ ] Click a thumbnail → "Selected image" fills, the conversation resets, and
      the benchmark questions for that image are listed below the chat.
- [ ] Click one of the listed questions → it drops into the input box; or type
      your own. Press **Send** → answer appears; metrics show `cache miss`,
      tokens `576 → 384 → 128`, and a latency value.
- [ ] Ask a second question about the same image → metrics show `CACHE HIT` with
      a latency below the `No-cache (same turn)` figure.
- [ ] R1 overlay (384) is identical across turns on one image; R2 overlay (128)
      changes with the question (query-aware).
- [ ] Click a different thumbnail → conversation resets; first question is a miss.

## Automated slow test
`pytest -m slow tests/test_demo_engine_smoke.py -v`  (Vast.ai only)
