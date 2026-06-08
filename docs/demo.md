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
- [ ] Upload an image, ask Q1 → answer appears; metrics show `cache miss`,
      tokens `576 → 384 → 128`, and a latency value.
- [ ] Ask Q2 about the same image → metrics show `CACHE HIT` and a latency
      lower than the `No-cache (same turn)` figure beside it.
- [ ] R1 overlay (384) is identical across Q1/Q2; R2 overlay (128) changes
      with the question (query-aware).
- [ ] Uncheck "Enable retain-token cache", ask Q3 → `cache miss` again.
- [ ] Upload a different image → conversation resets; first question is a miss.

## Automated slow test
`pytest -m slow tests/test_demo_engine_smoke.py -v`  (Vast.ai only)
