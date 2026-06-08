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

## Manual smoke checklist (unified 3-tab app)
- [ ] Pick benchmark(s) + images/benchmark → **Load** → thumbnail gallery appears.
- [ ] Click a thumbnail → "Selected image" fills; its benchmark questions list
      below. Click a question → it fills the Inference input and the TVA target
      words; all three tabs now share it.
- [ ] **Pruning viz** tab: pick budget (32/64/128) + diversity %, **Run** →
      exp2 (Attention | Diversity) and exp3 (After R1 | After R2) overlays appear.
- [ ] **Text-visual attention** tab: tick a target word (e.g. the object asked
      about), **Run** → a 3-layer (shallow/middle/deep) attention heatmap +
      a concentration bar chart appear; deeper layers are more focused.
- [ ] **Inference** tab: **Send** → one bubble with Proposed (128 tok) vs Vanilla
      (576 tok) answers + latencies; a same-image follow-up shows `CACHE HIT`.
- [ ] Click a different thumbnail → all tabs reset (chat, figures, words cleared).

## Automated slow test
`pytest -m slow tests/test_demo_engine_smoke.py -v`  (Vast.ai only)
