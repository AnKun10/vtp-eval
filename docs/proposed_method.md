# Proposed pruning method — design notes

Training-free, two-stage visual-token pruning for LLaVA-1.5-7B. Full design:
`docs/superpowers/specs/2026-06-01-proposed-pruning-method-design.md`.
Implementation plan: `docs/superpowers/plans/2026-06-01-proposed-pruning-method.md`.

## Insight exploited
Text→vision attention resists the LLM "attention shift" bias and focuses on
query-relevant regions in middle layers (reproduced in
`vtp_eval/insight/text_visual_attention/`). CLS attention in the vision encoder
suffers attention-sink, so we complement the dominant CLS tokens with a
diversity-based context set rather than relying on CLS alone.

## Two stages
1. **Vision tower (Stage 1):** keep `dominant_k` patches by penultimate-layer
   CLS→patch attention, then add `diversity_m` farthest-point context tokens
   seeded by the dominant set (cosine on penultimate hidden states). Output
   R1 = dominant_k + diversity_m tokens. Code: `proposed_method/stage1_vision.py`.
2. **LLM middle layer (Stage 2):** at layer `pruned_layer`, score each vision
   token by mean instruction→vision attention, keep top `llm_keep_r2`, drop the
   rest (sequence + position ids + KV cache). Code: `proposed_method/stage2_llm.py`.

## Integration
`proposed_method/patch.py:proposed_prune(model, cfg)` swaps the two forwards at
runtime (no training). lmms-eval adapter: `llava_proposed`
(`proposed_method/adapter.py`). Config: `configs/proposed_method.yaml`.

The pure selection math lives in `proposed_method/selection.py` and is unit-tested
(`tests/test_proposed_selection.py`). The `LlamaModel.forward` port for Stage 2 is
version-pinned to transformers 4.37.2 and is synced/verified on the Vast.ai image
(`stage2_llm._llama_forward_437`); the end-to-end check is `tests/test_proposed_smoke.py`
(`@pytest.mark.slow`).

## Backbone & environment
Original `liuhaotian/llava-v1.5-7b` + transformers 4.37.2 (same env as the
VisionZip adapter). Requires `attn_implementation=eager`. Install:
`install/proposed.sh`.

## avg-tokens reporting convention
`ProposedConfig.avg_tokens` counts layers `0..k` at R1 and `k+1..L-1` at R2
(FastV convention, because layer `k`'s attention is read before the drop). This
differs by one layer from LearnPruner's reported 12/20 split — keep it consistent
in result tables.

## Cross-references
- `docs/text_visual_attention.md` — the motivating insight.
- `docs/eval.md` — benchmarking (restored from archive branch).
