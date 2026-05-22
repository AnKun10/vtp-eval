# Proposed pruning method — design notes

Placeholder. When the thesis method is designed, document:

- The insight being exploited (e.g. learnable importance vs CLS attention).
- The architecture of the pruning module.
- Where it integrates in the LLaVA forward pass (after vision encoder?
  inside LLM block?).
- Training procedure + dataset (subset of LLaVA-665K? full?).
- Evaluation protocol (link to `eval.md`).

## Cross-references

- `vtp_eval/proposed_method/` — source code (empty for now).
- `docs/text_visual_attention.md` — the interpretability insight that
  motivates this method.
