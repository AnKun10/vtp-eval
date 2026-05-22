# Proposed pruning method (thesis contribution)

Empty placeholder for the visual-token pruning method being developed
as part of the Đồ Án Tốt Nghiệp.

## When implementing, expect

- `model.py` — the pruning module (e.g. learnable importance predictor,
  inspired by LearnPruner's LPM).
- `train.py` — training loop on a subset of LLaVA-665K.
- `inference.py` — the patched LLaVA forward that invokes the module.
- `configs/proposed_method.yaml` — hyperparameters (already a sibling
  to configs/text_visual_attention.yaml).

## Related work to draw from

- LearnPruner (ICLR 2026) — middle-layer text-attention pruning.
- FastV, VisionZip, SparseVLMs, DivPrune, SparseVILA — see the
  `archive/lmms-eval-pre-cleanup` branch for prior adapter code.

## Cross-references

- `vtp_eval/insight/text_visual_attention/` — the interpretability
  experiment whose result motivates this method.
- `docs/proposed_method.md` — design notes.
