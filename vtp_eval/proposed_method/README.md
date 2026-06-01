# Proposed pruning method (thesis contribution)

Training-free, two-stage visual-token pruning for LLaVA-1.5-7B.

## Files
- `config.py` — `ProposedConfig` (knobs + validation + avg-tokens metric).
- `selection.py` — pure tensor math (dominant top-k, diversity FPS, text→vision
  scoring, keep-index, KV-cache slice). Unit-tested in `tests/`.
- `stage1_vision.py` — `CLIPVisionTower.forward` replacement (Stage 1).
- `stage2_llm.py` — `LlamaModel.forward` prune + span recorder (Stage 2). The
  `_llama_forward_437` body is synced on the Vast.ai transformers-4.37.2 image
  (see its docstring).
- `patch.py` — `proposed_prune(model, cfg)` runtime patcher.
- `adapter.py` — `llava_proposed` lmms-eval adapter (eval environment only).

## Usage
```python
from vtp_eval.proposed_method import proposed_prune, ProposedConfig
cfg = ProposedConfig(dominant_k=54, diversity_m=10, pruned_layer=12, llm_keep_r2=37)
model = proposed_prune(model, cfg)   # model loaded with attn_implementation="eager"
```

## Not training-based
Unlike LearnPruner's LPM, this method has no learnable parameters — there is no
`model.py` / `train.py`. Tune behaviour entirely through the four config knobs.

## Related work
LearnPruner (middle-layer text-attention), VisionZip (CLS dominant + merge),
DivPrune (diversity), FastV (middle-layer prune). Archived adapters:
`git show archive/lmms-eval-pre-cleanup:vtp_eval/adapters/`.
