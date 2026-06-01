# vtp_eval/proposed_method/adapter.py
"""lmms-eval adapter for the proposed pruning method.

Registered as "llava_proposed". Applies proposed_prune after load; exposes
pruning_meta for the timing/results sidecar (same contract as the archived
LlavaVisionZip adapter).

NOTE: ``LlavaPruningBase`` and the lmms-eval harness live on branch
``archive/lmms-eval-pre-cleanup`` and are restored under ``vtp_eval/eval/`` as a
separate concern. This module is import-able only in that eval environment
(lmms_eval + vtp_eval.adapters._base present); it is intentionally not part of
the locally-testable surface.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase  # restored from archive branch
from vtp_eval.proposed_method import ProposedConfig, proposed_prune


@register_model("llava_proposed")
class LlavaProposed(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 dominant_k: int = 54, diversity_m: int = 10,
                 pruned_layer: int = 12, llm_keep_r2: int = 37, **kw):
        super().__init__(pretrained=pretrained, **kw)
        cfg = ProposedConfig(int(dominant_k), int(diversity_m),
                             int(pruned_layer), int(llm_keep_r2))
        n_layers = self._model.config.num_hidden_layers
        cfg.validate(num_llm_layers=n_layers)
        self._model = proposed_prune(self._model, cfg)
        self.pruning_meta = {
            "method": "proposed",
            **cfg.as_dict(),
            "R1": cfg.r1,
            "avg_tokens": cfg.avg_tokens(n_layers),
        }
