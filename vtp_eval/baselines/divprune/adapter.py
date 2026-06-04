# vtp_eval/baselines/divprune/adapter.py
"""lmms-eval adapter for DivPrune (CVPR 2025). Registered as "llava_divprune"."""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase
from vtp_eval.baselines.budgets import divprune_ratio
from vtp_eval.baselines.divprune.patch import divprune_prune


@register_model("llava_divprune")
class LlavaDivPrune(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 avg_tokens: int = 64, **kw):
        super().__init__(pretrained=pretrained, **kw)
        avg = int(avg_tokens)
        keep = round(divprune_ratio(avg) * 576)
        self._model = divprune_prune(self._model, keep)
        self.pruning_meta = {"method": "divprune", "avg_tokens": keep,
                             "keep": keep, "ratio": round(divprune_ratio(avg), 4)}
