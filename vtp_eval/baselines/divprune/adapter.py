# vtp_eval/baselines/divprune/adapter.py
"""lmms-eval adapter for DivPrune (CVPR 2025). Registered as "llava_divprune".

Retain-token convention: keep N = retain_tokens visual tokens (constant across
all layers), so avg_tokens == retain_tokens.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase
from vtp_eval.baselines.divprune.patch import divprune_prune


@register_model("llava_divprune")
class LlavaDivPrune(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 retain_tokens: int = 64, **kw):
        super().__init__(pretrained=pretrained, **kw)
        keep = int(retain_tokens)
        self._model = divprune_prune(self._model, keep)
        self.pruning_meta = {"method": "divprune", "retain_tokens": keep,
                             "avg_tokens": keep, "harness": "ours",
                             "ratio": round(keep / 576, 4)}
