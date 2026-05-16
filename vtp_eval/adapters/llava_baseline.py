"""Baseline LLaVA-1.5 adapter — no pruning. Reference point for comparison."""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_baseline")
class LlavaBaseline(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b", **kw):
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "baseline", "keep_ratio": 1.0}
