"""DivPrune adapter — diversity-based pruning at projector (pre-LLM).

Reference: DivPrune (https://github.com/vbdi/divprune).
Pruning is controlled by env vars BASELINE / LAYER_INDEX / SUBSET_RATIO read in
divprune/LLaVA/llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal.
"""
import os

from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_divprune")
class LlavaDivPrune(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        subset_ratio: float = 0.098,
        layer_index: int = 0,
        **kw,
    ):
        os.environ["BASELINE"] = "OURS"
        os.environ["LAYER_INDEX"] = str(int(layer_index))
        os.environ["SUBSET_RATIO"] = str(float(subset_ratio))
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {
            "method": "divprune",
            "ratio": float(subset_ratio),
            "layer_index": int(layer_index),
        }
