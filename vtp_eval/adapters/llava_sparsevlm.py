"""SparseVLMs adapter — text-guided multi-layer pruning at layers 2/6/15.

Reference: SparseVLMs (https://github.com/Gumpest/SparseVLMs).
Pruning is controlled by env var RETAIN_TOKN read on import inside
SparseVLMs/llava/model/language_model/score.py. We set it BEFORE super()
init triggers the LLaVA import chain.
"""
import os

from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_sparsevlm")
class LlavaSparseVLM(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        retain_token: int = 192,
        **kw,
    ):
        if int(retain_token) not in {64, 96, 128, 192}:
            raise ValueError(
                f"retain_token must be one of 64/96/128/192 (per SparseVLMs "
                f"sparse_token_dict), got {retain_token}"
            )
        os.environ["RETAIN_TOKN"] = str(int(retain_token))
        super().__init__(pretrained=pretrained, **kw)
        self.pruning_meta = {"method": "sparsevlm", "retain": int(retain_token)}
