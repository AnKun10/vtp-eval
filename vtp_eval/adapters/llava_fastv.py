"""FastV adapter — token pruning AFTER layer K based on attention rank.

Reference: FastV (https://github.com/pkunlp-icler/FastV).
Requires the FastV-patched modeling_llama.py in the active transformers install,
applied via install/_patch_fastv.py. With patch, the LlamaModel reads these
fields on model.config and runs the pruning forward path.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_fastv")
class LlavaFastV(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        fast_v_agg_layer: int = 2,
        fast_v_attention_rank: int = 128,
        fast_v_image_token_length: int = 576,
        **kw,
    ):
        super().__init__(pretrained=pretrained, **kw)
        self._model.config.use_fast_v = True
        self._model.config.fast_v_agg_layer = int(fast_v_agg_layer)
        self._model.config.fast_v_attention_rank = int(fast_v_attention_rank)
        self._model.config.fast_v_sys_length = None
        self._model.config.fast_v_image_token_length = int(fast_v_image_token_length)
        # Reset internal state; method comes from the FastV patch
        if hasattr(self._model.model, "reset_fastv"):
            self._model.model.reset_fastv()
        self.pruning_meta = {
            "method": "fastv",
            "K": int(fast_v_agg_layer),
            "R": int(fast_v_attention_rank),
        }
