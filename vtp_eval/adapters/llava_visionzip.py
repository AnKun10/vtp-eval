"""VisionZip adapter — pre-LLM pruning via CLIP CLS-attention dominant +
contextual stride sampling.

Reference: VisionZip (https://github.com/dvlab-research/VisionZip).
Pruning is applied by calling visionzip(model, dominant, contextual) after
load — replaces several module forwards at runtime.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase


@register_model("llava_visionzip")
class LlavaVisionZip(LlavaPruningBase):
    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        dominant: int = 54,
        contextual: int = 10,
        **kw,
    ):
        super().__init__(pretrained=pretrained, **kw)
        from visionzip import visionzip  # import deferred — only on Colab
        self._model = visionzip(
            self._model,
            dominant=int(dominant),
            contextual=int(contextual),
        )
        self.pruning_meta = {
            "method": "visionzip",
            "dominant": int(dominant),
            "contextual": int(contextual),
        }
