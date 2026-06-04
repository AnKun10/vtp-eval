# vtp_eval/baselines/visionzip/adapter.py
"""lmms-eval adapter for VisionZip (CVPR 2025). Registered as "llava_visionzip".

Imports the official visionzip() entry point (repo cloned alongside vtp-eval and
added to sys.path by install/baselines.sh). visionzip()'s apply_info keeps
dominant-1 dominant patches + contextual merged tokens, so we pass dominant+1 to
realize exactly `dominant + contextual == avg_tokens` kept tokens.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase
from vtp_eval.baselines.budgets import visionzip_knobs


@register_model("llava_visionzip")
class LlavaVisionZip(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 avg_tokens: int = 64, **kw):
        super().__init__(pretrained=pretrained, **kw)
        from visionzip import visionzip          # official; deferred import
        avg = int(avg_tokens)
        dominant, contextual = visionzip_knobs(avg)
        # apply_info uses dominant-1 internally -> pass dominant+1 to realize `avg`.
        self._model = visionzip(self._model, dominant=dominant + 1, contextual=contextual)
        self.pruning_meta = {"method": "visionzip", "avg_tokens": avg,
                             "dominant": dominant, "contextual": contextual}
