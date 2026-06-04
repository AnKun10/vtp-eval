# vtp_eval/baselines/visionzip/adapter.py
"""lmms-eval adapter for VisionZip (CVPR 2025). Registered as "llava_visionzip".

Imports the official visionzip() entry point (repo cloned alongside vtp-eval and
added to sys.path by install/baselines.sh).

Verified realized-token formula (derived from clip_encoder.py):
  visionzip(model, dominant=D_arg, contextual=C_arg) realizes exactly D_arg + C_arg
  visual tokens fed to the LLM.

How clip_encoder.py produces that count (VisionZip repo, clip_encoder.py):
  - Line 53: topk selects `_info["dominant"]` (== D_arg - 1) patch tokens by CLS
    attention.
  - Line 54: CLS token index (0) is unconditionally prepended via torch.cat, so
    all_indices has length D_arg.
  - Line 57: dominant_tokens has shape (..., dominant_num + 1, ...) == (..., D_arg, ...).
  - Line 80: contextual_tokens has exactly `_info["contextual"]` == C_arg tokens
    (aggregate-merge of remaining patches into C_arg cluster centres).
  - Line 83: final output = torch.cat([dominant_tokens, contextual_tokens], dim=1)
    => D_arg + C_arg tokens total.

  apply_info (utils.py line 7 / main.py line 7) stores dominant_num = D_arg - 1,
  so the internal dominant count is D_arg - 1 patches + 1 CLS = D_arg, and
  C_arg contextual tokens, giving D_arg + C_arg total.

visionzip_knobs(avg) returns (dominant, contextual) with dominant + contextual == avg,
so calling visionzip(model, dominant=dominant, contextual=contextual) yields exactly
avg tokens.  Example: avg=64 -> dominant=54, contextual=10 -> realized = 54 + 10 = 64.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase
from vtp_eval.baselines.budgets import visionzip_knobs


@register_model("llava_visionzip")
class LlavaVisionZip(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 retain_tokens: int = 64, **kw):
        super().__init__(pretrained=pretrained, **kw)
        from visionzip import visionzip          # official; deferred import
        n = int(retain_tokens)
        dominant, contextual = visionzip_knobs(n)
        # realized = dominant + contextual == n (clip_encoder.py lines 54, 57, 83:
        # CLS prepend means dominant_tokens count = dominant, not dominant-1).
        self._model = visionzip(self._model, dominant=dominant, contextual=contextual)
        self.pruning_meta = {"method": "visionzip", "retain_tokens": n, "avg_tokens": n,
                             "dominant": dominant, "contextual": contextual,
                             "harness": "ours"}
