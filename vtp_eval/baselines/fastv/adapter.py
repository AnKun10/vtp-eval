# vtp_eval/baselines/fastv/adapter.py
"""lmms-eval adapter for FastV (ECCV 2024). Registered as "llava_fastv"."""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase
from vtp_eval.baselines.budgets import fastv_knobs
from vtp_eval.baselines.fastv.patch import FastVConfig, fastv_prune


@register_model("llava_fastv")
class LlavaFastV(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 avg_tokens: int = 64, agg_layer: int = 2, **kw):
        super().__init__(pretrained=pretrained, **kw)
        k, r = fastv_knobs(int(avg_tokens), k=int(agg_layer))   # raises below floor
        # FastV agg_layer K = K full layers; our forward prunes AFTER layer index
        # `pruned_layer`, giving pruned_layer+1 full layers -> pruned_layer = K-1.
        cfg = FastVConfig(pruned_layer=k - 1, llm_keep_r2=r)
        n_layers = self._model.config.num_hidden_layers
        self._model = fastv_prune(self._model, cfg)
        self.pruning_meta = {"method": "fastv", "K": k, "R": r,
                             "avg_tokens": round(cfg.avg_tokens(n_layers), 1)}
