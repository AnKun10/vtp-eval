# vtp_eval/baselines/fastv/adapter.py
"""lmms-eval adapter for FastV (ECCV 2024). Registered as "llava_fastv".

Retain-token convention: FastV keeps R = retain_tokens image tokens after K full
layers (paper default K=2). Our forward prunes AFTER layer index `pruned_layer`,
giving pruned_layer+1 full layers, so pruned_layer = K-1. The TRUE avg-token
(layers 0..K-1 at 576, the rest at R) drives TFLOPs and comes from
FastVConfig.avg_tokens.
"""
from lmms_eval.api.registry import register_model

from vtp_eval.adapters._base import LlavaPruningBase
from vtp_eval.baselines.fastv.patch import FastVConfig, fastv_prune


@register_model("llava_fastv")
class LlavaFastV(LlavaPruningBase):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 retain_tokens: int = 64, agg_layer: int = 2, **kw):
        super().__init__(pretrained=pretrained, **kw)
        R, K = int(retain_tokens), int(agg_layer)
        cfg = FastVConfig(pruned_layer=K - 1, llm_keep_r2=R)
        n_layers = self._model.config.num_hidden_layers
        self._model = fastv_prune(self._model, cfg)
        self.pruning_meta = {"method": "fastv", "retain_tokens": R, "K": K, "R": R,
                             "avg_tokens": round(cfg.avg_tokens(n_layers), 1),
                             "harness": "ours"}
