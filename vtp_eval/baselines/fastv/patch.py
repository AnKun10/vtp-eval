# vtp_eval/baselines/fastv/patch.py
"""Apply FastV (prune@layer K by last-token attention) to a loaded LLaVA model,
reusing the GPU-validated no-slice Stage-2 forward from proposed_method.stage2_llm.

FastV has NO Stage-1: the vision tower stays unpatched (full 576 tokens), so the
span recorder is installed with r1=576. Only LlamaModel.forward is swapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from vtp_eval.proposed_method import stage2_llm
from vtp_eval.baselines.fastv.score import last_token_to_vision_scores

NUM_PATCHES = 576


@dataclass
class FastVConfig:
    """Duck-typed to what stage2_llm reads: r1, pruned_layer, llm_keep_r2,
    score_fn, stage2_enabled. NOT a ProposedConfig (no Stage-1 fields)."""
    pruned_layer: int = 2
    llm_keep_r2: int = 30
    stage2_enabled: bool = True
    score_fn: Callable = field(default=last_token_to_vision_scores)

    @property
    def r1(self) -> int:
        return NUM_PATCHES   # no Stage-1 vision prune

    def avg_tokens(self, num_layers: int) -> float:
        # Mirrors ProposedConfig.avg_tokens (enabled branch): layers 0..k run at
        # r1 (=576 here), layers k+1..L-1 run at llm_keep_r2. Keep the two in sync.
        k = self.pruned_layer
        return ((k + 1) * self.r1 + (num_layers - k - 1) * self.llm_keep_r2) / num_layers


def fastv_prune(model, cfg: FastVConfig):
    """Patch a loaded LLaVA model in place for FastV; return it."""
    attn_impl = getattr(model.config, "_attn_implementation", None)
    if attn_impl not in (None, "eager"):
        raise ValueError("FastV needs attn_implementation='eager' "
                         f"(got {attn_impl!r}); layer-K attention is required.")
    stage2_llm.install_span_recorder(model, cfg)            # uses cfg.r1 == 576
    stage2_llm.install_full_rotary(model.config.max_position_embeddings)
    from transformers.models.llama.modeling_llama import LlamaModel
    LlamaModel.forward = stage2_llm.make_llama_forward(cfg)
    model._proposed_cfg = cfg
    getattr(model, "model", model)._proposed_last_prune = None
    return model
