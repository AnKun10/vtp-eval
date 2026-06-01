# vtp_eval/proposed_method/patch.py
"""proposed_prune(model, cfg): apply both pruning stages to a loaded LLaVA model.

Mirrors VisionZip's visionzip(model, ...) entry point. Swaps module forwards at
runtime -- no training, no global file patching.
"""
from __future__ import annotations

from . import stage1_vision, stage2_llm


def proposed_prune(model, cfg):
    """Patch a loaded original-LLaVA model in place; return it.

    Stage 1: CLIPVisionTower.forward -> dominant + diversity selection.
    Stage 2: LlamaModel.forward -> text->vision prune at layer cfg.pruned_layer,
             plus a span recorder on prepare_inputs_labels_for_multimodal.
    """
    # --- Stage 1: vision tower (llava import deferred to call time) ---
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = stage1_vision.make_forward(cfg)

    # --- Stage 2: guard eager attention, install span recorder + forward ---
    attn_impl = getattr(model.config, "_attn_implementation", None)
    if attn_impl not in (None, "eager"):
        raise ValueError(
            "proposed method Stage 2 requires attn_implementation='eager' "
            f"(got {attn_impl!r}); attention weights are needed at the pruned layer.")

    stage2_llm.install_span_recorder(model, cfg)

    from transformers.models.llama.modeling_llama import LlamaModel
    LlamaModel.forward = stage2_llm.make_llama_forward(cfg)

    model._proposed_cfg = cfg
    # _proposed_last_prune is written by the Stage-2 forward on the INNER model
    # (its `self`); initialize it there so readers find the attribute.
    getattr(model, "model", model)._proposed_last_prune = None
    return model
