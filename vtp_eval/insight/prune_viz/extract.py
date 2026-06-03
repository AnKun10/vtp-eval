"""Viz-only forward + R1/R2 keep-index extraction.

Reuses the PURE proposed_method.selection helpers and runs its own
output_attentions forward. It never calls proposed_prune and adds no hooks to the
production inference path. GPU + original liuhaotian LLaVA only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vtp_eval.proposed_method.selection import (cls_topk_dominant,
                                                farthest_point_diversity,
                                                select_stage1,
                                                text_to_vision_scores)
from vtp_eval.proposed_method import stage1_vision


def load_model(model_path: str = "liuhaotian/llava-v1.5-7b"):
    """Load the original LLaVA (eager attention) + processors. Heavy import
    deferred so --list-samples never imports llava/torch."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    tok, model, image_processor, _ = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path),
        attn_implementation="eager", device_map="cuda")
    return tok, model, image_processor


@torch.no_grad()
def penultimate_vision(model, image_tensor):
    """Run the CLIP vision tower; return (attn_penult [1,H,577,577],
    hidden_penult [1,577,D]). image_tensor: [1,3,336,336] on model dtype/device."""
    vt = model.get_model().get_vision_tower()
    fo = vt.vision_tower(image_tensor.to(device=vt.device, dtype=vt.dtype),
                         output_hidden_states=True, output_attentions=True)
    return fo.attentions[-2], fo.hidden_states[-2]


def r1_selections(attn_penult, hidden_penult, r1: int, dominant_k: int,
                  diversity_m: int):
    """Return dict of [r1] patch-index tensors for the three R1 strategies."""
    cls_attn = attn_penult[:, :, 0, 1:].sum(dim=1)              # [1, 576]
    attention_only = cls_topk_dominant(cls_attn, r1)[0]        # [r1]

    feats = F.normalize(hidden_penult[:, 1:, :], dim=-1)        # [1, 576, D]
    seed = cls_topk_dominant(cls_attn, 1)                       # [1, 1] top-attn anchor
    div_rest = farthest_point_diversity(feats, seed, r1 - 1)[0]  # [r1-1]
    diversity_only = torch.cat([seed[0], div_rest])            # [r1]

    combined = select_stage1(attn_penult, hidden_penult, dominant_k, diversity_m)[0]
    return {"attention": attention_only.sort().values,
            "diversity": diversity_only.sort().values,
            "combined": combined}


@torch.no_grad()
def r2_selection(tok, model, image_tensor, question, cfg, r1_combined_patches):
    """Run a SINGLE viz-only forward with the vision tower emitting R1 features,
    capture layer-12 text->vision attention, return the R2 keep set mapped back
    to original patch indices ([r2]).

    cfg: ProposedConfig (dominant_k/diversity_m/pruned_layer/llm_keep_r2).
    r1_combined_patches: [r1] original patch indices kept by Stage 1 (sorted).
    Restores the vision tower forward afterwards (no lasting hook).
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    vt = model.get_model().get_vision_tower()
    orig_fwd = vt.forward
    vt.forward = stage1_vision.make_forward(cfg).__get__(vt, type(vt))  # Stage-1 -> R1
    try:
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to(model.device)
        out = model(ids, images=image_tensor.to(model.device, dtype=vt.dtype),
                    image_sizes=[(336, 336)], output_attentions=True, use_cache=False)
        attn_k = out.attentions[cfg.pruned_layer]                  # [1,H,S,S]
    finally:
        vt.forward = orig_fwd

    S = attn_k.shape[-1]
    nv = cfg.r1
    vstart = int((ids[0] == IMAGE_TOKEN_INDEX).float().argmax())   # image-token pos
    instr_lo, instr_hi = vstart + nv, S
    scores = text_to_vision_scores(attn_k, (vstart, vstart + nv), (instr_lo, instr_hi))
    top_local = scores.topk(cfg.llm_keep_r2, dim=1).indices[0]     # [r2] indices into R1
    return r1_combined_patches[top_local.cpu()].sort().values     # -> original patches
