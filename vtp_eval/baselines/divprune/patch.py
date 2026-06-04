# vtp_eval/baselines/divprune/patch.py
"""Apply DivPrune to a loaded LLaVA model: prune visual tokens in the merged
embedding sequence (post-projector, before the LLM), mirroring the reference's
LAYER_INDEX=0 path. Vision tower + LlamaModel.forward are left untouched.
"""
from __future__ import annotations

import torch

from .select import divprune_select

SYS_TOKEN_LEN = 35   # LLaVA-1.5 system-prompt length (reference constant)


def divprune_prune(model, keep: int):
    """Patch prepare_inputs_labels_for_multimodal to keep `keep` visual tokens."""
    orig = model.prepare_inputs_labels_for_multimodal
    img_token_index = getattr(model.config, "image_token_index", -200)

    def wrapped(input_ids, position_ids, attention_mask, past_key_values,
                labels, images, image_sizes=None, *args, **kw):
        out = orig(input_ids, position_ids, attention_mask, past_key_values,
                   labels, images, image_sizes, *args, **kw)
        _, position_ids2, attention_mask2, past2, inputs_embeds, labels2 = out
        # Only prune during prefill (a real image batch produces inputs_embeds).
        if inputs_embeds is None or input_ids is None \
                or not (input_ids == img_token_index).any():
            return out
        # image tokens = post-merge length minus pre-merge text (1 placeholder).
        img_len = inputs_embeds.shape[1] - (input_ids.shape[1] - 1)
        if img_len <= keep:
            return out
        vstart = SYS_TOKEN_LEN
        visual = inputs_embeds[0, vstart:vstart + img_len]        # [img_len, D]
        sel = divprune_select(visual, keep) + vstart
        keep_idx = torch.cat([
            torch.arange(vstart, device=inputs_embeds.device),
            sel,
            torch.arange(vstart + img_len, inputs_embeds.shape[1], device=inputs_embeds.device),
        ]).sort().values
        inputs_embeds = inputs_embeds[:, keep_idx]
        if position_ids2 is not None:
            position_ids2 = position_ids2[:, keep_idx] if position_ids2.dim() == 2 \
                else position_ids2[:, keep_idx, :]
        if attention_mask2 is not None:
            attention_mask2 = attention_mask2[:, keep_idx]
        if labels2 is not None:
            labels2 = labels2[:, keep_idx]
        return None, position_ids2, attention_mask2, past2, inputs_embeds, labels2

    model.prepare_inputs_labels_for_multimodal = wrapped
    return model
