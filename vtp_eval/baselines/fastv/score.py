# vtp_eval/baselines/fastv/score.py
"""FastV (ECCV 2024) ranking: importance = the LAST token's attention to each
vision token, averaged over heads. Mirrors fastv_forward.py's
last_layer_attention_avg_last_tok_image. instr_slice is accepted (for a uniform
score_fn signature) but ignored — FastV is text-agnostic.
"""
from __future__ import annotations

import torch


def last_token_to_vision_scores(attn_layer: torch.Tensor,
                                vision_slice: tuple[int, int],
                                instr_slice: tuple[int, int]) -> torch.Tensor:
    """attn_layer: [B, H, S, S]. Returns [B, Nv] last-row attn over vision cols."""
    vs, ve = vision_slice
    return attn_layer[:, :, -1, vs:ve].mean(dim=1)   # [B, Nv]
