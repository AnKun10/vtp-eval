# vtp_eval/proposed_method/selection.py
"""Pure tensor helpers for the proposed two-stage pruning method.

No model dependencies — every function takes/returns plain tensors so it can
be unit-tested on CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cls_topk_dominant(cls_attn_sum: torch.Tensor, dominant_k: int) -> torch.Tensor:
    """cls_attn_sum: [B, P] head-summed CLS->patch attention.
    Returns [B, dominant_k] patch indices of the most-attended tokens."""
    return cls_attn_sum.topk(dominant_k, dim=1).indices
