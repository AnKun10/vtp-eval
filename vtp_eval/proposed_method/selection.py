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


def farthest_point_diversity(feats: torch.Tensor, seed_idx: torch.Tensor,
                             m: int) -> torch.Tensor:
    """Greedy max-min (farthest-point) selection of ``m`` new tokens.

    feats:    [B, P, D], assumed L2-normalized (so dot product == cosine).
    seed_idx: [B, k] indices already selected (the dominant set).
    Returns:  [B, m] newly selected indices, disjoint from the seed set and
              from each other, each chosen to minimize its maximum cosine
              similarity to the already-selected set.
    """
    B, P, D = feats.shape
    device = feats.device
    dtype = feats.dtype
    pos_inf = torch.finfo(dtype).max
    neg_inf = torch.finfo(dtype).min

    sim = torch.bmm(feats, feats.transpose(1, 2))  # [B, P, P] cosine
    selected = torch.zeros(B, P, dtype=torch.bool, device=device)
    selected.scatter_(1, seed_idx, True)

    # max_sim[b, j] = max cosine of token j to any selected token
    masked = torch.where(selected.unsqueeze(1), sim, sim.new_full(sim.shape, neg_inf))
    max_sim = masked.max(dim=2).values  # [B, P]

    out = torch.empty(B, m, dtype=torch.long, device=device)
    for i in range(m):
        cand = max_sim.masked_fill(selected, pos_inf)   # never re-pick selected
        nxt = cand.argmin(dim=1)                         # [B] most-dissimilar token
        out[:, i] = nxt
        selected.scatter_(1, nxt.unsqueeze(1), True)
        new_sim = torch.gather(sim, 2, nxt.view(B, 1, 1).expand(B, P, 1)).squeeze(2)
        max_sim = torch.maximum(max_sim, new_sim)
    return out
