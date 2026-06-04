# vtp_eval/baselines/divprune/select.py
"""DivPrune diversity selection (CVPR 2025), ported verbatim from
divprune/LLaVA/llava/model/llava_arch.py::DivPrune + pairwise_cosine_similarity.

Greedy max-min: iteratively add the token whose minimum cosine-distance to the
already-selected set is largest. Operates on post-projector visual tokens.
"""
from __future__ import annotations

import torch


def _pairwise_cosine_similarity(matrix: torch.Tensor) -> torch.Tensor:
    norm = matrix / matrix.norm(dim=1, keepdim=True)
    return torch.mm(norm, norm.t())


def divprune_select(visual_features: torch.Tensor, keep: int) -> torch.Tensor:
    """visual_features: [P, D] post-projector tokens. Returns [keep] indices.

    Distance matrix = 1 - cosine_similarity. Step 0 seeds with the pair-closest
    token (topk(2, largest=False)[1]); each later step adds argmax of the
    per-token min-distance to the selected set. Matches the reference exactly.
    """
    cosine_dist = 1.0 - _pairwise_cosine_similarity(visual_features)
    s = torch.empty(keep, dtype=torch.long, device=visual_features.device)
    for i in range(keep):
        if i == 0:
            scores = torch.topk(cosine_dist, 2, dim=0, largest=False).values[1, :]
        else:
            m2 = torch.index_select(cosine_dist, 0, s[:i])
            scores = torch.min(m2, dim=0).values
        s[i] = torch.argmax(scores)
    return s
