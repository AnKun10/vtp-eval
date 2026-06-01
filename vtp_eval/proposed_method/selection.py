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

    feats:    [B, P, D], floating-point and L2-normalized (dot == cosine).
    seed_idx: [B, k] indices already selected (the dominant set; k >= 1 —
              callers pass the dominant top-k from cls_topk_dominant).
    Returns:  [B, m] newly selected indices, disjoint from the seed set and
              from each other, each chosen to minimize its maximum cosine
              similarity to the already-selected set.
    """
    assert feats.is_floating_point(), "feats must be a floating-point tensor"
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


def select_stage1(attn_penult: torch.Tensor, hidden_penult: torch.Tensor,
                  dominant_k: int, diversity_m: int) -> torch.Tensor:
    """Compose Stage 1 selection over the penultimate vision-encoder layer.

    attn_penult:   [B, H, 1+P, 1+P] attention (index 0 = CLS).
    hidden_penult: [B, 1+P, D] hidden states (index 0 = CLS).
    Returns: [B, R1] PATCH indices in [0, P), sorted ascending. R1 = k + m.
    CLS is used only as the attention query; it is not part of the output.
    """
    cls_attn = attn_penult[:, :, 0, 1:].sum(dim=1)        # [B, P] head-summed
    dominant = cls_topk_dominant(cls_attn, dominant_k)    # [B, k]

    if diversity_m > 0:
        patches = hidden_penult[:, 1:, :]                 # [B, P, D]
        feats = F.normalize(patches, dim=-1)
        diversity = farthest_point_diversity(feats, dominant, diversity_m)  # [B, m]
        keep = torch.cat([dominant, diversity], dim=1)
    else:
        keep = dominant

    return keep.sort(dim=1).values


def text_to_vision_scores(attn_layer: torch.Tensor,
                          vision_slice: tuple[int, int],
                          instr_slice: tuple[int, int]) -> torch.Tensor:
    """Importance = attention each vision token receives from instruction tokens.

    attn_layer:   [B, H, S, S] attention from the pruned LLM layer.
    vision_slice: (start, end) exclusive-end of the vision-token block.
    instr_slice:  (start, end) exclusive-end of the post-image instruction tokens.
    Returns: [B, Nv] mean over heads and instruction rows.
    """
    vs, ve = vision_slice
    is_, ie = instr_slice
    sub = attn_layer[:, :, is_:ie, vs:ve]   # [B, H, Ni, Nv]
    return sub.mean(dim=(1, 2))             # [B, Nv]


def build_keep_index(scores: torch.Tensor, r2: int,
                     vision_start: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Keep ALL non-vision tokens + the top-``r2`` vision tokens.

    scores:       [B, Nv] vision-token importance.
    vision_start: [B] absolute start position of the vision block per sample.
    Returns: [B, seq_len - Nv + r2] sorted absolute indices.
    Assumes the vision block is contiguous and length Nv is the same across B.
    """
    B, Nv = scores.shape
    assert vision_start.shape[0] == B, "vision_start length must equal batch size"
    device = scores.device
    top_local = scores.topk(r2, dim=1).indices            # [B, r2]
    keep_rows = []
    for b in range(B):
        vstart = int(vision_start[b])
        vis_abs = top_local[b] + vstart
        non_vis = torch.cat([
            torch.arange(0, vstart, device=device),
            torch.arange(vstart + Nv, seq_len, device=device),
        ])
        keep_rows.append(torch.cat([non_vis, vis_abs]).sort().values)
    return torch.stack(keep_rows, dim=0)


def slice_past_key_values(past_key_values, keep_index: torch.Tensor) -> tuple:
    """Gather the sequence dimension of a legacy past_key_values tuple.

    Legacy format: tuple(per layer) of (key, value), each
    [B, n_heads, seq, head_dim]. keep_index: [B, keep_len].
    Returns a new tuple in the same format with seq sliced to keep_len.
    (DynamicCache: see stage2_llm.py, which converts to legacy before calling.)
    """
    keep_len = keep_index.shape[1]
    out = []
    for key, value in past_key_values:
        B, nh, _, hd = key.shape
        idx = keep_index[:, None, :, None].expand(B, nh, keep_len, hd)
        out.append((key.gather(2, idx), value.gather(2, idx)))
    return tuple(out)
