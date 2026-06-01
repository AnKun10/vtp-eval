# tests/test_proposed_selection.py
import torch
import torch.nn.functional as F
from vtp_eval.proposed_method import selection


def test_cls_topk_dominant_picks_highest():
    # [B=1, P=5] attention sums; top-3 should be indices {4,3,2}
    s = torch.tensor([[0.1, 0.2, 0.5, 0.9, 1.0]])
    idx = selection.cls_topk_dominant(s, dominant_k=3)
    assert set(idx[0].tolist()) == {2, 3, 4}


def test_diversity_excludes_seed_indices():
    feats = F.normalize(torch.randn(1, 10, 8), dim=-1)
    seed = torch.tensor([[0, 1]])
    out = selection.farthest_point_diversity(feats, seed, m=3)
    assert out.shape == (1, 3)
    assert set(out[0].tolist()).isdisjoint({0, 1})            # never re-pick seeds
    assert len(set(out[0].tolist())) == 3                      # all distinct


def test_diversity_prefers_dissimilar_token():
    # 3 tokens: two near-identical (0,1), one orthogonal (2). Seed={0}.
    # The most-dissimilar pick must be token 2.
    feats = torch.tensor([[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]])
    feats = F.normalize(feats, dim=-1)
    seed = torch.tensor([[0]])
    out = selection.farthest_point_diversity(feats, seed, m=1)
    assert out[0, 0].item() == 2


def test_select_stage1_shapes_and_sorted():
    B, H, P, Dh, D = 2, 4, 20, 6, 16
    attn = torch.rand(B, H, 1 + P, 1 + P)
    hidden = torch.randn(B, 1 + P, D)
    keep = selection.select_stage1(attn, hidden, dominant_k=5, diversity_m=3)
    assert keep.shape == (B, 8)                       # R1 = 5 + 3
    assert (keep[:, 1:] >= keep[:, :-1]).all()        # sorted ascending
    assert keep.max().item() < P and keep.min().item() >= 0
    for b in range(B):
        assert len(set(keep[b].tolist())) == 8        # no duplicates


def test_select_stage1_diversity_zero_returns_dominant_only():
    B, H, P, D = 1, 2, 12, 8
    attn = torch.rand(B, H, 1 + P, 1 + P)
    hidden = torch.randn(B, 1 + P, D)
    keep = selection.select_stage1(attn, hidden, dominant_k=4, diversity_m=0)
    assert keep.shape == (1, 4)
