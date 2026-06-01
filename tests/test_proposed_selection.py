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
