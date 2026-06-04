# tests/test_divprune_select.py
import torch
from vtp_eval.baselines.divprune.select import divprune_select


def test_picks_spread_out_points():
    # Four tokens: two near-identical, two far apart. Selecting 3 must drop one
    # of the duplicates (keep diverse coverage), not pick both duplicates first.
    feats = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],   # near-duplicate of row 0
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    idx = divprune_select(feats, keep=3)
    chosen = set(idx.tolist())
    assert len(chosen) == 3
    assert not ({0, 1} <= chosen)          # never both duplicates
    assert 2 in chosen and 3 in chosen     # both far points kept


def test_returns_requested_count():
    feats = torch.randn(50, 8)
    assert divprune_select(feats, keep=10).shape == (10,)
