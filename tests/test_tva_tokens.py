# tests/test_tva_tokens.py
from vtp_eval.demo.tva import to_merged_index


def test_to_merged_index_before_at_and_after_placeholder():
    # vstart = pre-merge index of the single image placeholder.
    assert to_merged_index(1, vstart=5, n_vis=576) == 1        # before: unchanged
    assert to_merged_index(5, vstart=5, n_vis=576) == 5        # the placeholder itself
    assert to_merged_index(6, vstart=5, n_vis=576) == 6 + 575  # after: + (n_vis-1)
    assert to_merged_index(10, vstart=5, n_vis=4) == 10 + 3    # n_vis-1 = 3


import numpy as np
import torch
from vtp_eval.demo.tva import pwl_from_attentions


def test_pwl_from_attentions_shapes_and_values():
    S, n_vis, vstart = 12, 4, 2
    torch.manual_seed(0)
    attns = [torch.rand(1, 2, S, S) for _ in range(3)]   # 3 layers, [B,H,S,S]
    wp_merged = {"cat": [7, 9]}                           # merged token positions
    pwl, sinks = pwl_from_attentions(
        attns, wp_merged, vstart, layers=[0, 1, 2], n_vis=n_vis)

    assert set(pwl) == {"cat"}
    assert set(pwl["cat"]) == {"shallow", "middle", "deep"}
    assert pwl["cat"]["shallow"].shape == (n_vis,)
    # shallow = layer 0, head-averaged, rows {7,9}, cols vstart:vstart+n_vis, mean
    A = attns[0][0].mean(0)
    expect = torch.stack([A[7, vstart:vstart + n_vis],
                          A[9, vstart:vstart + n_vis]]).mean(0).numpy()
    assert np.allclose(pwl["cat"]["shallow"], expect, atol=1e-6)
    assert len(sinks) >= 1 and all(0 <= s < n_vis for s in sinks)
