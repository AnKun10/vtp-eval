import numpy as np
from PIL import Image
from vtp_eval.insight.prune_viz import render


def test_patch_to_box_corners():
    # grid 24, patch 14 -> 336x336. patch 0 = top-left, 25 = (row1,col1).
    assert render.patch_to_box(0) == (0, 0, 14, 14)
    assert render.patch_to_box(25) == (14, 14, 28, 28)         # row=1,col=1
    assert render.patch_to_box(575) == (322, 322, 336, 336)     # bottom-right


def test_keep_mask_marks_only_kept_cells():
    m = render.keep_mask([0, 25, 575])
    assert m.shape == (24, 24)
    assert m[0, 0] and m[1, 1] and m[23, 23]
    assert m.sum() == 3


def test_mask_overlay_kept_more_opaque_than_pruned():
    img = Image.new("RGB", (336, 336), (200, 100, 50))
    rgba = render.mask_overlay(img, keep_indices=[0])   # only patch 0 kept
    assert rgba.shape == (336, 336, 4)
    # alpha channel: kept patch (top-left 14x14) opaque, elsewhere dimmed
    assert rgba[0, 0, 3] > rgba[300, 300, 3]
    assert rgba[0, 0, 3] == 1.0
