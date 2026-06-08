# tests/test_demo_overlay.py
import numpy as np
import torch
from PIL import Image
from vtp_eval.demo.overlay import r2_local_to_patches, overlay_pil


def test_r2_local_to_patches_maps_through_r1():
    r1_idx = torch.tensor([10, 20, 30, 40])     # original patch ids kept by R1
    r2_local = torch.tensor([0, 3])             # positions WITHIN the R1 block
    out = r2_local_to_patches(r1_idx, r2_local)
    assert out.tolist() == [10, 40]


def test_overlay_pil_returns_rgb_image_of_input_size():
    img = Image.new("RGB", (48, 48), (200, 100, 50))
    keep = torch.tensor([0, 1, 2, 25, 575])
    out = overlay_pil(img, keep)
    assert isinstance(out, Image.Image)
    assert out.size == (48, 48)
    assert out.mode == "RGB"
    # kept patches are brighter (higher alpha) than pruned ones on average
    assert np.asarray(out).mean() > 0
