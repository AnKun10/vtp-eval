# vtp_eval/demo/overlay.py
"""Map R2-local indices to original patches and render keep-set overlays.

Reuses insight/prune_viz/render.mask_overlay (kept patches opaque, pruned dimmed)
and converts its RGBA float array to an RGB PIL image for Gradio. Pure.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vtp_eval.insight.prune_viz.render import mask_overlay


def r2_local_to_patches(r1_idx: torch.Tensor, r2_local: torch.Tensor) -> torch.Tensor:
    """r1_idx: [R1] original patch ids kept by Stage 1 (sorted).
    r2_local: [r2] positions WITHIN the R1 block kept by Stage 2.
    Returns [r2] original patch ids, sorted ascending."""
    return r1_idx[r2_local.to(r1_idx.device)].sort().values


def overlay_pil(image: Image.Image, keep_indices) -> Image.Image:
    """Render `image` with kept patches highlighted, pruned patches dimmed,
    flattened onto white. Returns an RGB PIL image at the input size."""
    if isinstance(keep_indices, torch.Tensor):
        keep_indices = keep_indices.cpu().numpy()
    rgba = mask_overlay(image, keep_indices)              # [H, W, 4] float in [0,1]
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    composited = rgb * alpha + (1.0 - alpha)              # over white
    return Image.fromarray((composited * 255).astype(np.uint8), mode="RGB")
