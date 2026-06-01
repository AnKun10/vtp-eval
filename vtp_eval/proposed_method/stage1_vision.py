# vtp_eval/proposed_method/stage1_vision.py
"""Stage 1: replace CLIPVisionTower.forward to emit only R1 selected patches.

Mirrors VisionZip's clip_encoder patch point on the original LLaVA repo, but
selects dominant (CLS-attention top-k) + diversity (farthest-point) patches
instead of dominant + merged-contextual.
"""
from __future__ import annotations

import torch

from .selection import select_stage1


def select_features(image_forward_outs, cfg) -> torch.Tensor:
    """Pure core: given a CLIP forward output (with .attentions/.hidden_states),
    return [B, R1, D] selected penultimate patch features."""
    attn_penult = image_forward_outs.attentions[-2]
    hidden_penult = image_forward_outs.hidden_states[-2]
    keep = select_stage1(attn_penult, hidden_penult, cfg.dominant_k, cfg.diversity_m)
    B, R1 = keep.shape
    D = hidden_penult.shape[-1]
    patches = hidden_penult[:, 1:, :]                       # drop CLS
    return patches.gather(1, keep[:, :, None].expand(B, R1, D))


def make_forward(cfg):
    """Build a CLIPVisionTower.forward bound to this config."""

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            out = []
            for image in images:
                fo = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True, output_attentions=True)
                out.append(select_features(fo, cfg).to(image.dtype))
            return out
        fo = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True, output_attentions=True)
        return select_features(fo, cfg).to(images.dtype)

    return forward
