# vtp_eval/demo/tva.py
"""Text-visual-attention (TVA) extraction ported to the shared liuhaotian model.

The original TVA (`vtp_eval/insight/text_visual_attention/attention.py`) runs on
llava-hf, where input_ids already contain 576 inline image tokens. The liuhaotian
model instead carries ONE image placeholder (IMAGE_TOKEN_INDEX = -200) that
expands into 576 vision tokens inside prepare_inputs_labels_for_multimodal. So a
target word's pre-merge token index must be shifted into the merged frame before
indexing the attention tensors. Pure helpers (`to_merged_index`,
`pwl_from_attentions`) are CPU-testable; `extract_attention` needs the GPU model.
Reuses the model-agnostic tokens/metrics/visualize helpers from the TVA package.
"""
from __future__ import annotations

import torch


def to_merged_index(pos: int, vstart: int, n_vis: int = 576) -> int:
    """Map a PRE-merge token index to the merged sequence, where the single image
    placeholder at ``vstart`` expands into ``n_vis`` tokens.

    Positions before the placeholder are unchanged; the placeholder maps to the
    first vision token (``vstart``); positions after shift by ``n_vis - 1``.
    """
    if pos < vstart:
        return pos
    if pos == vstart:
        return vstart
    return pos + (n_vis - 1)
