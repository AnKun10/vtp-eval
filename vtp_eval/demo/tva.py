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


DEPTH_NAMES = ("shallow", "middle", "deep")


def pwl_from_attentions(attns, word_positions_merged, vstart, layers,
                        n_vis: int = 576, depth_names=DEPTH_NAMES):
    """Pure post-processing of attention tensors into per-(word, depth) vectors.

    ``attns``: sequence indexable by layer index -> tensor [B, H, S, S] (or
    [H, S, S]). ``word_positions_merged``: dict[word, list[int]] token positions
    in the MERGED frame. Returns (pwl, sinks):
      pwl[word][depth] = float ndarray (n_vis,) head-averaged attention from the
        word's tokens to the vision block [vstart, vstart + n_vis);
      sinks = set of vision indices that are top-1 across all (word, depth).
    """
    lyrs = dict(zip(depth_names, layers))
    pwl = {w: {} for w in word_positions_merged}
    for d, L in lyrs.items():
        A = attns[L]
        A = A[0] if A.dim() == 4 else A          # [H, S, S]
        A = A.mean(0)                            # [S, S] head-averaged
        for w, positions in word_positions_merged.items():
            stacked = torch.stack([A[t, vstart:vstart + n_vis] for t in positions],
                                  dim=0)
            pwl[w][d] = stacked.mean(0).detach().float().cpu().numpy()
    sinks = {int(pwl[w][d].argmax()) for w in pwl for d in depth_names}
    return pwl, sinks
