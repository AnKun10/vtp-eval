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


def extract_attention(model, tok, image_processor, image, query, words,
                      layers, depth_names=DEPTH_NAMES):
    """Liuhaotian-stack TVA extraction. Returns
    {pwl, word_positions, grid, sinks, lyrs} (same contract the TVA
    metrics/visualize helpers consume). ``words`` must appear verbatim in
    ``query`` (they are drawn from the benchmark question)."""
    import math

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    from vtp_eval.insight.text_visual_attention.tokens import find_word_positions

    prompt = f"USER: <image>\n{query} ASSISTANT:"
    input_ids = tokenizer_image_token(
        prompt, tok, IMAGE_TOKEN_INDEX,
        return_tensors="pt").unsqueeze(0).to(model.device)
    ids = input_ids[0]
    vpos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
    if len(vpos) != 1:
        raise ValueError(f"expected exactly one image placeholder, got {len(vpos)}")
    vstart = vpos[0]
    n_vis = 576
    grid = int(math.sqrt(n_vis))                 # 24 for LLaVA-1.5

    # Target-word positions in the PRE-merge ids (exclude the 1 placeholder token).
    word_positions, missing = {}, []
    for w in words:
        p = find_word_positions(ids, tok, w, vstart, vstart)
        if p is not None:
            word_positions[w] = p
        else:
            missing.append(w)
    if missing:
        raise ValueError(f"words not found verbatim in query: {missing}")
    wp_merged = {w: [to_merged_index(t, vstart, n_vis) for t in ps]
                 for w, ps in word_positions.items()}

    image_tensor = process_images([image.convert("RGB")], image_processor,
                                  model.config)[0].unsqueeze(0).half().to(model.device)
    with torch.inference_mode():
        out = model(input_ids, images=image_tensor, image_sizes=[image.size],
                    output_attentions=True, use_cache=False, return_dict=True)
    pwl, sinks = pwl_from_attentions(out.attentions, wp_merged, vstart, layers,
                                     n_vis=n_vis, depth_names=depth_names)
    lyrs = dict(zip(depth_names, layers))
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pwl": pwl, "word_positions": word_positions, "grid": grid,
            "sinks": sinks, "lyrs": lyrs}
