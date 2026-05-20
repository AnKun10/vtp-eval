"""LLaVA-1.5-7B loading, forward pass, and per-word attention extraction."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .tokens import find_word_positions

DEPTH_NAMES = ("shallow", "middle", "deep")


def load_llava(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """Load processor + model in fp16 with ``attn_implementation='eager'``.

    Eager attention is required so ``output_attentions=True`` returns actual
    attention tensors (SDPA / FlashAttention silently return None).
    """
    proc = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map="cuda:0",
    ).eval()
    return model, proc


def extract_attention(model, processor, image: Image.Image, query: str,
                      words: Iterable[str], layers):
    """Run a single forward and return per-word head-averaged attention vectors.

    Returns
    -------
    pwl : dict[word, dict[depth_name, np.ndarray(576,)]]
        Head-averaged attention from each (multi-piece) target word to the 576
        vision tokens, at layers[0] / layers[1] / layers[2].
    word_positions : dict[word, list[int]]
        The token-position list each word resolved to.
    grid : int
        Spatial side of the vision-token grid (24 for LLaVA-1.5).
    sinks : set[int]
        Vision-token indices that are top-1 across all (word, depth) slices —
        flagged for sink-masking during downstream metrics / visualization.
    lyrs : dict[depth_name, int]
        Mapping from "shallow"/"middle"/"deep" to the requested layer indices.
    """
    import numpy as np

    img_tok_id = model.config.image_token_index
    prompt = f"USER: <image>\n{query} ASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        "cuda:0", torch.float16)
    ids = inputs["input_ids"][0]
    img_pos = (ids == img_tok_id).nonzero(as_tuple=False).flatten().tolist()
    s, e = img_pos[0], img_pos[-1]
    n_img = e - s + 1
    grid = int(math.sqrt(n_img))
    if grid * grid != n_img:
        raise ValueError(f"Expected square vision grid, got {n_img} tokens")

    word_positions = {}
    missing = []
    for w in words:
        p = find_word_positions(ids, processor.tokenizer, w, s, e)
        if p is None:
            missing.append(w)
        else:
            word_positions[w] = p
    if missing:
        raise ValueError(
            f"Words not found as contiguous tokens in prompt: {missing}. "
            f"Ensure --query contains each word verbatim."
        )

    with torch.inference_mode():
        out = model(**inputs, output_attentions=True, use_cache=False,
                    return_dict=True)
    attns = out.attentions

    lyrs = dict(zip(DEPTH_NAMES, layers))
    pwl = {w: {} for w in word_positions}
    for d, L in lyrs.items():
        A = attns[L][0].mean(0)                                # (seq, seq) head-avg
        for w, positions in word_positions.items():
            stacked = torch.stack([A[t, s:e + 1] for t in positions], dim=0)
            pwl[w][d] = stacked.mean(0).detach().float().cpu().numpy()
    del attns, out
    torch.cuda.empty_cache()

    sinks = {int(pwl[w][d].argmax()) for w in pwl for d in DEPTH_NAMES}
    return pwl, word_positions, grid, sinks, lyrs
