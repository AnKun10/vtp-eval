# vtp_eval/proposed_method/stage2_llm.py
"""Stage 2: mid-stack text->vision pruning inside LlamaModel.forward.

Pieces:
  - compute_spans / install_span_recorder: locate (and stash) where the vision
    and instruction tokens sit in the merged sequence.
  - prune_after_layer_k: the core prune (pure-ish; unit-tested) applied once
    after the configured layer during prefill.
  - _make_causal_mask_like: fresh additive causal mask for the reduced length.
  - make_llama_forward / _llama_forward_437: a copy of transformers 4.37.2
    LlamaModel.forward with prune_after_layer_k spliced in at layer k. The
    verbatim 4.37.2 body must be synced on the Vast.ai image (see Part E).
"""
from __future__ import annotations

import torch

from .selection import build_keep_index, slice_past_key_values, text_to_vision_scores


def compute_spans(input_ids: torch.Tensor, image_token_index: int, r1: int) -> dict:
    """Locate the vision + instruction spans in the POST-merge sequence.

    input_ids: [B, L] PRE-merge ids (one IMAGE_TOKEN_INDEX placeholder/sample).
    r1:        number of image feature tokens the placeholder expands into.
    Returns dict with per-sample vision_start/instr_start tensors ([B]),
    scalar vision_len (= r1), and post-merge seq_len.
    """
    B, L = input_ids.shape
    assert (input_ids == image_token_index).any(), \
        "compute_spans called on input_ids with no image token"
    img_pos = (input_ids == image_token_index).float().argmax(dim=1)  # [B] first hit
    vision_start = img_pos
    instr_start = img_pos + r1
    seq_len = L - 1 + r1            # placeholder (1 token) -> r1 tokens
    return {
        "vision_start": vision_start,
        "vision_len": r1,
        "instr_start": instr_start,
        "seq_len": seq_len,
    }


def _make_causal_mask_like(hidden_states: torch.Tensor, dtype) -> torch.Tensor:
    B, S, _ = hidden_states.shape
    min_val = torch.finfo(dtype).min
    mask = torch.full((S, S), min_val, device=hidden_states.device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(B, 1, S, S).contiguous()


def prune_after_layer_k(hidden_states, attn_k, position_ids, past_key_values,
                        spans, cfg, use_cache):
    """Apply the Stage-2 prune after layer k (prefill only; caller guards seq_len>1).

    Returns (hidden_states, position_ids, attention_mask, past_key_values) for
    layers k+1..L. attention_mask is a fresh additive causal mask for the reduced
    length. If there is no text to score with at all, returns inputs unchanged
    with attention_mask=None (caller keeps its existing mask).
    """
    vs = spans["vision_start"]            # [B]
    nv = spans["vision_len"]
    vstart = int(vs[0])                   # prefill: same vision_start across batch
    S = hidden_states.shape[1]
    instr_lo, instr_hi = vstart + nv, S
    if instr_hi <= instr_lo:              # no post-image instruction tokens
        instr_lo, instr_hi = 0, vstart    # fall back to pre-image text
    if instr_hi <= instr_lo:              # truly no text -> skip prune
        return hidden_states, position_ids, None, past_key_values

    assert attn_k is not None, (
        "layer-k attention is None — load the model with attn_implementation='eager' "
        "so output_attentions returns real tensors")
    scores = text_to_vision_scores(attn_k, (vstart, vstart + nv), (instr_lo, instr_hi))
    keep = build_keep_index(scores, cfg.llm_keep_r2, vs, seq_len=S)   # [B, keep_len]
    kl = keep.shape[1]
    D = hidden_states.shape[-1]
    hidden_states = hidden_states.gather(1, keep[:, :, None].expand(-1, kl, D))
    if position_ids is not None:
        # 2D [B,S] -> per-sample gather; 1D [S] (shared) -> [B,kl] via advanced index
        position_ids = position_ids.gather(1, keep) if position_ids.dim() == 2 \
            else position_ids[keep]
    attention_mask = _make_causal_mask_like(hidden_states, dtype=hidden_states.dtype)
    if past_key_values is not None:
        legacy = past_key_values.to_legacy_cache() \
            if hasattr(past_key_values, "to_legacy_cache") else past_key_values
        legacy = slice_past_key_values(legacy, keep)
        past_key_values = type(past_key_values).from_legacy_cache(legacy) \
            if hasattr(type(past_key_values), "from_legacy_cache") else legacy
    return hidden_states, position_ids, attention_mask, past_key_values


def install_span_recorder(model, cfg):
    """Wrap prepare_inputs_labels_for_multimodal to stash spans on the model.

    The wrapper records spans for the CURRENT batch into model._proposed_spans
    BEFORE delegating to the original (which merges the image features). Stage 2's
    forward reads model._proposed_spans.
    """
    orig = model.prepare_inputs_labels_for_multimodal
    image_token_index = getattr(model.config, "image_token_index", -200)

    def wrapped(input_ids, position_ids, attention_mask, past_key_values,
                labels, images, image_sizes=None, *args, **kw):
        if input_ids is not None and (input_ids == image_token_index).any():
            model._proposed_spans = compute_spans(input_ids, image_token_index, cfg.r1)
        else:
            model._proposed_spans = None
        return orig(input_ids, position_ids, attention_mask, past_key_values,
                    labels, images, image_sizes, *args, **kw)

    model.prepare_inputs_labels_for_multimodal = wrapped


def make_llama_forward(cfg):
    """Return a LlamaModel.forward that prunes once, after layer cfg.pruned_layer.

    Binds cfg to the version-pinned forward copy in _llama_forward_437, which
    must contain the verbatim transformers 4.37.2 LlamaModel.forward body with
    prune_after_layer_k spliced in (see module docstring + Part E sync note).
    """
    def forward(self, *args, **kw):
        return _llama_forward_437(self, cfg, *args, **kw)
    return forward


def _llama_forward_437(self, cfg, *args, **kw):
    """Verbatim transformers 4.37.2 LlamaModel.forward + Stage-2 prune splice.

    SYNC PROCEDURE (run on the Vast.ai image where transformers==4.37.2):
      1. `python -c "import inspect; from transformers.models.llama.modeling_llama \\
         import LlamaModel; print(inspect.getsource(LlamaModel.forward))"`
      2. Paste that body here, replacing this stub. Keep the signature
         `(self, cfg, *args, **kw)` by binding the original kwargs.
      3. Inside the decoder-layer loop:
         - force attention at layer k: pass
           `output_attentions=output_attentions or (idx == cfg.pruned_layer)`
           into `decoder_layer(...)`.
         - immediately AFTER `hidden_states = layer_outputs[0]`, insert (note the
           mask guard: the no-text skip path returns _mask=None meaning "keep the
           existing mask" — never overwrite attention_mask with None, or causal
           masking is lost for layers k+1..L):
             spans = getattr(self, "_proposed_spans", None)
             if idx == cfg.pruned_layer and spans is not None and hidden_states.shape[1] > 1:
                 _hs, _pos, _mask, _pkv = prune_after_layer_k(
                     hidden_states, layer_outputs[1], position_ids,
                     past_key_values, spans, cfg, use_cache)
                 hidden_states, position_ids, past_key_values = _hs, _pos, _pkv
                 if _mask is not None:
                     attention_mask = _mask
      4. Verify with `tests/test_proposed_smoke.py -m slow`.
    """
    raise NotImplementedError(
        "_llama_forward_437 must be synced with transformers 4.37.2 on the Vast.ai "
        "image — see this function's docstring for the procedure.")
