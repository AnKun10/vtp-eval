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

from .selection import build_keep_index, text_to_vision_scores


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


def prune_after_layer_k(hidden_states, attn_k, spans, cfg):
    """Apply the Stage-2 prune after layer k (prefill only; caller guards seq_len>1).

    FastV-style **NO-SLICE**: drop the low-scoring vision tokens from the hidden
    states / position_ids / causal mask feeding layers k+1..L, but leave the KV
    cache untouched. Layers 0..k therefore keep their FULL KV (and full length),
    so generate()'s past_length (read from layer 0) stays equal to the true
    sequence length and decode feeds exactly one token (no re-feed). Layers
    k+1..L build a shorter KV from only the kept tokens — that is where the
    compute/memory saving comes from (the cache becomes ragged across layers).

    The kept tokens retain their ORIGINAL position ids (gapped); attention among
    them is causal in kept order (`keep` is ascending). Original positions need
    the RoPE cache enlarged via install_full_rotary so cos[position] does not
    overflow on the shorter kv. Returns (hidden_states, position_ids,
    attention_mask); the KV cache is intentionally NOT touched. If there is no
    text to score with, returns inputs unchanged with attention_mask=None.
    """
    vs = spans["vision_start"]            # [B]
    nv = spans["vision_len"]
    vstart = int(vs[0])                   # prefill: same vision_start across batch
    S = hidden_states.shape[1]
    instr_lo, instr_hi = vstart + nv, S
    if instr_hi <= instr_lo:              # no post-image instruction tokens
        instr_lo, instr_hi = 0, vstart    # fall back to pre-image text
    if instr_hi <= instr_lo:              # truly no text -> skip prune
        return hidden_states, None, None

    assert attn_k is not None, (
        "layer-k attention is None — load the model with attn_implementation='eager' "
        "so output_attentions returns real tensors")
    scores = text_to_vision_scores(attn_k, (vstart, vstart + nv), (instr_lo, instr_hi))
    keep = build_keep_index(scores, cfg.llm_keep_r2, vs, seq_len=S)   # [B, keep_len]
    B, kl = keep.shape
    D = hidden_states.shape[-1]
    hidden_states = hidden_states.gather(1, keep[:, :, None].expand(-1, kl, D))
    position_ids = keep.to(torch.long)    # ORIGINAL positions (full rotary required)
    attention_mask = _make_causal_mask_like(hidden_states, dtype=hidden_states.dtype)
    return hidden_states, position_ids, attention_mask


def install_span_recorder(model, cfg):
    """Wrap prepare_inputs_labels_for_multimodal to stash spans on the model.

    The wrapper records spans for the CURRENT batch into model._proposed_spans
    BEFORE delegating to the original (which merges the image features). Stage 2's
    forward reads model._proposed_spans.
    """
    orig = model.prepare_inputs_labels_for_multimodal
    image_token_index = getattr(model.config, "image_token_index", -200)
    # Stash spans on the INNER language model — that is the object whose patched
    # forward (`self`) reads `_proposed_spans`. (For the bare-stub unit test with
    # no `.model`, fall back to the model itself.)
    lm = getattr(model, "model", model)

    def wrapped(input_ids, position_ids, attention_mask, past_key_values,
                labels, images, image_sizes=None, *args, **kw):
        if input_ids is not None and (input_ids == image_token_index).any():
            lm._proposed_spans = compute_spans(input_ids, image_token_index, cfg.r1)
        else:
            lm._proposed_spans = None
        return orig(input_ids, position_ids, attention_mask, past_key_values,
                    labels, images, image_sizes, *args, **kw)

    model.prepare_inputs_labels_for_multimodal = wrapped


_ORIG_ROTARY_FORWARD = None


def install_full_rotary(min_seqlen):
    """Patch ``LlamaRotaryEmbedding.forward`` (transformers 4.37.2) to always
    build/return cos/sin covering at least ``min_seqlen`` positions.

    Needed only for ``keep_position_ids=True`` (SparseVLM-v2): the kept tokens
    keep their ORIGINAL (possibly large, gapped) position ids, but stock 4.37.2
    sizes each layer's RoPE cache to the reduced kv length, so ``cos[position_ids]``
    overflows. Bumping the requested ``seq_len`` to ``max_position_embeddings``
    makes the returned cos/sin large enough. Idempotent; transformers import
    deferred so this module stays portable.
    """
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    global _ORIG_ROTARY_FORWARD
    if _ORIG_ROTARY_FORWARD is None:
        _ORIG_ROTARY_FORWARD = LlamaRotaryEmbedding.forward
    orig = _ORIG_ROTARY_FORWARD

    def forward(self, x, seq_len=None):
        seq_len = max(int(seq_len) if seq_len is not None else 0, int(min_seqlen))
        return orig(self, x, seq_len=seq_len)

    LlamaRotaryEmbedding.forward = forward


def make_llama_forward(cfg):
    """Return a LlamaModel.forward (transformers 4.37.2) with the Stage-2 prune
    spliced in at layer ``cfg.pruned_layer``."""
    def forward(self, *args, **kw):
        return _llama_forward_437(self, cfg, *args, **kw)
    return forward


def _llama_forward_437(self, cfg, input_ids=None, attention_mask=None,
                       position_ids=None, past_key_values=None, inputs_embeds=None,
                       use_cache=None, output_attentions=None,
                       output_hidden_states=None, return_dict=None, **kwargs):
    """transformers 4.37.2 ``LlamaModel.forward`` + Stage-2 text->vision prune.

    A faithful copy of the 4.37.2 body (inference path; the train-only
    gradient-checkpointing branch is dropped) with one splice: after layer
    ``cfg.pruned_layer`` runs during PREFILL, drop all but the top-R2 vision
    tokens (scored by instruction->vision attention) from the sequence,
    position_ids and causal mask for the remaining layers.

    The KV cache is deliberately NOT sliced: layers 0..k keep their full KV
    (FastV behavior — savings come from layers k+1..L processing fewer tokens),
    so layer 0's cache length stays equal to the true sequence length and
    generate() feeds exactly one token per decode step (no re-feed). The kept
    tokens keep their ORIGINAL position ids, so the RoPE cache must be enlarged
    (install_full_rotary). The resulting cache is ragged across layers; at decode
    the attention mask is dropped (a single new token attends to all of each
    layer's kv anyway) so the one fixed-length 4D mask never has to span both.

    transformers-internal imports are deferred so importing this module never
    requires transformers 4.37.2 (keeps the pure-tensor unit tests portable).
    """
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask,
        _prepare_4d_causal_attention_mask_for_sdpa,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Stage-2 is NO-SLICE: layers 0..k keep their full KV while layers k+1..L
    # keep a shorter KV, so the cache is ragged across layers. A single fixed 4D
    # mask cannot span both — but at decode (seq_len==1) the lone new token
    # should attend to ALL of each layer's cache, so drop the mask (with
    # seq_len==1, _prepare_4d_* returns None) and let every layer attend its full
    # kv. position_ids = the token's TRUE position (full rotary keeps it in
    # range). Fire only once a prune has happened (_proposed_last_prune set).
    if seq_length == 1 and past_key_values_length > 0 \
            and getattr(self, "_proposed_last_prune", None) is not None:
        attention_mask = None
        _dev = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype=torch.long, device=_dev).unsqueeze(0)

    if getattr(self, "_use_flash_attention_2", False):
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif getattr(self, "_use_sdpa", False) and not output_attentions:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
    else:
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_oa = output_attentions or (idx == cfg.pruned_layer)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=layer_oa,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]

        # --- Stage 2: prune vision tokens after layer k (prefill only) ---
        spans = getattr(self, "_proposed_spans", None)
        if idx == cfg.pruned_layer and spans is not None and hidden_states.shape[1] > 1:
            _s_before = int(hidden_states.shape[1])
            _hs, _pos, _mask = prune_after_layer_k(
                hidden_states, layer_outputs[1], spans, cfg)
            hidden_states = _hs
            if _pos is not None:        # None only when there was no text to score
                position_ids = _pos
            if _mask is not None:
                attention_mask = _mask
            # The KV cache is deliberately left intact (no-slice); layers k+1..L
            # below simply append shorter KVs for the kept tokens.
            # Persistent proof the prune fired (survives decode steps, unlike
            # _proposed_spans which the span recorder resets each decode token).
            self._proposed_last_prune = {"layer": idx, "seq_before": _s_before,
                                         "seq_after": int(hidden_states.shape[1])}

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = past_key_values.to_legacy_cache() if use_legacy_cache else past_key_values
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
