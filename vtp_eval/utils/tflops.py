"""Theoretical prefill FLOPs estimates for LLaVA-1.5 7B with visual-token pruning.

Approximation only — ignores RMSNorm/RoPE/SwiGLU activation, treats attention
as full quadratic (no causal mask discount). Sufficient for relative comparison.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class LLaMAConfig:
    n_layers: int = 32
    d_model: int = 4096
    d_ffn: int = 11008  # SwiGLU intermediate
    n_heads: int = 32


LLAVA_15_7B = LLaMAConfig()

# POPE prompt layout
SYS_LEN = 35
VISUAL_FULL = 576
TEXT_Q = 10
TEXT_A = 1


def layer_flops(seq_len: int, cfg: LLaMAConfig = LLAVA_15_7B) -> float:
    """Prefill FLOPs for one transformer layer at given seq_len."""
    d, m = cfg.d_model, cfg.d_ffn
    attn_proj = 4 * seq_len * d * d            # Q, K, V, O linear
    attn_mm   = 2 * seq_len * seq_len * d      # QK^T + softmax · V
    ffn       = 3 * seq_len * d * m            # SwiGLU: gate + up + down
    return float(attn_proj + attn_mm + ffn)


def total_llm_flops(visual_per_layer: List[int],
                    sys_len: int = SYS_LEN,
                    text_q: int = TEXT_Q,
                    text_a: int = TEXT_A,
                    cfg: LLaMAConfig = LLAVA_15_7B) -> float:
    """Sum prefill FLOPs across all LLM layers.

    visual_per_layer[i] is the visual-token count at layer i.
    """
    return sum(
        layer_flops(sys_len + v + text_q + text_a, cfg)
        for v in visual_per_layer
    )


# Visual-token-per-layer shapes per method ---------------------------------

def shape_baseline() -> List[int]:
    return [VISUAL_FULL] * 32


def shape_fastv(K: int, R: int) -> List[int]:
    """FastV prunes after layer K (1-indexed exclusive: layer K still sees all)."""
    return [VISUAL_FULL] * (K + 1) + [R] * (32 - K - 1)


def shape_sparsevlm(retain: int) -> List[int]:
    """SparseVLMs progressive schedule (verified from score.py:11-14).

    For retain=192: '2*576 4*300 10*200 16*110' (from source comment).
    """
    schedule = {
        192: (300, 200, 110),
        128: (303, 110, 36),
        96:  (238, 48,  26),
        64:  (66,  30,  17),
    }
    s = schedule[retain]
    return [VISUAL_FULL] * 2 + [s[0]] * 4 + [s[1]] * 10 + [s[2]] * 16


def shape_visionzip(dominant: int, contextual: int) -> List[int]:
    """VisionZip prunes before LLM — same count across all layers."""
    return [dominant + contextual] * 32


def shape_divprune(ratio: float) -> List[int]:
    """DivPrune prunes at projector (pre-LLM)."""
    return [int(VISUAL_FULL * ratio)] * 32


def shape_sparsevila(enc_ratio: float, dec_ratio: float) -> List[int]:
    """SparseVILA encoder prune dominates prefill cost.

    Decode-time retrieval (dec_ratio) reduces decode FLOPs but not prefill,
    so it doesn't enter this prefill formula.
    """
    return [int(VISUAL_FULL * (1 - enc_ratio))] * 32


def clip_encoder_flops(visual_keep: int = 577) -> float:
    """CLIP-ViT-L/14: 24 layers, d=1024, d_ffn=4096."""
    cfg = LLaMAConfig(n_layers=24, d_model=1024, d_ffn=4096, n_heads=16)
    return cfg.n_layers * layer_flops(visual_keep, cfg)


# Top-level facade --------------------------------------------------------

_SHAPE_FNS = {
    "baseline":   lambda **_: shape_baseline(),
    "fastv":      lambda K, R, **_: shape_fastv(K, R),
    "sparsevlm":  lambda retain, **_: shape_sparsevlm(retain),
    "visionzip":  lambda dominant, contextual, **_: shape_visionzip(dominant, contextual),
    "divprune":   lambda ratio, **_: shape_divprune(ratio),
    "sparsevila": lambda enc_ratio, dec_ratio, **_: shape_sparsevila(enc_ratio, dec_ratio),
}


def compute_method_tflops(method: str, **kwargs) -> dict:
    """Compute prefill TFLOPs for a method given its pruning args.

    Returns: dict with keys
      tflops_llm_prefill, tflops_encoder, tflops_total, visual_shape
    """
    if method not in _SHAPE_FNS:
        raise KeyError(f"Unknown method: {method!r}. "
                       f"Choices: {sorted(_SHAPE_FNS.keys())}")
    visual_shape = _SHAPE_FNS[method](**kwargs)
    llm = total_llm_flops(visual_shape)
    enc = clip_encoder_flops()
    return {
        "tflops_llm_prefill": llm / 1e12,
        "tflops_encoder":     enc / 1e12,
        "tflops_total":       (llm + enc) / 1e12,
        "visual_shape":       visual_shape,
    }
