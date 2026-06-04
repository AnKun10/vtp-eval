"""Map an average-token budget (avg over 32 LLM layers) to each baseline's knobs.

Vision-only methods (VisionZip, DivPrune) keep a constant N tokens across every
layer, so avg == N. FastV keeps 576 until layer K then R after, so
avg = (K*576 + (L-K)*R)/L with L=32.
"""
from __future__ import annotations

NUM_PATCHES = 576
NUM_LLM_LAYERS = 32
VISIONZIP_CONTEXTUAL_FRACTION = 1 / 6.4   # dominant:contextual ~= 5.4:1 (paper)


def divprune_ratio(avg_budget: int) -> float:
    """DivPrune SUBSET_RATIO = kept fraction of the 576 visual tokens."""
    return float(avg_budget) / NUM_PATCHES


def visionzip_knobs(avg_budget: int) -> tuple[int, int]:
    """(dominant, contextual) with dominant + contextual == avg_budget."""
    contextual = max(1, round(avg_budget * VISIONZIP_CONTEXTUAL_FRACTION))
    dominant = avg_budget - contextual
    return dominant, contextual


def fastv_knobs(avg_budget: int, k: int = 2) -> tuple[int, int]:
    """(K, R): R = number of image tokens kept after layer K.

    Raises ValueError if avg_budget is below the K-layer floor (K*576/L), where
    even R=0 cannot reach it — e.g. K=2 cannot reach avg-32 (floor = 36).
    """
    floor = k * NUM_PATCHES / NUM_LLM_LAYERS
    if avg_budget <= floor:
        raise ValueError(
            f"avg-{avg_budget} is below the FastV K={k} floor ({floor:.0f} tokens); "
            "even R=0 cannot reach it")
    r = round((avg_budget * NUM_LLM_LAYERS - k * NUM_PATCHES) / (NUM_LLM_LAYERS - k))
    return k, r
