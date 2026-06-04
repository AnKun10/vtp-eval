"""Map an average-token budget (avg over 32 LLM layers) to each baseline's knobs.

Vision-only methods (VisionZip, DivPrune) keep a constant N tokens across every
layer, so avg == N. For FastV, use FastVConfig.avg_tokens(num_layers) directly —
the true avg (which drives TFLOPs) is computed from pruned_layer and retain_tokens.
"""
from __future__ import annotations

NUM_PATCHES = 576
NUM_LLM_LAYERS = 32
VISIONZIP_CONTEXTUAL_FRACTION = 1 / 6.4   # dominant:contextual ~= 5.4:1 (paper)


def visionzip_knobs(avg_budget: int) -> tuple[int, int]:
    """(dominant, contextual) with dominant + contextual == avg_budget."""
    contextual = max(1, round(avg_budget * VISIONZIP_CONTEXTUAL_FRACTION))
    dominant = avg_budget - contextual
    return dominant, contextual
