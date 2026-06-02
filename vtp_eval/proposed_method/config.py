"""Hyperparameters for the proposed two-stage pruning method."""
from __future__ import annotations

from dataclasses import asdict, dataclass

NUM_PATCHES = 576  # LLaVA-1.5 CLIP ViT-L/14-336 → 24x24 patches


@dataclass
class ProposedConfig:
    dominant_k: int = 54     # Stage 1: top-k CLS-attention dominant tokens
    diversity_m: int = 10    # Stage 1: diversity top-up (R1 = dominant_k + diversity_m)
    pruned_layer: int = 12   # Stage 2: LLM layer whose attention drives scoring
    llm_keep_r2: int = 37    # Stage 2: vision tokens kept after layer k
    keep_position_ids: bool = False  # Stage 2 RoPE mode (both GPU-validated):
    #   False = re-index kept tokens to contiguous positions (FastV/SparseVLM-v1);
    #   True = keep original position ids (SparseVLM-v2 "Keep Position ID") +
    #   enlarge the RoPE cache so cos[position_ids] does not overflow.
    stage2_enabled: bool = True  # Run the Stage-2 LLM mid-layer prune.
    #   When False, only Stage 1 (vision-tower) pruning applies: every LLM layer
    #   then processes R1 visual tokens with a NORMAL (unpruned) KV cache, so
    #   decode stays clean (seq=1 per step). Temporarily False in the eval config
    #   while the Stage-2 decode path (KV-cache vs generate bookkeeping mismatch
    #   that re-feeds tokens) is reworked to no-slice + full-rotary.

    @property
    def r1(self) -> int:
        return self.dominant_k + self.diversity_m

    def avg_tokens(self, num_layers: int) -> float:
        """Average vision tokens across all LLM layers (FastV layer-counting:
        layers 0..k run at R1, layers k+1..L-1 run at R2). With Stage 2 disabled
        every layer runs at R1 (no mid-stack reduction)."""
        if not self.stage2_enabled:
            return float(self.r1)
        k = self.pruned_layer
        return ((k + 1) * self.r1 + (num_layers - k - 1) * self.llm_keep_r2) / num_layers

    def validate(self, num_llm_layers: int, num_patches: int = NUM_PATCHES) -> None:
        if self.dominant_k <= 0:
            raise ValueError(f"dominant_k must be > 0, got {self.dominant_k}")
        if self.diversity_m < 0:
            raise ValueError(f"diversity_m must be >= 0, got {self.diversity_m}")
        if self.r1 > num_patches:
            raise ValueError(
                f"dominant_k + diversity_m = {self.r1} exceeds {num_patches} patches")
        if not (0 <= self.pruned_layer < num_llm_layers):
            raise ValueError(
                f"pruned_layer must be in [0, {num_llm_layers}), got {self.pruned_layer}")
        if not (0 < self.llm_keep_r2 <= self.r1):
            raise ValueError(
                f"llm_keep_r2 must be in (0, R1={self.r1}], got {self.llm_keep_r2}")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_args(cls, defaults: dict | None, overrides: dict | None) -> "ProposedConfig":
        merged = {**(defaults or {}), **(overrides or {})}
        int_keys = {"dominant_k", "diversity_m", "pruned_layer", "llm_keep_r2"}
        kw = {k: int(v) for k, v in merged.items() if k in int_keys}
        for bkey in ("keep_position_ids", "stage2_enabled"):
            if bkey in merged:
                kw[bkey] = bool(merged[bkey])
        return cls(**kw)
