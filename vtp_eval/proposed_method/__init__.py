# vtp_eval/proposed_method/__init__.py
"""Proposed two-stage visual-token pruning method (thesis contribution).

Training-free: Stage 1 prunes redundant tokens after the vision encoder
(CLS-attention dominant + diversity-seeded context); Stage 2 prunes
text-irrelevant tokens at an LLM middle layer (text->vision attention).
"""
from .config import ProposedConfig
from .patch import proposed_prune

__all__ = ["ProposedConfig", "proposed_prune"]
