# vtp_eval/demo/compare.py
"""Pure formatting of the per-turn proposed-vs-vanilla comparison bubble.

No gradio/torch imports, so it unit-tests on CPU.
"""
from __future__ import annotations


def format_comparison(proposed_answer: str, proposed_latency: float,
                      cache_hit: bool, r2_tokens: int, no_cache_latency: float,
                      vanilla_answer: str, vanilla_latency: float,
                      vanilla_tokens: int = 576) -> str:
    """Build the Markdown assistant bubble comparing proposed vs vanilla.

    Shows the proposed answer/latency/token-count/cache state, the proposed
    no-cache latency, the vanilla answer/latency/token-count, and the speedup
    (vanilla / proposed). Division is guarded so a 0.0s proposed latency yields
    0.0× instead of raising.
    """
    speedup = (vanilla_latency / proposed_latency) if proposed_latency > 0 else 0.0
    hit = "CACHE HIT" if cache_hit else "cache miss"
    return (
        f"**Proposed** ({proposed_latency:.2f}s · {r2_tokens} tok · {hit}): "
        f"{proposed_answer}\n\n"
        f"&nbsp;&nbsp;↳ proposed no-cache: {no_cache_latency:.2f}s\n\n"
        f"**Vanilla** ({vanilla_latency:.2f}s · {vanilla_tokens} tok): "
        f"{vanilla_answer}\n\n"
        f"**Speedup vs vanilla:** {speedup:.1f}×"
    )
