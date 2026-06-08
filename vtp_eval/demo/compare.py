# vtp_eval/demo/compare.py
"""Pure formatting of the per-turn proposed-vs-vanilla comparison bubble.

No gradio/torch imports, so it unit-tests on CPU.
"""
from __future__ import annotations


def format_comparison(proposed_answer: str, proposed_latency: float,
                      cache_hit: bool, r2_tokens: int,
                      no_cache_latency: float | None,
                      vanilla_answer: str, vanilla_latency: float,
                      vanilla_tokens: int = 576) -> str:
    """Build the Markdown assistant bubble comparing proposed vs vanilla.

    Shows the proposed answer/latency/token-count/cache state, the vanilla
    answer/latency/token-count, and the speedup (vanilla / proposed, guarded
    against a 0.0s proposed latency). The proposed no-cache line is shown ONLY
    when ``no_cache_latency`` is given (i.e. on a cache HIT, where recomputing
    without the cache is a meaningful contrast and reveals what the cache saved).
    On a cache MISS the caller passes ``None`` and the line is omitted, since the
    no-cache run would do identical work to the (miss) proposed run.
    """
    speedup = (vanilla_latency / proposed_latency) if proposed_latency > 0 else 0.0
    hit = "CACHE HIT" if cache_hit else "cache miss"
    lines = [
        f"**Proposed** ({proposed_latency:.2f}s · {r2_tokens} tok · {hit}): "
        f"{proposed_answer}",
    ]
    if no_cache_latency is not None:
        saved = no_cache_latency - proposed_latency
        lines.append(f"&nbsp;&nbsp;↳ no-cache: {no_cache_latency:.2f}s "
                     f"(cache saved {saved:.2f}s)")
    lines.append(f"**Vanilla** ({vanilla_latency:.2f}s · {vanilla_tokens} tok): "
                 f"{vanilla_answer}")
    lines.append(f"**Speedup vs vanilla:** {speedup:.1f}×")
    return "\n\n".join(lines)
