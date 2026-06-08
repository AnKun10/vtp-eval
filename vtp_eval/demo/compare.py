# vtp_eval/demo/compare.py
"""Pure formatting of the per-turn proposed-vs-vanilla comparison bubble.

No gradio/torch imports, so it unit-tests on CPU.
"""
from __future__ import annotations


def format_comparison(proposed_answer: str, proposed_latency: float,
                      cache_hit: bool, r2_tokens: int,
                      vision_encode_ms: float,
                      vanilla_answer: str, vanilla_latency: float,
                      vanilla_tokens: int = 576) -> str:
    """Build the Markdown assistant bubble comparing proposed vs vanilla.

    Shows the proposed answer/latency/token-count/cache state, the vanilla
    answer/latency/token-count, and the speedup (guarded against a 0.0s proposed
    latency).

    The retain-token cache's wall-clock benefit is the vision-encode + R1
    selection it skips on a repeat image — a small, DETERMINISTIC amount
    (``vision_encode_ms``, measured once). The LLM forward over the kept tokens
    runs every turn regardless, so a noisy turn-vs-turn wall-clock diff can even
    go negative. We therefore report the deterministic skipped time on a HIT
    instead of subtracting two full (noisy) generations.
    """
    speedup = (vanilla_latency / proposed_latency) if proposed_latency > 0 else 0.0
    hit = "CACHE HIT" if cache_hit else "cache miss"
    lines = [
        f"**Proposed** ({proposed_latency:.2f}s · {r2_tokens} tok · {hit}): "
        f"{proposed_answer}",
    ]
    if cache_hit:
        lines.append(
            f"&nbsp;&nbsp;↳ reused cached R1 → ~{vision_encode_ms:.0f}ms "
            f"vision-encode saved (the LLM forward over {r2_tokens} tokens still "
            "runs each turn)")
    else:
        lines.append("&nbsp;&nbsp;↳ first turn for this image — building the R1 cache")
    lines.append(f"**Vanilla** ({vanilla_latency:.2f}s · {vanilla_tokens} tok): "
                 f"{vanilla_answer}")
    lines.append(f"**Speedup vs vanilla:** {speedup:.1f}×")
    return "\n\n".join(lines)
