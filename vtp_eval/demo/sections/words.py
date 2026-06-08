# vtp_eval/demo/sections/words.py
"""Pure helper: pick content words from a benchmark question for the TVA tab's
target-word checkboxes. No gradio/torch imports (CPU-testable)."""
from __future__ import annotations

# Generic / non-visual words that make poor attention targets.
STOPWORDS = {
    "is", "are", "was", "were", "there", "the", "a", "an", "of", "in", "on",
    "at", "to", "and", "or", "with", "for", "this", "that", "these", "those",
    "what", "which", "who", "how", "do", "does", "did", "it", "its", "you",
    "your", "any", "many", "much", "some", "image", "photo", "picture", "scene",
}


def content_words(question: str, min_len: int = 3) -> list:
    """Return the distinct content words of ``question`` (original case preserved
    so they match the prompt verbatim), dropping short words and STOPWORDS.
    Deduplication is case-insensitive."""
    seen, out = set(), []
    for raw in question.split():
        w = raw.strip(".,?!\"'()[]:;")
        wl = w.lower()
        if len(wl) >= min_len and wl not in STOPWORDS and wl not in seen:
            seen.add(wl)
            out.append(w)
    return out
