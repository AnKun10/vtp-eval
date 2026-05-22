"""Target-word position discovery and query construction."""

from __future__ import annotations


def find_word_positions(input_ids, tokenizer, word: str,
                        img_start: int, img_end: int):
    """Find a contiguous token-id sequence matching ``word`` in the text region.

    The image-token span [img_start, img_end] is excluded. Tries the leading-
    space form ``" word"`` first (mid-sentence usage), then the raw form.

    Returns the list of positions, or None if no match found.
    """
    ids = input_ids.tolist()
    n = len(ids)
    for prefix in (" ", ""):
        target = tokenizer.encode(prefix + word, add_special_tokens=False)
        m = len(target)
        if m == 0:
            continue
        for pos in range(n - m + 1):
            # Skip if window intersects the image-token span
            if not (pos + m - 1 < img_start or pos > img_end):
                continue
            if ids[pos:pos + m] == target:
                return list(range(pos, pos + m))
    return None


def build_default_query(words):
    """Compose a natural English question containing each word verbatim."""
    if len(words) == 1:
        return f"Is there a {words[0]} in this image?"
    if len(words) == 2:
        return f"Are there both {words[0]} and {words[1]} in this image?"
    head = ", ".join(words[:-1])
    return f"Are there {head}, and {words[-1]} in this image?"
