"""Concentration metrics on per-(word, layer) attention distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(pwl, sinks, lyrs) -> pd.DataFrame:
    """Return a long-form DataFrame with columns
    ``token, depth, layer, entropy, max_share, top5pct_mass``.

    Sink token indices are zeroed before normalization to make depths
    comparable; otherwise a single outlier dominates every metric equally.
    """
    rows = []
    for w in pwl:
        for d in ("shallow", "middle", "deep"):
            v = pwl[w][d].astype(np.float64).copy()
            v[list(sinks)] = 0.0
            p = v / (v.sum() + 1e-12)
            k = max(1, int(0.05 * p.size))
            rows.append({
                "token": w,
                "depth": d,
                "layer": lyrs[d],
                "entropy": float(-np.sum(p * np.log(p + 1e-12))),
                "max_share": float(p.max()),
                "top5pct_mass": float(np.sort(p)[-k:].sum()),
            })
    return pd.DataFrame(rows)
