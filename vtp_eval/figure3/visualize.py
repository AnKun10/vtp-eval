"""Heatmap-overlay and concentration-bar visualizations for Figure 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def overlay_from_vec(v576, sinks, target_size, grid: int, p_clip: float = 99.0):
    """Convert a 576-dim attention vector to an upsampled, sink-masked heatmap.

    Sink token indices are zeroed before percentile-clipping so that one outlier
    pixel doesn't dominate the min-max normalization (LLaVA-1.5 has known
    attention sinks).
    """
    from PIL import Image
    x = v576.astype(np.float32).copy()
    if sinks:
        x[list(sinks)] = 0.0
    a = x.reshape(grid, grid)
    a = np.clip(a, 0.0, np.percentile(a, p_clip))
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    pil = Image.fromarray((a * 255).astype(np.uint8)).resize(target_size, Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0


def plot_heatmap_grid(pwl, image, sinks, grid: int, lyrs, query: str,
                      out_path: Path) -> None:
    """Save a (#words x 3) figure: rows = target words, cols = shallow/mid/deep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    W, H = image.size
    words = list(pwl)
    depths = ["shallow", "middle", "deep"]
    fig, axes = plt.subplots(len(words), 3, figsize=(12, 4 * len(words)))
    if len(words) == 1:
        axes = np.array([axes])
    cmap = cm.get_cmap("jet")
    for r, w in enumerate(words):
        for c, d in enumerate(depths):
            ax = axes[r, c]
            hm = overlay_from_vec(pwl[w][d], sinks, (W, H), grid)
            rgba = cmap(hm)
            rgba[..., 3] = 0.55 * hm + 0.10
            ax.imshow(image)
            ax.imshow(rgba)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{d.capitalize()} (layer {lyrs[d]})", fontsize=13)
            if c == 0:
                ax.set_ylabel(f"'{w}'", fontsize=13)
    fig.suptitle(f'Figure 3 reproduction  -  query: "{query}"',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(df, lyrs, out_path: Path) -> None:
    """Save a 1x2 bar chart: entropy + top-5% mass, grouped by depth, hue=token."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depths = ["shallow", "middle", "deep"]
    words = list(df["token"].unique())
    x = np.arange(3)
    wb = 0.8 / max(1, len(words))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, w in enumerate(words):
        sub = df[df["token"] == w].set_index("depth").loc[depths]
        a1.bar(x + (i - (len(words) - 1) / 2) * wb, sub["entropy"], wb,
               label=f"'{w}'")
        a2.bar(x + (i - (len(words) - 1) / 2) * wb, sub["top5pct_mass"], wb,
               label=f"'{w}'")
    a1.axhline(np.log(576), ls="--", color="gray", lw=1, label="uniform log(576)")
    a1.set_xticks(x); a1.set_xticklabels([f"{d}\nlayer {lyrs[d]}" for d in depths])
    a1.set_ylabel("Entropy"); a1.set_title("Lower = more focused"); a1.legend()
    a2.set_xticks(x); a2.set_xticklabels([f"{d}\nlayer {lyrs[d]}" for d in depths])
    a2.set_ylabel("Top-5% mass"); a2.set_title("Higher = more focused"); a2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
