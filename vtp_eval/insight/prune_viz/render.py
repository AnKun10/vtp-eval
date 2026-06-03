"""Pure masked-patch rendering: kept patches highlighted, pruned dimmed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

GRID = 24            # LLaVA-1.5 CLIP ViT-L/14-336 -> 24x24 patches
PATCH = 14           # 336 / 24


def patch_to_box(p: int, grid: int = GRID, patch: int = PATCH):
    """Patch index -> (x0, y0, x1, y1) pixel box on the grid*patch image."""
    row, col = divmod(int(p), grid)
    x0, y0 = col * patch, row * patch
    return (x0, y0, x0 + patch, y0 + patch)


def keep_mask(keep_indices, grid: int = GRID) -> np.ndarray:
    """[K] patch indices -> [grid, grid] bool mask (True = kept)."""
    m = np.zeros((grid, grid), dtype=bool)
    for p in keep_indices:
        r, c = divmod(int(p), grid)
        m[r, c] = True
    return m


def mask_overlay(image, keep_indices, grid: int = GRID,
                 pruned_alpha: float = 0.25, tint=(0.55, 0.55, 0.55),
                 tint_strength: float = 0.30) -> np.ndarray:
    """RGBA float array: kept patch = opaque w/ light-gray tint, pruned = dimmed.

    image: PIL.Image (any size; mask is upsampled to it). Returns [H, W, 4] in
    [0, 1] suitable for matplotlib imshow.
    """
    from PIL import Image
    W, H = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0     # [H, W, 3]
    m = keep_mask(keep_indices, grid).astype(np.uint8) * 255
    mask = np.asarray(Image.fromarray(m).resize((W, H), Image.NEAREST)) > 127  # [H, W]
    out = rgb.copy()
    tint_arr = np.asarray(tint, dtype=np.float32)
    out[mask] = (1.0 - tint_strength) * out[mask] + tint_strength * tint_arr
    alpha = np.where(mask, 1.0, pruned_alpha).astype(np.float32)
    return np.dstack([out, alpha])


def plot_prune_row(image, panels, titles, out_path: Path, suptitle: str = "") -> None:
    """Save a 1x(1+len(panels)) figure: original + one panel per keep-set.

    panels: list of keep_indices arrays; titles: list of str (len == len(panels)).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 1 + len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=13)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, keep, title in zip(axes[1:], panels, titles):
        ax.imshow(mask_overlay(image, keep))
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
