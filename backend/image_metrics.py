"""
Classical pixel-level and perceptual image-comparison metrics, computed
alongside the model's learned change map. These are cheap, well-understood
signals that complement the model output:

- MSE / PSNR: raw pixel-difference baseline, useful as a sanity check
  against the model's probability map.
- SSIM (+ a per-tile SSIM map): structural similarity, correlates with
  real structural change rather than noise or lighting drift.
- Histogram distance: flags when the two images have very different
  color/illumination profiles (season, sensor, time-of-day), which can
  otherwise be mistaken for "change" by naive methods.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim


def _as_uint8_rgb(rgb_float01: np.ndarray) -> np.ndarray:
    return np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)


def compute_mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Mean squared error between two same-shape [0,1] float images."""
    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(mse: float, max_pixel_value: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB) from an already-computed MSE."""
    if mse <= 1e-12:
        return float("inf")
    return float(20 * np.log10(max_pixel_value) - 10 * np.log10(mse))


def compute_ssim(image_a: np.ndarray, image_b: np.ndarray):
    """Global SSIM score + a full-resolution per-pixel SSIM map.

    Images are expected as [H, W, C] float arrays in [0, 1].
    """
    gray_a = image_a.mean(axis=-1) if image_a.ndim == 3 else image_a
    gray_b = image_b.mean(axis=-1) if image_b.ndim == 3 else image_b

    score, ssim_map = sk_ssim(gray_a, gray_b, data_range=1.0, full=True)
    return float(score), ssim_map


def compute_histogram_distance(image_a: np.ndarray, image_b: np.ndarray, bins=64):
    """Chi-square distance between per-channel color histograms.

    Images are [H, W, C] float arrays in [0, 1]. Returns the mean
    chi-square distance across channels (0 = identical color profile,
    higher = more different illumination/season/sensor profile).
    """
    channels = image_a.shape[-1] if image_a.ndim == 3 else 1
    distances = []

    for c in range(channels):
        a = image_a[..., c] if channels > 1 else image_a
        b = image_b[..., c] if channels > 1 else image_b

        hist_a, _ = np.histogram(a, bins=bins, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=True)

        eps = 1e-10
        chi_square = 0.5 * np.sum(
            ((hist_a - hist_b) ** 2) / (hist_a + hist_b + eps)
        )
        distances.append(float(chi_square))

    return float(np.mean(distances))


def save_ssim_map(ssim_map: np.ndarray, output_path):
    """Render the per-pixel SSIM map as a heatmap (dark = more structural change)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(ssim_map, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Structural similarity (SSIM) map")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="SSIM (1 = identical structure)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def compute_classical_metrics(earlier_rgb01: np.ndarray, later_rgb01: np.ndarray, output_dir, stem):
    """Run all classical metrics on a resized/aligned earlier/later RGB pair
    (both expected as float [H, W, 3] arrays in [0, 1], same shape).

    Returns (stats_dict, image_paths_dict).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mse = compute_mse(earlier_rgb01, later_rgb01)
    psnr = compute_psnr(mse, max_pixel_value=1.0)
    ssim_score, ssim_map = compute_ssim(earlier_rgb01, later_rgb01)
    hist_distance = compute_histogram_distance(earlier_rgb01, later_rgb01)

    ssim_path = output_dir / f"{stem}_ssim_map.png"
    save_ssim_map(ssim_map, ssim_path)

    # A simple heuristic flag: high histogram distance suggests the two
    # images differ a lot in overall color/illumination (season, sensor,
    # time of day), which can inflate naive change estimates.
    illumination_warning = hist_distance > 0.15

    stats = {
        "mse": mse,
        "psnr_db": psnr if psnr != float("inf") else None,
        "ssim_score": ssim_score,
        "histogram_distance": hist_distance,
        "illumination_mismatch_warning": illumination_warning,
    }

    images = {"ssim_map": ssim_path}

    return stats, images
