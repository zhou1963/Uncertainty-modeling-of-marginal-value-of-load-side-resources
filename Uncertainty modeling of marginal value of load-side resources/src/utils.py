"""Utility functions: shared plotting style, metrics and reproducibility."""

import os
import numpy as np

from config import FIG_DIR

METHOD_COLORS = {
    "CVAE-SP": "#1f5fa8",
    "VAE-LFE": "#2ca02c",
    "QRF-SP": "#ff8c00",
    "CGAN": "#d62728",
    "DR-MILP": "#7b2fbe",
}


def setup_matplotlib():
    """Configure a publication-quality matplotlib style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
    })
    return plt


def save_fig(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    print(f"[figure] saved -> {path}")
    return path


# ------------------------------------------------------------------ metrics --
def kl_divergence(p_samples, q_samples, bins=60, eps=1e-9):
    """Symmetric KL divergence between two sample sets via shared histograms."""
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    p_hist, edges = np.histogram(p_samples, bins=bins, range=(lo, hi), density=False)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(lo, hi), density=False)
    p = p_hist / (p_hist.sum() + eps) + eps
    q = q_hist / (q_hist.sum() + eps) + eps
    return float(np.sum(p * np.log(p / q)))


def ks_statistic(samples, reference):
    from scipy.stats import ks_2samp
    return ks_2samp(samples, reference)


def coverage_rate(lower, upper, truth):
    return float(np.mean((truth >= lower) & (truth <= upper)))


def crps_ensemble(samples, truth):
    """Continuous ranked probability score for an ensemble (samples: [K, N])."""
    samples = np.asarray(samples)
    K, N = samples.shape
    term1 = np.mean(np.abs(samples - truth[None, :]), axis=0)
    term2 = 0.5 * np.mean(
        np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1))
    return float(np.mean(term1 - term2))


def average_pairwise_distance(samples_2d):
    """Mean pairwise Euclidean distance across generated scenarios [N, D]."""
    x = np.asarray(samples_2d)
    n = x.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(n, 400), replace=False)
    x = x[idx]
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(x), k=1)
    return float(dist[iu].mean())


def sharp_ratio(expected_return, std_return, risk_free=0.0):
    return float((expected_return - risk_free) / (std_return + 1e-9))
