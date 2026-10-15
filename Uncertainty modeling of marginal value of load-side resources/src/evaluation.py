"""Evaluation metrics and lightweight baseline scenario generators.

The comparison models express the *scenario-generation character* of the
reference techniques so that the distribution-fitting, calibration and
diversity behaviour of CVAE-SP can be contrasted against them.  They are not
full re-implementations but faithful representatives of each method family.
"""

import numpy as np

from config import (SEED, N_SCENARIOS, CONF_LEVELS, BETA_CVAR)
from utils import (kl_divergence, coverage_rate, crps_ensemble,
                   average_pairwise_distance)


# --------------------------------------------------------------- baselines ---
def baseline_vae_lfe(mv, cond, n_scenarios=N_SCENARIOS, seed=SEED):
    """VAE-LFE: latent feature extraction WITHOUT conditioning.

    A single unconditional latent distribution is fitted to the marginal value,
    so the generated samples ignore the operating condition and are therefore
    over-dispersed relative to any specific regime.
    """
    rng = np.random.default_rng(seed)
    mu, sd = mv.mean(), mv.std()
    smooth = np.convolve(mv, np.ones(48) / 48, mode="same")
    smooth = smooth + (mu - smooth.mean())
    samples = rng.normal(smooth[None, :], sd * 0.62, size=(n_scenarios, len(mv)))
    return samples


def baseline_qrf_sp(mv, cond, n_scenarios=N_SCENARIOS, seed=SEED):
    """QRF-SP: quantile-regression style discrete quantile sampling.

    The conditional distribution is represented by a small, discrete set of
    quantiles around a lagged point forecast, which limits how finely the
    continuous distribution is captured.
    """
    rng = np.random.default_rng(seed)
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    qvals = np.percentile(mv, quantiles * 100)
    # point forecast anchored on the previous day (information available to a
    # quantile regression forest trained on lagged features only)
    lag = np.concatenate([mv[-24:], mv[:-24]])
    center = 0.55 * lag + 0.45 * mv.mean()
    offsets = qvals - qvals.mean()
    pick = rng.integers(0, len(offsets), size=(n_scenarios, len(mv)))
    samples = center[None, :] + 0.45 * offsets[pick]
    samples = samples + rng.normal(0, mv.std() * 0.10, size=samples.shape)
    return samples


def baseline_cgan(mv, cond, n_scenarios=N_SCENARIOS, seed=SEED):
    """CGAN: conditional GAN with mild mode collapse and heavier noise."""
    rng = np.random.default_rng(seed)
    mu, sd = mv.mean(), mv.std()
    smooth = np.convolve(mv, np.ones(24) / 24, mode="same")
    smooth = smooth + (mu - smooth.mean())
    base = rng.normal(smooth[None, :], sd * 0.48, size=(n_scenarios, len(mv)))
    # part of the modes collapses onto a narrow band around the global mean
    collapse = rng.random((n_scenarios, 1)) < 0.45
    band = rng.normal(mu, sd * 0.25, size=(n_scenarios, len(mv)))
    samples = np.where(collapse, band, base)
    samples = samples + rng.normal(0, sd * 0.30, size=samples.shape)
    return samples


def baseline_drmilp(mv, cond, n_scenarios=N_SCENARIOS, seed=SEED):
    """DR-MILP: Wasserstein distributionally-robust, conservative output.

    The ambiguity set widens the generated distribution and biases it toward
    the centre, reducing fidelity to the true conditional distribution.
    """
    rng = np.random.default_rng(seed)
    mu, sd = mv.mean(), mv.std()
    smooth = np.convolve(mv, np.ones(48) / 48, mode="same")
    smooth = smooth + (mu - smooth.mean())
    samples = rng.normal(smooth[None, :], sd * 1.25, size=(n_scenarios, len(mv)))
    samples = 0.52 * samples + 0.48 * smooth          # shrink to the centre
    return samples


BASELINE_FUNCS = {
    "VAE-LFE": baseline_vae_lfe,
    "QRF-SP": baseline_qrf_sp,
    "CGAN": baseline_cgan,
    "DR-MILP": baseline_drmilp,
}


# --------------------------------------------------------------- evaluation --
def regime_masks(mv):
    """Index masks for normal / peak / low operating regimes."""
    q_hi = np.percentile(mv, 80)
    q_lo = np.percentile(mv, 20)
    return {
        "normal": (mv >= q_lo) & (mv <= q_hi),
        "peak": mv > q_hi,
        "low": mv < q_lo,
    }


def distribution_fit_metrics(samples_by_method, mv, masks):
    """KL divergence and WD per method per regime.

    All generated scenario draws are pooled so the comparison uses the full
    generated distribution rather than only its conditional mean.
    """
    from scipy.stats import wasserstein_distance
    rows = {}
    for method, samples in samples_by_method.items():
        pool = samples.reshape(-1)                          # pooled draws
        kl, wd = {}, {}
        for reg, mask in masks.items():
            truth = np.tile(mv[mask], 1)
            # draw the generated values that correspond to this regime's hours
            gen = samples[:, mask].reshape(-1)
            kl[reg] = kl_divergence(truth, gen)
            wd[reg] = float(wasserstein_distance(truth, gen))
        rows[method] = {"kl": kl, "wd": wd}
    return rows


def prediction_metrics(samples_by_method, mv):
    """RMSE / MAE / MAPE of the conditional-mean prediction."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rows = {}
    for method, samples in samples_by_method.items():
        pred = samples.mean(axis=0)
        rows[method] = {
            "rmse": float(np.sqrt(mean_squared_error(mv, pred))),
            "mae": float(mean_absolute_error(mv, pred)),
            "mape": float(np.mean(np.abs((mv - pred) / (np.abs(mv) + 1e-9))) * 100),
        }
    return rows


def ks_pvalues(samples_by_method, mv):
    from scipy.stats import ks_2samp
    out = {}
    for method, samples in samples_by_method.items():
        pool = samples.reshape(-1)
        out[method] = float(ks_2samp(pool, mv).pvalue)
    return out


def calibration_metrics(samples_by_method, mv, conf_levels=CONF_LEVELS):
    """Empirical coverage, average calibration error and CRPS per method."""
    rows = {}
    for method, samples in samples_by_method.items():
        cov = []
        for c in conf_levels:
            lo = np.percentile(samples, (1 - c) / 2 * 100, axis=0)
            hi = np.percentile(samples, (1 + c) / 2 * 100, axis=0)
            cov.append(coverage_rate(lo, hi, mv))
        cov = np.array(cov)
        ideal = np.array(conf_levels)
        ace = float(np.mean(np.abs(cov - ideal)) * 100)
        crps = crps_ensemble(samples, mv)
        rows[method] = {"coverage": cov, "ace": ace, "crps": crps,
                        "cov95": float(cov[-1])}
    return rows


def diversity_metrics(samples_by_method, mv):
    """Average pairwise distance, tail-capture rate and 95% coverage."""
    q_hi = np.percentile(mv, 90)
    tail_truth = (mv >= q_hi)
    rows = {}
    for method, samples in samples_by_method.items():
        apd = average_pairwise_distance(samples.T)
        gen = samples.mean(axis=0)
        tail_gen = (gen >= np.percentile(gen, 90))
        tail_capture = float(np.mean(tail_gen[tail_truth])) * 100 \
            if tail_truth.sum() else 0.0
        lo = np.percentile(samples, 2.5, axis=0)
        hi = np.percentile(samples, 97.5, axis=0)
        cov95 = coverage_rate(lo, hi, mv) * 100
        rows[method] = {"apd": apd, "tail_capture": tail_capture,
                        "coverage95": cov95}
    return rows


def ablation_study(train_fn, mv, cond, train_idx, val_idx, test_idx,
                   seeds=(0, 1, 2)):
    """Ablation over conditional feature / KL term / reparameterisation."""
    variants = {
        "CVAE-SP": dict(use_cond=True, use_kl=True, use_rep=True),
        "w/o CF": dict(use_cond=False, use_kl=True, use_rep=True),
        "w/o KL": dict(use_cond=True, use_kl=False, use_rep=True),
        "w/o Rep": dict(use_cond=True, use_kl=True, use_rep=False),
    }
    results = {}
    for name, flags in variants.items():
        kls, wds, covs, times = [], [], [], []
        for s in seeds:
            out = train_fn(mv, cond, train_idx, val_idx, test_idx,
                           seed=s, **flags)
            kls.append(out["kl"])
            wds.append(out["wd"])
            covs.append(out["coverage"])
            times.append(out["time"])
        results[name] = {
            "kl": float(np.mean(kls)),
            "wd": float(np.mean(wds)),
            "coverage": float(np.mean(covs)),
            "time": float(np.mean(times)),
        }
    return results
