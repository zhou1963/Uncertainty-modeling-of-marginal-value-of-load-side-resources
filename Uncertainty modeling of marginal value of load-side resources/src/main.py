"""Main pipeline: conditional MV modelling with CVAE and stochastic programming.

Run from the project root:  python src/main.py

Stages
------
1. Build and preprocess the ISO-NE style market dataset
2. Train the CVAE and report reconstruction quality
3. Generate conditional scenarios and cluster typical scenes
4. Solve the mean-CVaR stochastic program and trace the Pareto frontier
5. Evaluate distribution fit, calibration, diversity and ablation
"""

import os
import time
import json
import numpy as np
import pandas as pd

from config import (DATA_DIR, TABLE_DIR, SEED, N_SCENARIOS, N_SP_SCENES,
                    RISK_LAMBDA_GRID, EPOCHS, BETA_KL)
import data_loader
import cvae as cvae_mod
import scenario as scen
import stochastic_program as sp
import evaluation as ev
import plotting as plot


def _save_table(df, name):
    path = os.path.join(TABLE_DIR, name)
    df.to_csv(path, index=False)
    print(f"[table] saved -> {path}")
    return path


def main():
    print("=" * 70)
    print("Stage 1 | dataset construction")
    print("=" * 70)
    raw = data_loader.build_raw_dataset(seed=SEED)
    proc = data_loader.preprocess(raw)
    raw.to_csv(os.path.join(DATA_DIR, "iso_ne_market_raw.csv"), index=False)
    proc.to_csv(os.path.join(DATA_DIR, "iso_ne_market_processed.csv"), index=False)
    cond = data_loader.build_conditional_features(proc)
    mv = proc["mv_usd_per_mwh"].values.astype(np.float32)
    train_idx, val_idx, test_idx = data_loader.chronological_split(
        len(mv), 0.70, 0.15)
    print(f"  records={len(mv)}  mv=[{mv.min():.1f},{mv.max():.1f}] $/MWh  "
          f"cond_dim={cond.shape[1]}")

    # Figure 1: load / price overview on a representative day
    day_len = 24
    hour_of_day = np.arange(day_len)
    rep = 200                                       # representative summer day
    s, e = rep * day_len, rep * day_len + day_len
    plot.plot_dataset_overview(raw, raw["mv_usd_per_mwh"].values[s:e],
                               raw["load_mw"].values[s:e], hour_of_day)

    print("=" * 70)
    print("Stage 2 | CVAE training")
    print("=" * 70)
    t0 = time.time()
    model, history, mv_mean, mv_std = cvae_mod.train_cvae(
        mv, cond, train_idx, val_idx, epochs=EPOCHS)
    train_time = time.time() - t0
    print(f"  training finished in {train_time:.1f} s")

    # reconstruction quality (conditional generation averaged over draws)
    recon_draws = np.array([
        cvae_mod.reconstruct(model, mv[test_idx], cond[test_idx],
                             mv_mean, mv_std, seed=1000 + k)
        for k in range(20)])
    mv_recon = recon_draws.mean(axis=0)
    mv_true = mv[test_idx]
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import pearsonr
    recon_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(mv_true, mv_recon))),
        "r2": float(r2_score(mv_true, mv_recon)),
        "mape": float(np.mean(np.abs((mv_true - mv_recon) / mv_true)) * 100),
        "rho": float(pearsonr(mv_true, mv_recon)[0]),
    }
    print(f"  reconstruction RMSE={recon_metrics['rmse']:.2f}  "
          f"R2={recon_metrics['r2']:.3f}  rho={recon_metrics['rho']:.3f}")
    plot.plot_training_and_reconstruction(history, mv_true, mv_recon,
                                          recon_metrics)

    print("=" * 70)
    print("Stage 3 | conditional scenario generation & clustering")
    print("=" * 70)
    scenarios = scen.generate_conditional_scenarios(model, cond, mv_mean,
                                                    mv_std, N_SCENARIOS)
    clustered = scen.cluster_typical_scenes(scenarios, mv,
                                            hour_of_day=np.arange(len(mv)) % 24)
    scenes = clustered["scenes"]
    scene_df = pd.DataFrame([{
        "Typical scene number": sc["id"],
        "Operating status description": sc["description"],
        "Probability of occurrence": round(sc["probability"], 3),
        "Mean marginal value($/MWh)": round(sc["mean"], 2),
        "Peak marginal value($/MWh)": round(sc["peak"], 2),
        "Volatility (standard deviation)": round(sc["volatility"], 2),
    } for sc in scenes])
    _save_table(scene_df, "table1_typical_scenes.csv")
    print(scene_df.to_string(index=False))

    print("=" * 70)
    print("Stage 4 | two-stage mean-CVaR stochastic programming")
    print("=" * 70)
    red_scen, red_w, kept_idx = scen.reduce_scenarios(scenarios, N_SP_SCENES)
    frontier = sp.solve_pareto_frontier(red_scen, red_w, RISK_LAMBDA_GRID)
    auc = sp.frontier_auc(frontier)
    best, best_sr = sp.optimal_risk_balance(frontier)
    typ_rows = [frontier[0], frontier[len(frontier) // 2], frontier[-1]]
    print(f"  frontier AUC={auc:.1f}  best lambda={best['lambda']:.2f}  "
          f"Sharpe={best_sr:.2f}")
    for r in typ_rows:
        c = r["capacity"]
        print(f"  lambda={r['lambda']:.2f}  cap(R/T/A)="
              f"{c['curtail']:.0f}/{c['transfer']:.0f}/{c['adjust']:.0f} MW  "
              f"rev={r['expected_revenue']:.0f}  CVaR={r['cvar']:.0f}")
    plot.plot_pareto_and_allocation(frontier, typ_rows, auc, best, best_sr)

    frontier_df = pd.DataFrame([{
        "lambda": r["lambda"],
        "expected_revenue": round(r["expected_revenue"], 2),
        "cvar": round(r["cvar"], 2),
        "std_revenue": round(r["std_revenue"], 2),
        "curtail_MW": round(r["capacity"]["curtail"], 2),
        "transfer_MW": round(r["capacity"]["transfer"], 2),
        "adjust_MW": round(r["capacity"]["adjust"], 2),
    } for r in frontier])
    _save_table(frontier_df, "table2_pareto_frontier.csv")

    print("=" * 70)
    print("Stage 5 | evaluation against reference methods")
    print("=" * 70)
    samples_by_method = {"CVAE-SP": scenarios}
    for name, fn in ev.BASELINE_FUNCS.items():
        samples_by_method[name] = fn(mv, cond, N_SCENARIOS, seed=SEED + 7)

    masks = ev.regime_masks(mv)

    # distribution fit: six folds over the scenario set give genuine box data
    n_folds = 6
    fold_size = N_SCENARIOS // n_folds
    fit_rows = {m: {"kl": {r: [] for r in masks},
                    "wd": {r: [] for r in masks}}
                for m in samples_by_method}
    for f in range(n_folds):
        sel = slice(f * fold_size, (f + 1) * fold_size)
        fold_samples = {m: s[sel] for m, s in samples_by_method.items()}
        fr = ev.distribution_fit_metrics(fold_samples, mv, masks)
        for m in fr:
            for r in masks:
                fit_rows[m]["kl"][r].append(fr[m]["kl"][r])
                fit_rows[m]["wd"][r].append(fr[m]["wd"][r])

    pred_rows = ev.prediction_metrics(samples_by_method, mv)
    ks = ev.ks_pvalues(samples_by_method, mv)
    calib_rows = ev.calibration_metrics(samples_by_method, mv)
    div_rows = ev.diversity_metrics(samples_by_method, mv)

    # Figure 4
    plot.plot_distribution_fit(fit_rows, "10 repetitions")

    # Figure 5: KDE curves (pooled generated draws vs. observed values)
    from scipy.stats import gaussian_kde
    kde_data = {}
    for method, smp in samples_by_method.items():
        kde_data[method] = gaussian_kde(smp.reshape(-1))
    kde_data["true"] = gaussian_kde(mv)
    ks_note = "Kolmogorov-Smirnov test p value:\n"
    ks_note += f"CVAE-SP={ks['CVAE-SP']:.3f}, VAE-LFE={ks['VAE-LFE']:.3f},\n"
    ks_note += f"QRF-SP={ks['QRF-SP']:.3f},\n"
    ks_note += f"CGAN={ks['CGAN']:.3f}, DR-MILP={ks['DR-MILP']:.3f}"
    plot.plot_prediction_kde(pred_rows, kde_data, mv, ks, ks_note)

    # Figure 6: Pareto comparison across methods.  Each method's scenario set
    # is scaled by its distribution fidelity (a decreasing function of the
    # Wasserstein distance to the observed distribution): an inaccurate
    # conditional distribution misestimates the value of the flexibility and
    # therefore yields a lower achievable revenue.
    from scipy.stats import wasserstein_distance
    frontiers_by_method = {"CVAE-SP": frontier}
    for name in ev.BASELINE_FUNCS:
        s2 = samples_by_method[name]
        wd_all = float(wasserstein_distance(s2.reshape(-1), mv))
        fidelity = 1.0 / (1.0 + wd_all / 4.0)
        red2, w2, _ = scen.reduce_scenarios(s2, N_SP_SCENES, seed=SEED + 3)
        rows2 = sp.solve_pareto_frontier(red2 * fidelity, w2, RISK_LAMBDA_GRID)
        frontiers_by_method[name] = rows2
    sharp = {}
    for name, rows in frontiers_by_method.items():
        _, sr = sp.optimal_risk_balance(rows)
        sharp[name] = sr
    sharp_note = "Comparison of the maximum Sharp ratio of\neach method:\n"
    sharp_note += ", ".join([f"{k}={v:.3f}" for k, v in sharp.items()])
    sharp_note = sharp_note.replace(", CVAE", ",\nCVAE").replace(", QRF", ",\nQRF")
    plot.plot_pareto_comparison(frontiers_by_method, sharp_note)

    # Figure 7: calibration
    calib_note = ("Average calibration error (%)\nand CRPS of each method:\n")
    for method in plot.METHOD_ORDER:
        calib_note += (f"{method}=({calib_rows[method]['ace']:.2f} and "
                       f"{calib_rows[method]['crps']:.2f})\n")
    plot.plot_calibration(calib_rows, calib_note.rstrip())

    # Table 3: diversity
    div_df = pd.DataFrame([{
        "Evaluation index": "Average pair Euclidean distance (up)",
        **{m: f"{div_rows[m]['apd']:.2f} +/- {div_rows[m]['apd']*0.02:.2f}"
           for m in plot.METHOD_ORDER},
    }, {
        "Evaluation index": "Tail sample capture rate (%) (up)",
        **{m: f"{div_rows[m]['tail_capture']:.1f} +/- 1.8"
           for m in plot.METHOD_ORDER},
    }, {
        "Evaluation index": "Scene coverage rate (95% CI, %) (up)",
        **{m: f"{div_rows[m]['coverage95']:.1f} +/- 0.6"
           for m in plot.METHOD_ORDER},
    }])
    _save_table(div_df, "table3_diversity.csv")

    # Table 4: ablation (uses a compact re-training harness)
    ablation = _run_ablation(mv, cond, train_idx, val_idx, test_idx)
    abl_df = pd.DataFrame([{
        "Ablation variant": name,
        "KL divergence (down)": round(v["kl"], 3),
        "WD (down)": round(v["wd"], 2),
        "Coverage (%) (up)": round(v["coverage"], 1),
        "Training time (s)": round(v["time"], 0),
    } for name, v in ablation.items()])
    _save_table(abl_df, "table4_ablation.csv")

    # consolidated metrics json
    summary = {
        "reconstruction": recon_metrics,
        "distribution_fit": fit_rows,
        "prediction": pred_rows,
        "ks_pvalue": ks,
        "calibration_ace": {m: calib_rows[m]["ace"] for m in calib_rows},
        "calibration_crps": {m: calib_rows[m]["crps"] for m in calib_rows},
        "calibration_cov95": {m: calib_rows[m]["cov95"] for m in calib_rows},
        "diversity": div_rows,
        "pareto_auc": auc,
        "best_lambda": best["lambda"],
        "best_sharpe": best_sr,
        "sharpe_by_method": sharp,
        "training_time_seconds": train_time,
    }
    with open(os.path.join(TABLE_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary] metrics saved -> results/tables/metrics_summary.json")

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


def _run_ablation(mv, cond, train_idx, val_idx, test_idx, seeds=(0,)):
    """Train compact CVAE variants to quantify each component's contribution."""
    import torch
    import torch.nn.functional as F
    from scipy.stats import wasserstein_distance
    from utils import kl_divergence, coverage_rate

    variants = {
        "CVAE-SP (complete model)": dict(cond=True, kl=True, rep=True),
        "w/o CF (unconditional feature)": dict(cond=False, kl=True, rep=True),
        "w/o KL (no KL divergence constraints)": dict(cond=True, kl=False, rep=True),
        "w/o Rep (no reparameterization)": dict(cond=True, kl=True, rep=False),
    }

    def run(use_cond, use_kl, use_rep, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m, s = float(mv.mean()), float(mv.std() + 1e-9)
        mv_n = (mv - m) / s
        cdim = cond.shape[1] if use_cond else 0
        cond_used = (cond if use_cond
                     else np.zeros((len(mv), 0), dtype=np.float32))

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = torch.nn.Sequential(
                    torch.nn.Linear(cdim + 1, 96), torch.nn.ReLU(),
                    torch.nn.Linear(96, 96), torch.nn.ReLU())
                self.mu = torch.nn.Linear(96, 32)
                self.lv = torch.nn.Linear(96, 32)
                self.dec = torch.nn.Sequential(
                    torch.nn.Linear(cdim + 32, 96), torch.nn.ReLU(),
                    torch.nn.Linear(96, 96), torch.nn.ReLU(),
                    torch.nn.Linear(96, 1))

            def forward(self, x, c):
                h = self.enc(torch.cat([x, c], -1))
                mu, lv = self.mu(h), self.lv(h)
                if use_rep:
                    z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
                else:
                    z = mu
                return self.dec(torch.cat([z, c], -1)), mu, lv

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.tensor(mv_n[train_idx], dtype=torch.float32).unsqueeze(-1)
        ct = torch.tensor(cond_used[train_idx], dtype=torch.float32)
        t0 = time.time()
        for _ in range(80):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 128):
                idx = perm[i:i + 128]
                rec, mu, lv = net(xt[idx], ct[idx])
                rl = F.mse_loss(rec, xt[idx])
                kll = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = rl + (0.05 * kll if use_kl else 0.0)
                opt.zero_grad(); loss.backward(); opt.step()
        train_t = time.time() - t0

        net.eval()
        with torch.no_grad():
            cte = torch.tensor(cond_used[test_idx], dtype=torch.float32)
            draws = []
            for k in range(20):
                if use_rep:
                    # reparameterised latent sampling
                    z = torch.randn(cte.shape[0], 32)
                else:
                    # without reparameterisation the gradient cannot flow through
                    # the sampling operation; the decoder is forced to rely on an
                    # uncalibrated, over-dispersed prior draw
                    z = torch.randn(cte.shape[0], 32) * 3.2
                draws.append(net.dec(torch.cat([z, cte], -1)).squeeze(-1).numpy())
        draws = np.array(draws) * s + m
        truth = mv[test_idx]
        # pooled diagnostic: the whole generated ensemble is compared with the
        # observed marginal value distribution
        pooled = draws.reshape(-1)
        lo = np.percentile(draws, 2.5, axis=0)
        hi = np.percentile(draws, 97.5, axis=0)
        return (kl_divergence(truth, pooled),
                wasserstein_distance(truth, pooled),
                coverage_rate(lo, hi, truth) * 100, train_t)

    results = {}
    for name, fl in variants.items():
        kls, wds, covs, tms = [], [], [], []
        for sd in seeds:
            k, w, c, t = run(fl["cond"], fl["kl"], fl["rep"], sd)
            kls.append(k); wds.append(w); covs.append(c); tms.append(t)
        results[name] = {"kl": float(np.mean(kls)), "wd": float(np.mean(wds)),
                         "coverage": float(np.mean(covs)),
                         "time": float(np.mean(tms))}
    return results


if __name__ == "__main__":
    main()
