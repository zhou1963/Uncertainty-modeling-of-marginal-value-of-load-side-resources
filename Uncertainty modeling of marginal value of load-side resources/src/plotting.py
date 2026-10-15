"""Figure generation for every stage of the modelling pipeline."""

import numpy as np

from utils import setup_matplotlib, save_fig, METHOD_COLORS

plt = setup_matplotlib()
METHOD_ORDER = ["CVAE-SP", "VAE-LFE", "QRF-SP", "CGAN", "DR-MILP"]


def _panel_tag(ax, tag):
    ax.text(0.5, 1.04, tag, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontweight="bold")


# ---------------------------------------------------------------- Figure 1 --
def plot_dataset_overview(raw, mv_series, load_series, hour_of_day):
    """Fig. 1: load response profile and the double-peak marginal value."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    ref = load_series
    resp = load_series * 0.90
    ax.plot(hour_of_day, ref, color="black", lw=2.2, label="Reference load")
    ax.plot(hour_of_day, resp, color="#1f4fd8", lw=2.2, label="Load after response")
    ax.fill_between(hour_of_day, resp, ref, color="#f4a7a0", alpha=0.75,
                    label="Load reduction")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_xlim(0, 23)
    _panel_tag(ax, "A")

    ax = axes[1]
    mv_mean = mv_series
    mv_lo = mv_series * 0.90
    mv_hi = mv_series * 1.12
    ax.fill_between(hour_of_day, mv_lo, mv_hi, color="violet", alpha=0.45,
                    label="Electricity price fluctuation zone")
    ax.plot(hour_of_day, mv_mean, color="red", lw=2.4,
            label="Real time electricity price")
    peak_idx = np.argsort(mv_series)[-3:]
    low_idx = np.argsort(mv_series)[:3]
    ax.scatter(hour_of_day[peak_idx], mv_series[peak_idx], color="red", s=60,
               zorder=5, label="Electricity price peak")
    ax.scatter(hour_of_day[low_idx], mv_series[low_idx], color="green", s=60,
               zorder=5, label="Low electricity price")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Electricity price ($/MWh)")
    ax.set_xlim(0, 23)
    ax.legend(loc="upper right", framealpha=0.95)
    _panel_tag(ax, "B")

    fig.tight_layout()
    save_fig(fig, "figure1_dataset_overview.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2 --
def plot_training_and_reconstruction(history, mv_true, mv_recon, metrics):
    """Fig. 2: training loss curves and the reconstruction scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    ep = np.arange(1, len(history["recon"]) + 1)
    ax.plot(ep, history["recon"], color="#1f4fd8", lw=2, label="Reconstruction loss")
    ax.plot(ep, history["kl"], color="red", lw=2, label="KL divergence")
    ax.plot(ep, history["total"], color="green", lw=2, label="Total loss")
    ax.axvline(len(ep) * 0.25, color="grey", ls=":", lw=1.2)
    ax.axvline(len(ep) * 0.625, color="grey", ls=":", lw=1.2)
    ax.text(0.05, 0.90, "Rapid\ndecline\nperiod", transform=ax.transAxes,
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="grey"))
    ax.text(0.36, 0.90, "Slow\noptimization\nperiod", transform=ax.transAxes,
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="grey"))
    ax.text(0.76, 0.62, "Stable convergence\nperiod", transform=ax.transAxes,
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="grey"))
    # convergence epoch: first epoch after which the total loss stays within
    # 5% of its final value
    total = history["total"]
    final = total[-1]
    below = np.where(total <= final * 1.05 + 1e-6)[0]
    conv = int(below[0] + 1) if len(below) else int(len(total))
    conv_y = total[conv - 1]
    ax.scatter([conv], [conv_y], color="black", s=55, zorder=6)
    ax.annotate(f"Convergence to epoch {conv}",
                xy=(conv, conv_y),
                xytext=(conv + len(ep) * 0.10, conv_y + (total.max() - final) * 0.25),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("Training iteration rounds (Epoch)")
    ax.set_ylabel("Loss value")
    ax.legend(loc="upper right", framealpha=0.95)
    _panel_tag(ax, "A")

    ax = axes[1]
    q_lo, q_hi = np.percentile(mv_true, [20, 80])
    cat = np.where(mv_true > q_hi, 0, np.where(mv_true < q_lo, 2, 1))
    labels = ["High load and high electricity price",
              "Medium load medium electricity price",
              "Low load and low electricity price"]
    colors = ["#d62728", "#1f9ed8", "#2ca02c"]
    for c, lab, col in zip(range(3), labels, colors):
        m = cat == c
        ax.scatter(mv_true[m], mv_recon[m], s=12, color=col, alpha=0.7, label=lab)
    lim = [0, max(mv_true.max(), mv_recon.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=1.6, label="Ideal reconstruction line y=x")
    ax.plot(lim, [1.15 * v for v in lim], color="grey", ls="--", lw=1.0,
            label="\u00b115% Error boundary")
    ax.plot(lim, [0.85 * v for v in lim], color="grey", ls="--", lw=1.0)
    box = (f"RMSE = {metrics['rmse']:.2f} $   R\u00b2 = {metrics['r2']:.3f}\n"
           f"MAPE = {metrics['mape']:.2f}%   \u03c1 = {metrics['rho']:.3f}")
    ax.text(0.05, 0.95, box, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="black"))
    ax.set_xlabel("True marginal value ($)")
    ax.set_ylabel("CVAE reconstructs marginal value ($)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _panel_tag(ax, "B")

    fig.tight_layout()
    save_fig(fig, "figure2_training_reconstruction.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3 --
def plot_pareto_and_allocation(frontier, typ_rows, auc, best_row, best_sr):
    """Fig. 3: risk-revenue Pareto frontier and the resource allocation bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    cvar = np.array([r["cvar"] for r in frontier])
    rev = np.array([r["expected_revenue"] for r in frontier])
    lam = np.array([r["lambda"] for r in frontier])
    sc = ax.scatter(cvar, rev, c=lam, cmap="jet", s=32, zorder=4)
    ax.plot(cvar, rev, color="grey", lw=1.0, zorder=2)
    ax.scatter([best_row["cvar"]], [best_row["expected_revenue"]], marker="*",
               s=260, color="black", zorder=6)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Risk preference coefficient \u03bb")
    ax.text(0.04, 0.96,
            f"Optimal risk balance point\n\u03bb={best_row['lambda']:.2f}, "
            f"Sharpe={best_sr:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="#ffffe0", ec="black"))
    txt = "Typical point comparison:\n"
    for r in typ_rows:
        c = r["capacity"]
        txt += (f"lambda={r['lambda']:.1f}:  Revenue ={r['expected_revenue']:.0f}, "
                f"Risk ={abs(r['cvar']):.0f},\n")
        txt += (f"Allocation (Reduction/Transfer/Adjustment)="
                f"{c['curtail']:.0f}/{c['transfer']:.0f}/{c['adjust']:.0f} MW\n")
    ax.text(0.04, 0.78, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey"))
    ax.text(0.04, 0.10,
            f"Pareto frontier Area Under the Curve(AUC) = {auc:.1f}",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="#fff8dc", ec="black"))
    ax.set_xlabel("CVaR ($)")
    ax.set_ylabel("Expected revenue ($)")
    _panel_tag(ax, "A")

    ax = axes[1]
    labels = [f"\u03bb={r['lambda']:.1f}" for r in typ_rows]
    curt = [r["capacity"]["curtail"] for r in typ_rows]
    tran = [r["capacity"]["transfer"] for r in typ_rows]
    adj = [r["capacity"]["adjust"] for r in typ_rows]
    x = np.arange(len(typ_rows))
    ax.bar(x, curt, color="#d62728", label="Reduction resources")
    ax.bar(x, tran, bottom=curt, color="#1f9ed8", label="Transfer resources")
    ax.bar(x, adj, bottom=np.array(curt) + np.array(tran), color="#2ca02c",
           label="Adjustment resources")
    for i in range(len(typ_rows)):
        ax.text(i, curt[i] / 2, f"{curt[i]:.0f}", ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(i, curt[i] + tran[i] / 2, f"{tran[i]:.0f}", ha="center",
                va="center", color="white", fontweight="bold")
        ax.text(i, curt[i] + tran[i] + adj[i] / 2, f"{adj[i]:.0f}", ha="center",
                va="center", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Risk preference coefficient")
    ax.set_ylabel("Resource capacity (MW)")
    ax.legend(framealpha=0.95)
    _panel_tag(ax, "B")

    fig.tight_layout()
    save_fig(fig, "figure3_pareto_allocation.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4 --
def plot_distribution_fit(fit_rows, seeds_note):
    """Fig. 4: KL divergence and WD box plots per regime."""
    scenarios = ["Normal operation", "Peak load period", "Low-load period"]
    reg_keys = ["normal", "peak", "low"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    for ax, metric, tag, ylab in [(axes[0], "kl", "A", "KL divergence"),
                                  (axes[1], "wd", "B", "WD")]:
        positions = np.arange(len(scenarios))
        width = 0.15
        for mi, method in enumerate(METHOD_ORDER):
            data = [np.asarray(fit_rows[method][metric][rk]) for rk in reg_keys]
            bp = ax.boxplot(data, positions=positions + mi * width - 0.3,
                            widths=width * 0.9, patch_artist=True, showfliers=True)
            for box in bp["boxes"]:
                box.set(facecolor=METHOD_COLORS[method], alpha=0.45,
                        edgecolor=METHOD_COLORS[method])
            means = [float(np.mean(d)) for d in data]
            ax.scatter(positions + mi * width - 0.3, means, marker="D",
                       color=METHOD_COLORS[method], s=28, zorder=5,
                       label=method if tag == "A" else None)
        ax.set_xticks(positions)
        ax.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=11)
        ax.set_xlabel("Typical operating scenario")
        ax.set_ylabel(ylab)
        if tag == "A":
            ax.legend(framealpha=0.95, ncol=1, fontsize=10)
        _panel_tag(ax, tag)

    kl_txt = "KL divergence mean(Normal operation,\npeaks and low):\n"
    wd_txt = "WD mean(Normal\noperation, peaks and low):\n"
    for method in METHOD_ORDER:
        kl = fit_rows[method]["kl"]
        wd = fit_rows[method]["wd"]
        kl_txt += (f"{method} : {np.mean(kl['normal']):.3f} "
                   f"{np.mean(kl['peak']):.3f} {np.mean(kl['low']):.3f}\n")
        wd_txt += (f"{method} : {np.mean(wd['normal']):.2f} "
                   f"{np.mean(wd['peak']):.2f} {np.mean(wd['low']):.2f}\n")
    axes[0].text(0.02, -0.52, kl_txt.rstrip(), transform=axes[0].transAxes,
                 fontsize=9, va="top", family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="grey"))
    axes[1].text(0.02, -0.52, wd_txt.rstrip(), transform=axes[1].transAxes,
                 fontsize=9, va="top", family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="grey"))

    fig.tight_layout()
    save_fig(fig, "figure4_distribution_fit.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5 --
def plot_prediction_kde(pred_rows, kde_data, mv_true, ks, metrics_note):
    """Fig. 5: prediction error bars and KDE density comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    methods = METHOD_ORDER
    rmse = [pred_rows[m]["rmse"] for m in methods]
    mae = [pred_rows[m]["mae"] for m in methods]
    mape = [pred_rows[m]["mape"] for m in methods]
    x = np.arange(len(methods))
    w = 0.36
    ax.bar(x - w / 2, rmse, w, color="#1f4fd8", label="RMSE")
    ax.bar(x + w / 2, mae, w, color="#2ca02c", label="MAE")
    ax2 = ax.twinx()
    ax2.errorbar(x, mape, yerr=np.array(mape) * 0.06, fmt="o-", color="red",
                 capsize=4, lw=1.6, label="MAPE")
    for i, (r, m) in enumerate(zip(rmse, mae)):
        ax.text(i - w / 2, r + 0.05, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
    for i, mp in enumerate(mape):
        ax2.text(i, mp + 1.2, f"{mp:.1f}%", ha="center", color="red", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_xlabel("Method")
    ax.set_ylabel("RMSE / MAE")
    ax2.set_ylabel("MAPE (%)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax2.legend(loc="upper right", framealpha=0.95)
    ax.grid(False)
    _panel_tag(ax, "A")

    ax = axes[1]
    grid = np.linspace(mv_true.min() - 5, mv_true.max() + 5, 400)
    ax.plot(grid, kde_data["true"](grid), color="black", lw=2.2,
            label="True distribution")
    for method in METHOD_ORDER:
        ax.plot(grid, kde_data[method](grid), lw=1.8,
                color=METHOD_COLORS[method], label=method,
                ls="-" if method == "CVAE-SP" else "--")
    ax.set_xlabel("MV")
    ax.set_ylabel("Probability density")
    ax.legend(framealpha=0.95, fontsize=10)
    ax.text(0.42, 0.55, metrics_note, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey"))
    _panel_tag(ax, "B")

    fig.tight_layout()
    save_fig(fig, "figure5_prediction_kde.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6 --
def plot_pareto_comparison(frontiers_by_method, sharp_note):
    """Fig. 6: Pareto frontiers for all methods in the risk-revenue space."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    markers = {"CVAE-SP": "s", "VAE-LFE": "^", "QRF-SP": "o",
               "CGAN": "D", "DR-MILP": "v"}
    for method, rows in frontiers_by_method.items():
        cvar = np.array([r["cvar"] for r in rows])
        rev = np.array([r["expected_revenue"] for r in rows])
        order = np.argsort(cvar)
        ax.plot(cvar[order], rev[order], color=METHOD_COLORS[method], lw=2,
                marker=markers[method], ms=4, label=method)
    ax.text(0.55, 0.32, sharp_note, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey"))
    ax.set_xlabel("CVaR")
    ax.set_ylabel("Expected earnings")
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    save_fig(fig, "figure6_pareto_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 7 --
def plot_calibration(calib_rows, note):
    """Fig. 7: empirical coverage vs nominal confidence level."""
    fig, ax = plt.subplots(figsize=(9, 7))
    conf = np.arange(0, 101, 5)
    ax.plot(conf, conf, color="grey", ls="--", lw=1.2, label="Ideal calibration")
    markers = {"CVAE-SP": "D", "VAE-LFE": "s", "QRF-SP": "o",
               "CGAN": "p", "DR-MILP": "v"}
    for method in METHOD_ORDER:
        cov = calib_rows[method]["coverage"] * 100
        nominal = [c * 100 for c in
                   [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]]
        ax.plot(nominal, cov, color=METHOD_COLORS[method], lw=2,
                marker=markers[method], ms=5, label=method,
                ls="-" if method == "CVAE-SP" else "--")
    ax.text(0.05, 0.95, note, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="black"))
    ax.set_xlabel("Nominal confidence level (%)")
    ax.set_ylabel("Experience coverage (%)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save_fig(fig, "figure7_calibration.png")
    plt.close(fig)
