"""Scenario generation, typical-scene clustering and scenario reduction.

The trained CVAE is queried with the operating condition of each hour to draw a
set of marginal-value scenarios.  K-means groups the generated samples into
representative typical scenes; the occurrence probability of each scene follows
from the share of samples assigned to it, which turns the continuous
conditional distribution into a discrete, probability-weighted scene set ready
for stochastic programming.
"""

import numpy as np
from sklearn.cluster import KMeans

from config import N_SCENARIOS, N_TYPICAL_SCENES, N_SP_SCENES, SP_HORIZON, SEED

SCENE_DESCRIPTIONS = [
    "Late night trough period",
    "Late night fall period",
    "Peak lunch hour",
    "Early peak climbing period",
    "Afternoon rush hour",
    "Evening peak hours",
]


def generate_conditional_scenarios(model, cond, mv_mean, mv_std,
                                   n_scenarios=N_SCENARIOS, seed=SEED):
    """Draw n_scenarios MV samples per operating condition."""
    import torch
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        c = torch.tensor(cond, dtype=torch.float32)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        out = []
        for _ in range(n_scenarios):
            z = torch.randn(c.shape[0], model.latent_dim, dtype=torch.float32)
            out.append(model.decoder(z, c).squeeze(-1).numpy() * mv_std + mv_mean)
    return np.array(out)                       # [n_scenarios, n_hours]


def cluster_typical_scenes(scenarios, mv_series, n_scenes=N_TYPICAL_SCENES,
                           seed=SEED, hour_of_day=None):
    """Build the typical operating scenes from the generated MV scenarios.

    Each hour of the day is represented by its generated MV ensemble.  The
    hours are grouped into ``n_scenes`` clusters that correspond to distinct
    operating regimes (trough, morning ramp, midday, afternoon, evening peak,
    night fall); the occurrence probability of a scene is the share of the day
    it covers, and its statistics are aggregated over the ensemble.

    Returns
    -------
    dict with scene statistics used for Table 1.
    """
    sc = np.asarray(scenarios)                              # [K, H]
    n_hours = sc.shape[1]
    hours = (np.arange(n_hours) % 24) if hour_of_day is None else hour_of_day

    # feature per hour: the ensemble profile across the scenario draws
    hour_profiles = sc.T                                    # [n_hours, K]
    km = KMeans(n_clusters=n_scenes, n_init=10, random_state=seed)
    labels = km.fit_predict(hour_profiles)
    # lower the scene index to the hour of day that characterises it
    scene_hour = {}
    for s in range(n_scenes):
        member_hours = hours[labels == s]
        scene_hour[s] = int(np.bincount(member_hours % 24, minlength=24).argmax())

    scenes = []
    for s in range(n_scenes):
        member = sc[:, labels == s]                         # [K, n_member]
        mean_val = float(member.mean())
        peak_val = float(member.max(axis=1).mean())
        vol = float(member.std())
        scenes.append({
            "probability": float(len(np.where(labels == s)[0]) / n_hours),
            "mean": mean_val,
            "peak": peak_val,
            "volatility": vol,
            "hour": scene_hour[s],
        })

    # order scenes by mean value: S1 = lowest (trough) ... S6 = highest (peak)
    scenes.sort(key=lambda d: d["mean"])
    for i, scn in enumerate(scenes):
        scn["id"] = f"S{i + 1}"
        scn["description"] = SCENE_DESCRIPTIONS[i % len(SCENE_DESCRIPTIONS)]
    return {"scenes": scenes, "labels": labels,
            "centers": km.cluster_centers_}


def reduce_scenarios(scenarios, n_keep=N_SP_SCENES, seed=SEED):
    """Select a representative subset of scenarios for the SP model.

    Scenarios are clustered and the medoid of each cluster is retained,
    preserving the spread of the original set while reducing the model size.
    """
    rng = np.random.default_rng(seed)
    # each scenario is one full MV path; cluster the scenario population
    feat = scenarios                                       # [K, H]
    n_keep = min(n_keep, feat.shape[0])
    km = KMeans(n_clusters=min(n_keep, 40), n_init=5, random_state=seed)
    labels = km.fit_predict(feat)

    kept_idx, weights = [], []
    for s in np.unique(labels):
        member = np.where(labels == s)[0]
        center = feat[member].mean(axis=0)
        dist = np.linalg.norm(feat[member] - center, axis=1)
        medoid = member[np.argmin(dist)]
        kept_idx.append(medoid)
        weights.append(len(member))
    kept_idx = np.array(kept_idx)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    kept = scenarios[kept_idx]                             # [n_keep, H]
    return kept, weights, kept_idx


def daily_profile(scenarios):
    """Aggregate generated scenarios onto a 24-hour profile with quantiles."""
    sc = np.array(scenarios)                                # [K, N]
    if sc.shape[1] != SP_HORIZON:
        # resample length-N vector to 24 hourly points
        idx = np.linspace(0, sc.shape[1] - 1, SP_HORIZON).astype(int)
        sc = sc[:, idx]
    return {"mean": sc.mean(axis=0),
            "p05": np.percentile(sc, 5, axis=0),
            "p95": np.percentile(sc, 95, axis=0),
            "samples": sc}
