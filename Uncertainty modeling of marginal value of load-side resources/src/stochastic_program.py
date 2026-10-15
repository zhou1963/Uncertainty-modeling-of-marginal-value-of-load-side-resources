"""Two-stage stochastic programming with a mean-CVaR objective.

Stage 1 allocates capacity of the three load-side resource categories
(reduction / transfer / adjustment) across the dispatch blocks of the day;
Stage 2 evaluates the resulting portfolio under every marginal-value scenario.
A conditional value-at-risk (CVaR) penalty, weighted by the risk preference
coefficient lambda, trades expected revenue against tail risk and traces a
risk-revenue Pareto frontier.  The per-block decision space gives the model the
freedom to shift the resource mix smoothly as risk aversion increases.
"""

import numpy as np
from scipy.optimize import linprog

from config import (C_TOTAL, C_MAX, BETA_CVAR, RISK_LAMBDA_GRID, SP_HORIZON)

RESOURCE_KEYS = ["curtail", "transfer", "adjust"]
N_BLOCKS = 4                                   # dispatch blocks (6 h each)
BLOCK_HOURS = SP_HORIZON // N_BLOCKS

# steady-state availability of each resource category (share of capacity that
# can actually be delivered when called)
UTILISATION = {"curtail": 0.90, "transfer": 0.78, "adjust": 0.70}
# allocation cost ($ per MW of installed capacity)
UNIT_COST = {"curtail": 6.0, "transfer": 5.2, "adjust": 4.6}
# maximum share of the fleet that can be assigned to one resource category
MAX_SHARE = {"curtail": 0.70, "transfer": 0.78, "adjust": 0.72}


def _block_aggregate(scenarios):
    """Aggregate hourly scenarios to per-block mean value [S, n_blocks]."""
    sc = np.asarray(scenarios)                              # [S, H]
    if sc.shape[1] != SP_HORIZON:
        idx = np.linspace(0, sc.shape[1] - 1, SP_HORIZON).astype(int)
        sc = sc[:, idx]
    S = sc.shape[0]
    out = np.zeros((S, N_BLOCKS))
    for b in range(N_BLOCKS):
        out[:, b] = sc[:, b * BLOCK_HOURS:(b + 1) * BLOCK_HOURS].mean(axis=1)
    return out                                              # [S, n_blocks]


def _scenario_revenue_block(scenarios_block):
    """Revenue [S, n_blocks * n_res] per MW of capacity in each block.

    Each resource category is exposed to a different combination of the
    underlying market drivers, so the three categories are only partially
    correlated across scenarios.  Reduction resources track the magnitude of the
    high-value hours, transfer resources track the intra-day price spread and
    adjustment resources track the stability of the value level.  Because the
    profiles are not perfectly correlated, combining them yields a genuine
    diversification benefit and the risk preference reshapes the mix.
    """
    S, B = scenarios_block.shape
    rev = np.zeros((S, B * len(RESOURCE_KEYS)))
    for b in range(B):
        lvl = scenarios_block[:, b]                          # [S]
        # three partially independent exposure signals derived from the block
        exposure_peak = np.clip((lvl - np.percentile(lvl, 20))
                                / (np.percentile(lvl, 95) - np.percentile(lvl, 20) + 1e-9),
                                0.0, 1.6)
        spread = np.abs(lvl - lvl.mean()) / (lvl.std() + 1e-9)
        curve = np.exp(-0.5 * ((lvl - lvl.mean()) / (lvl.std() + 1e-6)) ** 2)

        col0 = UTILISATION["curtail"] * (0.80 + 0.75 * exposure_peak) * BLOCK_HOURS
        col1 = UTILISATION["transfer"] * (1.05 + 0.35 * spread) * BLOCK_HOURS
        col2 = UTILISATION["adjust"] * (1.10 + 0.30 * curve) * BLOCK_HOURS
        rev[:, 3 * b + 0] = lvl * col0
        rev[:, 3 * b + 1] = lvl * col1
        rev[:, 3 * b + 2] = lvl * col2
    return rev


def solve_mean_cvar(scenarios, weights, lam, beta=BETA_CVAR):
    """Solve the mean-CVaR stochastic program for a given lambda.

    Decision variables: x (n_blocks * 3 capacities), t (VaR), u (S shortfalls).
    """
    block = _block_aggregate(scenarios)                     # [S, B]
    rev = _scenario_revenue_block(block) / 1e3              # k$ / MW
    S = rev.shape[0]
    n_x = rev.shape[1]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    cost = np.tile([UNIT_COST[k] for k in RESOURCE_KEYS], N_BLOCKS) / 1e3
    mean_rev = w @ rev                                       # [n_x]

    n_t, n_u = 1, S
    nvar = n_x + n_t + n_u
    c = np.zeros(nvar)
    # minimise  -(E[rev] - cost)  +  lambda * CVaR(loss)
    # with CVaR(loss) = t + 1/(1-beta) * E[(loss - t)+]  (Rockafellar-Uryasev)
    c[:n_x] = -mean_rev + cost
    c[n_x] = lam
    c[n_x + 1:n_x + 1 + n_u] = lam / (1.0 - beta) * w

    A_ub, b_ub = [], []
    # loss_s = -rev_s . x ; u_s >= loss_s - t  ->  -rev_s.x - t - u_s <= 0
    for s in range(S):
        row = np.zeros(nvar)
        row[:n_x] = -rev[s]
        row[n_x] = -1.0
        row[n_x + 1 + s] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq, b_eq = [], []
    # total fleet capacity per block equals C_TOTAL
    for b in range(N_BLOCKS):
        row = np.zeros(nvar)
        row[3 * b:3 * b + 3] = 1.0
        A_eq.append(row)
        b_eq.append(C_TOTAL)
    # per-category aggregate cap across the day (inequality)
    for j, key in enumerate(RESOURCE_KEYS):
        row = np.zeros(nvar)
        for b in range(N_BLOCKS):
            row[3 * b + j] = 1.0
        A_ub.append(row)
        b_ub.append(MAX_SHARE[key] * C_TOTAL * N_BLOCKS)

    u_ub = float(np.abs(rev).max() * C_TOTAL * 4.0 + 1.0)
    bounds = ([(0, C_TOTAL) for _ in range(n_x)]
              + [(-u_ub, u_ub)] + [(0, u_ub)] * n_u)

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"SP infeasible: {res.message}")

    x = res.x[:n_x]
    t = res.x[n_x]
    u = res.x[n_x + 1:n_x + 1 + n_u]

    per_scenario = rev @ x * 1e3
    exp_rev = float(w @ per_scenario)
    alloc_cost = float(cost @ x) * 1e3
    # CVaR reported as the expected shortfall of the revenue below its
    # expectation (positive $ magnitude of the downside)
    shortfall = exp_rev - per_scenario
    order = np.argsort(shortfall)[::-1]
    cum = np.cumsum(w[order])
    tail = shortfall[order][cum <= 1.0 - beta + 1e-12]
    if len(tail) == 0:
        k = max(1, int(np.searchsorted(cum, 1.0 - beta)) + 1)
        tail = shortfall[order][:k]
    cvar_loss = float(tail.mean())
    std_rev = float(np.sqrt(np.sum(w * (per_scenario - exp_rev) ** 2)))

    # aggregate per-category capacity across blocks
    cap_by_res = {}
    for j, key in enumerate(RESOURCE_KEYS):
        cap_by_res[key] = float(sum(x[3 * b + j] for b in range(N_BLOCKS))
                                / N_BLOCKS)

    return {
        "lambda": lam,
        "capacity": cap_by_res,
        "x_full": x,
        "expected_revenue": exp_rev,
        "allocation_cost": alloc_cost,
        "cvar": cvar_loss,
        "std_revenue": std_rev,
        "net_revenue": exp_rev - alloc_cost,
    }


def solve_pareto_frontier(scenarios, weights, lambdas=RISK_LAMBDA_GRID):
    """Solve the SP model over a grid of risk preference coefficients."""
    return [solve_mean_cvar(scenarios, weights, lam) for lam in lambdas]


def frontier_auc(frontier):
    """Area under the expected-revenue / CVaR frontier (trapezoidal)."""
    cvar = np.array([r["cvar"] for r in frontier])
    rev = np.array([r["expected_revenue"] for r in frontier])
    order = np.argsort(cvar)
    return float(np.trapz(rev[order], cvar[order]))


def optimal_risk_balance(frontier):
    """Return the row maximising the Sharpe ratio."""
    best, best_sr = None, -np.inf
    for r in frontier:
        sr = r["net_revenue"] / (r["std_revenue"] + 1e-9)
        if sr > best_sr:
            best_sr, best = sr, r
    return best, best_sr
