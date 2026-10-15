# Load-Side Resource Marginal Value Uncertainty Modeling
## CVAE + Stochastic Programming

This project implements the complete computational pipeline of the paper
*"Uncertainty Modeling of Load-Side Resource Marginal Value Combining a
Conditional Variational Autoencoder and Stochastic Programming"*.  It couples a
**Conditional Variational Autoencoder (CVAE)** for conditional probability
learning of the marginal value (MV) with a **two-stage mean-CVaR stochastic
program (SP)** that turns the learned distribution into a risk-aware resource
allocation decision.

---

## 1. Method overview

| Stage | Module | Purpose |
|-------|--------|---------|
| Data | `src/data_loader.py` | ISO-NE style market dataset (load, real-time LMP, weather), outlier handling, interpolation, min-max normalisation, cyclical time encoding, multi-dimensional conditional feature vector |
| Model | `src/cvae.py` | CVAE with reparameterised latent sampling; learns the MV conditional distribution and reconstructs MV from the operating condition |
| Scenes | `src/scenario.py` | Conditional scenario sampling, K-means typical-scene clustering (probability = cluster share), representative scenario reduction |
| Decision | `src/stochastic_program.py` | Two-stage mean-CVaR SP; per-block resource allocation across reduction / transfer / adjustment categories; Pareto frontier over the risk-preference coefficient λ |
| Evaluation | `src/evaluation.py` | KL, Wasserstein distance, RMSE/MAE/MAPE, KS test, KDE, calibration/CRPS, diversity & tail capture, ablation study |
| Figures | `src/plotting.py` | All seven result figures |
| Driver | `src/main.py` | End-to-end orchestration |

The conditional feature vector (16-D) combines normalised load, dry-bulb and
dew-point temperature, humidity, wind speed, the user response-potential bound,
cyclical hour / weekday / season encodings and the one-hot weekend flag.

---

## 2. Data

The raw records emulate the structure of the
[ISO-NE public market archive](https://www.iso-ne.com/markets-operations/iso-express):
hourly nodal load, real-time locational marginal price and meteorological
observations, covering one full year (8760 hourly samples).  A representative
load node is used so the intra-day peak–valley structure and the coupling
between nodal load and nodal price are preserved.

The dataset is written to `data/`:

- `iso_ne_market_raw.csv` — raw hourly records
- `iso_ne_market_processed.csv` — cleaned and normalised records

The samples are split chronologically 70% / 15% / 15% into train / validation /
test sets, and every reported metric is the mean over independent repetitions.

---

## 3. Results produced

Running the pipeline generates:

**Figures** (`results/figures/`)

1. `figure1_dataset_overview.png` — load response profile and the double-peak
   real-time price with its fluctuation band
2. `figure2_training_reconstruction.png` — training loss curves (reconstruction,
   KL, total) and the MV reconstruction scatter
3. `figure3_pareto_allocation.png` — expected revenue – CVaR Pareto frontier and
   the resource capacity allocation under typical risk levels
4. `figure4_distribution_fit.png` — KL divergence and WD box plots per
   operating regime
5. `figure5_prediction_kde.png` — prediction error bars (RMSE/MAE/MAPE) and the
   KDE density comparison
6. `figure6_pareto_comparison.png` — Pareto frontiers of all methods
7. `figure7_calibration.png` — empirical coverage vs. nominal confidence level

**Tables** (`results/tables/`)

- `table1_typical_scenes.csv` — clustered typical MV scenes
- `table2_pareto_frontier.csv` — SP frontier over λ
- `table3_diversity.csv` — scenario diversity and extreme-value capture
- `table4_ablation.csv` — component ablation results
- `metrics_summary.json` — consolidated metrics

---

## 4. Environment

```
Python      3.10+
numpy       1.26
pandas      2.1
scikit-learn 1.4
scipy       1.11
torch       2.1
matplotlib  3.8
```

Install with:

```bash
pip install -r requirements.txt
```

## 5. Running

```bash
python src/main.py
```

Console output reports each stage; all figures and tables are written to
`results/`.  The random seed is fixed in `src/config.py` for reproducibility.

---

## 6. Project layout

```
.
├── src/
│   ├── config.py               # global parameters
│   ├── data_loader.py          # dataset construction & preprocessing
│   ├── cvae.py                 # Conditional VAE model & training
│   ├── scenario.py             # scenario generation & clustering
│   ├── stochastic_program.py   # mean-CVaR two-stage SP
│   ├── evaluation.py           # metrics & reference methods
│   ├── plotting.py             # figures
│   └── main.py                 # end-to-end driver
├── data/                       # generated datasets
├── results/
│   ├── figures/                # result figures
│   └── tables/                 # result tables
└── requirements.txt
```
