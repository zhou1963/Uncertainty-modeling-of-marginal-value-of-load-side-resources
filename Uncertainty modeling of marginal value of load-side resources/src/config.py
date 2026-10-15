"""Global configuration for load-side resource marginal value (MV) modeling.

The framework couples a Conditional Variational Autoencoder (CVAE) for
conditional probability learning of the marginal value with a two-stage
stochastic programming (SP) model that embeds the generated scenarios and a
conditional value-at-risk (CVaR) penalty term.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")

for _d in (DATA_DIR, FIG_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)

SEED = 2024

# ---------------------------------------------------------------- dataset ----
N_DAYS = 365                     # one full year of hourly records
N_SAMPLES = N_DAYS * 24          # 8760 hourly samples
PRICE_FLOOR = 11.0               # $/MWh, overnight low level
PRICE_CEIL = 46.0                # $/MWh, midday peak level

# ----------------------------------------------------------------- CVAE ------
COND_DIM = 16                    # conditional feature dimension
LATENT_DIM = 64                  # latent variable dimension
HIDDEN_DIM = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 200
BETA_KL = 0.05                   # KL divergence weight
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ------------------------------------------------------- scenario / SP -------
N_SCENARIOS = 100                # generated MV scenarios
N_TYPICAL_SCENES = 6             # clustered typical scenes
N_SP_SCENES = 20                 # reduced scene set for the SP model
SP_HORIZON = 24                  # dispatch horizon (hours)
BETA_CVAR = 0.95                 # CVaR confidence level
RISK_LAMBDA_GRID = [round(i / 20, 2) for i in range(21)]   # lambda in [0, 1]

# resource fleet limits (MW)
C_TOTAL = 500.0
C_MAX = {"curtail": 320.0, "transfer": 260.0, "adjust": 230.0}
RESOURCE_TYPES = ["Reduction resources", "Transfer resources", "Adjustment resources"]

# --------------------------------------------------------- calibration -------
CONF_LEVELS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# ------------------------------------------------------------- baselines -----
BASELINES = ["VAE-LFE", "QRF-SP", "CGAN", "DR-MILP"]
ALL_METHODS = ["CVAE-SP", "VAE-LFE", "QRF-SP", "CGAN", "DR-MILP"]

METHOD_COLORS = {
    "CVAE-SP": "#1f5fa8",
    "VAE-LFE": "#2ca02c",
    "QRF-SP": "#ff8c00",
    "CGAN": "#d62728",
    "DR-MILP": "#7b2fbe",
}
