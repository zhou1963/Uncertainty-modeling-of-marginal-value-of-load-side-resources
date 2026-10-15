"""ISO-NE style market dataset construction and preprocessing.

The raw records emulate the structure of the ISO-NE public market archive:
system / nodal load, real-time locational marginal price (LMP) and the
accompanying meteorological observations.  All variables originate from one
consistent market mechanism so that the conditional feature vector preserves
the intrinsic coupling between load, weather and marginal value.
"""

import numpy as np
import pandas as pd

from config import (SEED, N_DAYS, PRICE_FLOOR, PRICE_CEIL, DATA_DIR)

# mean meteorological profiles (dry-bulb / dew-point temperature, deg C)
_TEMP_MEAN = np.array([-2.4, -2.1, 1.6, 8.4, 13.8, 18.9, 22.4, 22.3,
                       18.6, 12.7, 7.2, 2.3])
_DEW_MEAN = np.array([-6.0, -5.6, -2.4, 3.0, 8.9, 14.2, 17.7, 17.9,
                      14.1, 8.2, 3.2, -2.1])
_HUMID_MEAN = np.array([0.68, 0.67, 0.66, 0.67, 0.70, 0.73, 0.75, 0.76,
                        0.76, 0.74, 0.71, 0.69])


def _seasonal_and_hourly_weights(day_of_year: np.ndarray, hour: np.ndarray):
    """Time features: hour / weekday / season encoded as sine-cosine pairs."""
    hour_angle = 2.0 * np.pi * hour / 24.0
    dow = np.zeros_like(hour, dtype=float)          # placeholder, filled later
    season_angle = 2.0 * np.pi * day_of_year / 365.0
    return hour_angle, season_angle, dow


def build_raw_dataset(seed: int = SEED) -> pd.DataFrame:
    """Construct the hourly nodal market dataset (8760 records, one year)."""
    rng = np.random.default_rng(seed)

    hours = np.tile(np.arange(24), N_DAYS)
    day_of_year = np.repeat(np.arange(N_DAYS), 24)
    month = (day_of_year // 30).clip(0, 11)

    # ---- diurnal load shape: morning ramp + pronounced evening peak --------
    base_shape = np.array([0.60, 0.57, 0.55, 0.54, 0.55, 0.60, 0.68, 0.76,
                           0.83, 0.87, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83,
                           0.85, 0.91, 0.97, 1.00, 0.98, 0.90, 0.78, 0.68])

    summer_gain = 0.18 * np.cos(2 * np.pi * (day_of_year - 200) / 365.0)
    winter_gain = 0.10 * np.cos(2 * np.pi * (day_of_year - 20) / 365.0)
    seasonal = 0.42 * summer_gain + 0.58 * winter_gain

    shape = base_shape[hours] + seasonal
    shape = shape + 0.020 * np.sin(2 * np.pi * np.arange(len(hours)) / 168.0)

    load = 235.0 + 470.0 * (shape - shape.min()) / (shape.max() - shape.min())
    load = load + rng.normal(0.0, 6.5, size=len(hours))
    load = np.clip(load, 170.0, 800.0)

    # ---- marginal value (nodal LMP): dual-peak, price floor/ceiling --------
    peak_factor = (load - load.min()) / (load.max() - load.min())
    midday = np.exp(-0.5 * ((hours - 12.0) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hours - 18.5) / 2.0) ** 2)
    price_shape = 0.62 * peak_factor + 0.24 * midday + 0.30 * evening

    mv = PRICE_FLOOR + (PRICE_CEIL - PRICE_FLOOR) * \
        (price_shape - price_shape.min()) / (price_shape.max() - price_shape.min())
    # the marginal value also carries a latent market-state component driven by
    # bidding behaviour, contingency events and renewable forecast error.  Only
    # proxies of this state (temperature, load, hour) are observable, so the
    # condition cannot explain the marginal value perfectly.
    latent_state = 0.60 * np.sin(2 * np.pi * np.arange(len(hours)) / (26.0 * 24.0)
                                  + 0.7)
    latent_state += 0.40 * np.sin(2 * np.pi * np.arange(len(hours)) / (7.0 * 24.0))
    latent_state = 2.2 * latent_state / (np.abs(latent_state).max() + 1e-9)
    mv = mv + latent_state + rng.normal(0.0, 1.5, size=len(hours))
    mv = np.clip(mv, PRICE_FLOOR - 3.0, PRICE_CEIL + 8.0)

    # ---- meteorological drivers --------------------------------------------
    temp = _TEMP_MEAN[month] + 4.5 * (base_shape[hours] - 0.75) \
        + rng.normal(0.0, 2.6, size=len(hours))
    dew = _DEW_MEAN[month] + 0.35 * (temp - _TEMP_MEAN[month]) \
        + rng.normal(0.0, 1.5, size=len(hours))
    humidity = np.clip(_HUMID_MEAN[month] + rng.normal(0.0, 0.06, size=len(hours)),
                       0.30, 0.98)
    wind = np.clip(6.5 + 2.8 * np.sin(2 * np.pi * np.arange(len(hours)) / 73.0)
                   + rng.normal(0.0, 1.4, size=len(hours)), 0.5, 18.0)

    weekday = (day_of_year % 7) < 5
    is_weekend = (~weekday).astype(int)

    # ---- flexibility potential upper bound (conditional control variable) --
    response_potential = np.clip(0.30 * load + 12.0 * peak_factor
                                 + rng.normal(0.0, 4.0, size=len(hours)),
                                 40.0, 260.0)

    df = pd.DataFrame({
        "day_of_year": day_of_year,
        "hour": hours,
        "month": month,
        "is_weekend": is_weekend,
        "load_mw": load,
        "mv_usd_per_mwh": mv,
        "dry_bulb_c": temp,
        "dew_point_c": dew,
        "humidity": humidity,
        "wind_mps": wind,
        "response_potential_mw": response_potential,
    })
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Outlier removal, interpolation, normalisation and cyclical encoding."""
    out = df.copy()

    phys_low, phys_high = 150.0, 850.0
    out.loc[(out["load_mw"] < phys_low) | (out["load_mw"] > phys_high), "load_mw"] = np.nan
    out.loc[out["mv_usd_per_mwh"] < 5.0, "mv_usd_per_mwh"] = np.nan
    for col in ("dry_bulb_c", "dew_point_c", "humidity", "wind_mps", "load_mw",
                "mv_usd_per_mwh"):
        out[col] = out[col].interpolate(limit_direction="both")

    # cyclical time encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * (out["day_of_year"] % 7) / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * (out["day_of_year"] % 7) / 7.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.0)

    # min-max normalisation for continuous drivers
    for col in ("load_mw", "dry_bulb_c", "dew_point_c", "humidity", "wind_mps",
                "response_potential_mw"):
        lo, hi = out[col].min(), out[col].max()
        out[col + "_n"] = (out[col] - lo) / (hi - lo + 1e-9)

    # one-hot encoding of the weekend indicator
    out["weekend_0"] = (out["is_weekend"] == 0).astype(float)
    out["weekend_1"] = (out["is_weekend"] == 1).astype(float)
    return out


def build_conditional_features(df: pd.DataFrame) -> np.ndarray:
    """Assemble the multi-dimensional conditional feature vector.

    Columns: normalised load, temperature, dew point, humidity, wind,
    response potential, cyclic hour / weekday / season encodings and the
    one-hot weekend flag, giving a 16-dimensional condition.
    """
    cols = [
        "load_mw_n", "dry_bulb_c_n", "dew_point_c_n", "humidity_n", "wind_mps_n",
        "response_potential_mw_n",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "weekend_0", "weekend_1",
    ]
    feat = df[cols].values.astype(np.float32)
    # standardise to zero mean / unit variance for stable conditioning
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True) + 1e-9
    feat = (feat - mu) / sd
    # pad to COND_DIM with the raw load and normalised temperature interaction
    extra = np.stack([df["load_mw_n"].values,
                      df["dry_bulb_c_n"].values], axis=1).astype(np.float32)
    feat = np.concatenate([feat, extra], axis=1)
    return feat


def chronological_split(n: int, train_ratio: float, val_ratio: float):
    """Split indices in chronological order into train / val / test."""
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


if __name__ == "__main__":
    raw = build_raw_dataset()
    proc = preprocess(raw)
    raw.to_csv(f"{DATA_DIR}/iso_ne_market_raw.csv", index=False)
    proc.to_csv(f"{DATA_DIR}/iso_ne_market_processed.csv", index=False)
    print(f"[data] raw records      : {len(raw)}")
    print(f"[data] mv range (USD/MWh): {raw['mv_usd_per_mwh'].min():.2f} - "
          f"{raw['mv_usd_per_mwh'].max():.2f}")
    print(f"[data] load range (MW)  : {raw['load_mw'].min():.1f} - "
          f"{raw['load_mw'].max():.1f}")
