"""Data loading, synthetic data generation and feature engineering.

The schema mirrors the public Kaggle "Credit Card Fraud Detection" dataset
(Time, V1..V28, Amount, Class) so the same code runs on real data when
``data/creditcard.csv`` exists, and on a synthetic stand-in otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LABEL = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]


def make_synthetic(n_samples: int = 60_000, fraud_rate: float = 0.01,
                   seed: int = 42) -> pd.DataFrame:
    """Generate an imbalanced, Kaggle-shaped dataset with injected fraud.

    Normal transactions come from 3 correlated Gaussian clusters. Fraud is
    drawn from the same clusters but shifted on a random subset of features,
    with a bimodal amount (small card-testing charges and large cash-outs)
    and a bias toward night-time hours. The shift is deliberately modest so
    the task is not trivial.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_samples * fraud_rate), 10)
    n_norm = n_samples - n_fraud
    d = len(V_COLS)

    mix = rng.normal(size=(d, d)) * 0.3 + np.eye(d)
    centers = rng.normal(scale=1.0, size=(3, d))

    def base(n: int) -> np.ndarray:
        comp = rng.integers(0, 3, size=n)
        return rng.normal(size=(n, d)) @ mix + centers[comp]

    x_norm, x_fraud = base(n_norm), base(n_fraud)
    shifted = rng.choice(d, size=max(d // 3, 3), replace=False)
    x_fraud[:, shifted] += rng.choice([-1.0, 1.0], size=len(shifted)) * 1.8
    x_fraud[:, shifted] *= 1.3

    amt_norm = rng.lognormal(mean=3.5, sigma=1.2, size=n_norm)
    small = rng.random(n_fraud) < 0.5
    amt_fraud = np.where(small,
                         rng.lognormal(1.5, 0.8, n_fraud),
                         rng.lognormal(5.5, 0.9, n_fraud))

    t_norm = rng.uniform(0, 172_800, n_norm)
    night_hours = rng.choice([0, 1, 2, 3, 4, 23], size=n_fraud)
    day_offset = rng.integers(0, 2, n_fraud) * 86_400
    t_fraud = day_offset + night_hours * 3600 + rng.uniform(0, 3600, n_fraud)

    df = pd.DataFrame(np.vstack([x_norm, x_fraud]), columns=V_COLS)
    df.insert(0, "Time", np.concatenate([t_norm, t_fraud]))
    df["Amount"] = np.concatenate([amt_norm, amt_fraud])
    df[LABEL] = np.concatenate([np.zeros(n_norm, int), np.ones(n_fraud, int)])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_data(path: str | Path | None = None, n_samples: int = 60_000,
              seed: int = 42) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source). Uses a real CSV if present, else synthetic."""
    candidate = Path(path) if path else Path("data/creditcard.csv")
    if candidate.exists():
        df = pd.read_csv(candidate)
        missing = {"Time", "Amount", LABEL} - set(df.columns)
        if missing:
            raise ValueError(f"{candidate} is missing columns: {sorted(missing)}")
        return df, f"real:{candidate}"
    if path:
        raise FileNotFoundError(f"Dataset not found: {candidate}")
    return make_synthetic(n_samples=n_samples, seed=seed), "synthetic"


class FeatureBuilder:
    """Feature engineering + standardisation (fit on training rows only)."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.columns_: list[str] = []

    @staticmethod
    def _engineer(X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=[c for c in (LABEL,) if c in X.columns]).copy()
        hour = (out.pop("Time") // 3600) % 24
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["log_amount"] = np.log1p(out.pop("Amount"))
        return out

    def fit(self, X: pd.DataFrame) -> "FeatureBuilder":
        feats = self._engineer(X)
        self.columns_ = list(feats.columns)
        self.scaler.fit(feats.to_numpy(dtype=float))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        feats = self._engineer(X)[self.columns_]
        return self.scaler.transform(feats.to_numpy(dtype=float))
