"""Uniform wrappers so every detector exposes fit(X) / score(X).

Convention: a HIGHER score always means MORE anomalous.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .autoencoder import make_autoencoder


class Detector:
    def __init__(self, name: str, estimator, max_fit_rows: int | None = None,
                 seed: int = 42) -> None:
        self.name, self.est = name, estimator
        self.max_fit_rows, self.seed = max_fit_rows, seed

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Detector":
        if self.max_fit_rows and len(X) > self.max_fit_rows:   # kernel/kNN methods scale badly
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_fit_rows, replace=False)
            X = X[idx]
        self.est.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.est.score_samples(X)          # sklearn: low score_samples = anomalous


class AutoencoderDetector(Detector):
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.est.score_samples(X)           # reconstruction error: high = anomalous


class SupervisedReference:
    """Labelled upper-bound reference. NOT an anomaly detector: it sees fraud labels."""

    def __init__(self, seed: int = 42) -> None:
        self.name = "RandomForest (supervised reference)"
        self.est = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                          n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


def build_detectors(seed: int = 42, backend: str = "auto", contamination: float = 0.01):
    ae = make_autoencoder(backend, seed)
    return [
        Detector("Isolation Forest",
                 IsolationForest(n_estimators=300, contamination=contamination,
                                 random_state=seed, n_jobs=-1), seed=seed),
        Detector("One-Class SVM",
                 OneClassSVM(kernel="rbf", gamma="scale", nu=max(contamination, 0.005)),
                 max_fit_rows=10_000, seed=seed),
        Detector("Local Outlier Factor",
                 LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1),
                 max_fit_rows=20_000, seed=seed),
        AutoencoderDetector(ae.name, ae, seed=seed),
    ]
