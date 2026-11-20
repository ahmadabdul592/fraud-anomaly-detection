"""Metrics suited to extreme class imbalance (accuracy is deliberately absent)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    """Precision among the k highest-scored items (default k = number of frauds)."""
    k = int(k or y_true.sum())
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(f1.max())


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision_at_k": precision_at_k(y_true, scores),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "best_f1": best_f1(y_true, scores),
    }
