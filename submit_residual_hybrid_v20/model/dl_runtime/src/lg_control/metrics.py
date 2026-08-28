"""Probability metrics matching the competition definition."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be non-empty and have the same shape")
    return float(np.mean(np.square(p - y)))


def brier_skill(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    baseline = float(y.mean() * (1.0 - y.mean()))
    if baseline <= 0.0:
        return 0.0
    return float(1.0 - brier_score(y, probabilities) / baseline)


def competition_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return max(0.0, 100_000.0 * brier_skill(y_true, probabilities))


def log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score(y_true, probabilities),
        "brier_skill": brier_skill(y_true, probabilities),
        "competition_score": competition_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "ece_15": expected_calibration_error(y_true, probabilities, n_bins=15),
    }

