"""Train-only affine logit calibration for Brier-score optimisation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


@dataclass
class AffineLogitCalibrator:
    """Apply ``sigmoid(scale * logit(p) + bias)`` with two fitted parameters."""

    scale: float = 1.0
    bias: float = 0.0
    objective: str = "brier"

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "AffineLogitCalibrator":
        logits = _logit(probabilities).reshape(-1)
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        if logits.shape != y.shape or y.size == 0:
            raise ValueError("probabilities and y_true must be non-empty with matching shapes")
        if not np.isfinite(logits).all() or not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("calibration inputs must be finite and targets binary")

        def loss(parameters: np.ndarray) -> float:
            p = _sigmoid(parameters[0] * logits + parameters[1])
            return float(np.mean(np.square(p - y)))

        result = minimize(
            loss,
            x0=np.asarray([1.0, 0.0]),
            method="L-BFGS-B",
            bounds=((0.05, 5.0), (-5.0, 5.0)),
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"calibrator optimisation failed: {result.message}")
        self.scale = float(result.x[0])
        self.bias = float(result.x[1])
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return _sigmoid(self.scale * _logit(probabilities) + self.bias)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "scale": self.scale,
            "bias": self.bias,
            "objective": self.objective,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AffineLogitCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported calibrator version: {payload.get('version')}")
        return cls(
            scale=float(payload["scale"]),
            bias=float(payload["bias"]),
            objective=str(payload.get("objective", "brier")),
        )

