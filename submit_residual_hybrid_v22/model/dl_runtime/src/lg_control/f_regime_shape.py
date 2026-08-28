"""Level-free F correction from post-2022 deep exclusive-state outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import json

import numpy as np
import pandas as pd

from .f_regime_residual import HEAD_COLUMNS


FEATURE_NAMES = (
    "dl_reverse_total",
    "dl_middle_total",
    "dl_reverse_minus_asof",
    "dl_middle_minus_asof",
    *(f"dl_{name}" for name in HEAD_COLUMNS),
)


def build_f_regime_shape_features(
    frame: pd.DataFrame,
    heads: Mapping[str, np.ndarray],
) -> np.ndarray:
    required = {"asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"F regime shape input is missing: {sorted(missing)}")
    rows = len(frame)
    values: dict[str, np.ndarray] = {}
    for name in HEAD_COLUMNS:
        if name not in heads:
            raise ValueError(f"F regime shape heads are missing {name}")
        array = np.asarray(heads[name], dtype=np.float64).reshape(-1)
        if len(array) != rows or not np.isfinite(array).all():
            raise ValueError(f"F regime shape head {name} is not finite and row-aligned")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(f"F regime shape head {name} is outside [0, 1]")
        values[name] = array
    outcomes = np.column_stack([values[name] for name in HEAD_COLUMNS[:5]])
    if not np.allclose(outcomes.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError("F regime exclusive outcome heads must sum to one")
    reverse = values["outcome_reverse_only"] + values["outcome_reverse_and_middle"]
    middle = values["outcome_middle_only"] + values["outcome_reverse_and_middle"]
    asof_reverse = pd.to_numeric(
        frame["asof_pitcher_reverse_rate"], errors="coerce"
    ).to_numpy(float)
    asof_middle = pd.to_numeric(
        frame["asof_pitcher_middle_rate"], errors="coerce"
    ).to_numpy(float)
    asof_reverse = np.where(np.isfinite(asof_reverse), asof_reverse, reverse)
    asof_middle = np.where(np.isfinite(asof_middle), asof_middle, middle)
    matrix = np.column_stack(
        [
            reverse,
            middle,
            reverse - asof_reverse,
            middle - asof_middle,
            *(values[name] for name in HEAD_COLUMNS),
        ]
    )
    if matrix.shape != (rows, len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise RuntimeError("F regime shape feature matrix is invalid")
    return matrix


@dataclass(frozen=True)
class FRegimeShapeArtifact:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    cap: float
    strength: float
    manifest: dict[str, object]

    def __post_init__(self) -> None:
        width = len(FEATURE_NAMES)
        for name, value in (
            ("feature_mean", self.feature_mean),
            ("feature_scale", self.feature_scale),
            ("coefficients", self.coefficients),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite vector of width {width}")
        if np.any(np.asarray(self.feature_scale) <= 0.0):
            raise ValueError("feature scales must be positive")
        if not 0.0 < self.cap <= 0.10 or not 0.0 < self.strength <= 1.0:
            raise ValueError("F regime shape cap/strength is invalid")


def load_f_regime_shape(directory: Path) -> FRegimeShapeArtifact:
    manifest = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "f_regime_deep_shape_v1":
        raise ValueError("unexpected F regime shape artifact kind")
    if tuple(manifest["feature_names"]) != FEATURE_NAMES:
        raise ValueError("F regime shape feature contract differs")
    return FRegimeShapeArtifact(
        feature_mean=np.asarray(manifest["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(manifest["feature_scale"], dtype=np.float64),
        coefficients=np.asarray(manifest["coefficients"], dtype=np.float64),
        cap=float(manifest["cap"]),
        strength=float(manifest["strength"]),
        manifest=manifest,
    )


def predict_f_regime_shape(
    artifact: FRegimeShapeArtifact,
    frame: pd.DataFrame,
    base_probability: np.ndarray,
    heads: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_probability, dtype=np.float64).reshape(-1)
    if len(base) != len(frame) or not np.isfinite(base).all():
        raise ValueError("F regime shape base prediction must be finite and row-aligned")
    mask = frame["game_type"].astype(str).str.upper().eq("F").to_numpy()
    output = base.copy()
    correction = np.zeros(len(frame), dtype=np.float64)
    if mask.any():
        subset_heads = {
            name: np.asarray(heads[name], dtype=np.float64).reshape(-1)[mask]
            for name in HEAD_COLUMNS
        }
        raw = build_f_regime_shape_features(
            frame.loc[mask].reset_index(drop=True), subset_heads
        )
        standardized = (raw - artifact.feature_mean) / artifact.feature_scale
        delta = artifact.strength * np.clip(
            standardized @ artifact.coefficients, -artifact.cap, artifact.cap
        )
        output[mask] = np.clip(base[mask] + delta, 1e-6, 1.0 - 1e-6)
        correction[mask] = delta
    if not np.isfinite(output).all():
        raise RuntimeError("F regime shape prediction is non-finite")
    return output, correction, mask


__all__ = [
    "FEATURE_NAMES",
    "FRegimeShapeArtifact",
    "build_f_regime_shape_features",
    "load_f_regime_shape",
    "predict_f_regime_shape",
]
