"""Row-local conditional fusion for the v17 and recent-regime predictors.

The gate never predicts the target on its own.  It only controls how far the
final probability may move from the stronger v17 anchor toward the recent
deep-learning prediction::

    p = p_v17 + weight(state) * (p_dl - p_v17)

The state is deliberately small and row-local: game type, count, and the
direction of model disagreement.  This keeps the learned complementarity
surface inspectable and safe under shuffled or single-row inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd


GATE_KEYS = ("game_type", "count", "direction")


@dataclass(frozen=True)
class ComplementaryGateArtifact:
    """Frozen conditional blend surface."""

    table: pd.DataFrame
    fallback_weight: float
    maximum_weight: float
    keys: tuple[str, ...] = GATE_KEYS

    def __post_init__(self) -> None:
        if not 0.0 <= self.fallback_weight <= self.maximum_weight <= 1.0:
            raise ValueError("gate weights must satisfy 0 <= fallback <= maximum <= 1")
        required = {*self.keys, "weight"}
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(f"gate table is missing columns: {sorted(missing)}")
        if self.table.duplicated(list(self.keys)).any():
            raise ValueError("gate table keys must be unique")
        weights = pd.to_numeric(self.table["weight"], errors="raise").to_numpy(float)
        if not np.isfinite(weights).all():
            raise ValueError("gate table contains non-finite weights")
        if np.any(weights < 0.0) or np.any(weights > self.maximum_weight + 1e-12):
            raise ValueError("gate table weight is outside its configured range")


def _canonical_game_type(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.upper()
    numeric = pd.to_numeric(values, errors="coerce")
    text = text.mask(numeric.eq(0), "F").mask(numeric.eq(1), "R")
    if not set(text.dropna().unique()).issubset({"F", "R"}):
        raise ValueError("game_type must use F/R or the frozen 0=F, 1=R mapping")
    return text


def build_gate_query(
    frame: pd.DataFrame,
    prediction_v17: np.ndarray,
    prediction_dl: np.ndarray,
) -> pd.DataFrame:
    """Create the frozen row-local gate state without changing row order."""

    required = {"game_type", "balls_before", "strikes_before"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input frame is missing gate columns: {sorted(missing)}")
    p_v17 = np.asarray(prediction_v17, dtype=np.float64)
    p_dl = np.asarray(prediction_dl, dtype=np.float64)
    if p_v17.shape != (len(frame),) or p_dl.shape != (len(frame),):
        raise ValueError("base prediction arrays must be one-dimensional and row-aligned")
    if not np.isfinite(p_v17).all() or not np.isfinite(p_dl).all():
        raise ValueError("base predictions must be finite")

    balls = pd.to_numeric(frame["balls_before"], errors="raise").astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").astype(int)
    query = pd.DataFrame(
        {
            "game_type": _canonical_game_type(frame["game_type"]).to_numpy(),
            "count": (balls.astype(str) + "-" + strikes.astype(str)).to_numpy(),
            "direction": np.where(p_dl >= p_v17, "up", "down"),
            "_row_position": np.arange(len(frame), dtype=np.int64),
        }
    )
    return query


def predict_complementary_hybrid(
    frame: pd.DataFrame,
    prediction_v17: np.ndarray,
    prediction_dl: np.ndarray,
    artifact: ComplementaryGateArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the gate and return ``(probability, weight)`` in input row order."""

    p_v17 = np.asarray(prediction_v17, dtype=np.float64)
    p_dl = np.asarray(prediction_dl, dtype=np.float64)
    query = build_gate_query(frame, p_v17, p_dl)
    lookup = artifact.table[[*artifact.keys, "weight"]].copy()
    merged = query.merge(lookup, on=list(artifact.keys), how="left", sort=False)
    if len(merged) != len(frame):
        raise RuntimeError("gate lookup changed the row count")
    merged = merged.sort_values("_row_position", kind="stable")
    weights = (
        pd.to_numeric(merged["weight"], errors="coerce")
        .fillna(artifact.fallback_weight)
        .clip(0.0, artifact.maximum_weight)
        .to_numpy(dtype=np.float64)
    )
    probability = np.clip(p_v17 + weights * (p_dl - p_v17), 1e-6, 1.0 - 1e-6)
    if not np.isfinite(probability).all():
        raise RuntimeError("conditional hybrid produced non-finite probabilities")
    return probability, weights


def save_complementary_gate(
    directory: Path,
    artifact: ComplementaryGateArtifact,
    metadata: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    artifact.table.to_csv(directory / "gate.csv", index=False, encoding="utf-8")
    payload: dict[str, Any] = {
        "kind": "v17_recent_regime_complementary_gate_v1",
        "keys": list(artifact.keys),
        "fallback_weight": artifact.fallback_weight,
        "maximum_weight": artifact.maximum_weight,
        "formula": "p_v17 + weight(state) * (p_dl - p_v17)",
    }
    if metadata:
        payload["metadata"] = metadata
    (directory / "gate_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_complementary_gate(directory: Path) -> ComplementaryGateArtifact:
    payload = json.loads((directory / "gate_config.json").read_text(encoding="utf-8"))
    if payload.get("kind") != "v17_recent_regime_complementary_gate_v1":
        raise ValueError("unexpected complementary gate kind")
    keys = tuple(payload["keys"])
    if keys != GATE_KEYS:
        raise ValueError("deployed complementary gate keys differ from the frozen contract")
    table = pd.read_csv(directory / "gate.csv", encoding="utf-8")
    return ComplementaryGateArtifact(
        table=table,
        fallback_weight=float(payload["fallback_weight"]),
        maximum_weight=float(payload["maximum_weight"]),
        keys=keys,
    )
