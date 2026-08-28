"""Strict training-row policies for the post-2022 target regime.

The 2023 boundary changed the meaning/distribution of ``game_type=F``.  This
module makes it impossible for a recent-regime experiment to silently learn
from the obsolete pre-2023 F labels.  Historical R rows may be retained as a
variance-reduction pool because their target relationship stayed much more
stable, but every F row must come from the new regime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


HistoryPolicy = Literal["recent_only", "stable_r_plus_recent"]


@dataclass(frozen=True)
class RecentRegimePolicy:
    """Auditable row-selection and weighting contract.

    ``cutoff_year`` is exclusive.  For a 2024 validation fold it is 2024;
    for the final 2023--2024 fit it is 2025.
    """

    regime_start_year: int = 2023
    history_policy: HistoryPolicy = "stable_r_plus_recent"
    stable_history_game_type: str = "R"
    historical_r_decay: float = 0.60
    recent_decay: float = 0.85

    def __post_init__(self) -> None:
        if isinstance(self.regime_start_year, bool) or self.regime_start_year < 1900:
            raise ValueError("regime_start_year must be an integer year >= 1900")
        if self.history_policy not in {"recent_only", "stable_r_plus_recent"}:
            raise ValueError("unsupported history_policy")
        stable = str(self.stable_history_game_type).strip().upper()
        if stable != "R":
            raise ValueError("only historical game_type=R is allowed")
        object.__setattr__(self, "stable_history_game_type", stable)
        for name in ("historical_r_decay", "recent_decay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_recent_regime_training_rows(
    frame: pd.DataFrame,
    *,
    cutoff_year: int,
    policy: RecentRegimePolicy,
) -> pd.DataFrame:
    """Select fit rows while hard-excluding obsolete F labels."""

    required = {"season", "game_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"recent-regime selection is missing columns: {missing}")
    if isinstance(cutoff_year, bool) or cutoff_year <= policy.regime_start_year:
        raise ValueError("cutoff_year must be after regime_start_year")

    season = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
    game_type = frame["game_type"].astype("string").str.strip().str.upper()
    before_cutoff = season < int(cutoff_year)
    recent = season >= policy.regime_start_year
    if policy.history_policy == "recent_only":
        keep = before_cutoff & recent
    else:
        historical_r = (season < policy.regime_start_year) & game_type.eq(
            policy.stable_history_game_type
        ).to_numpy()
        keep = before_cutoff & (recent | historical_r)

    selected = frame.loc[keep].copy()
    if selected.empty:
        raise ValueError("recent-regime policy selected no training rows")
    selected_season = pd.to_numeric(selected["season"], errors="raise")
    selected_game_type = (
        selected["game_type"].astype("string").str.strip().str.upper()
    )
    forbidden = (selected_season < policy.regime_start_year) & selected_game_type.eq("F")
    if forbidden.any():
        raise RuntimeError("obsolete pre-regime F rows escaped the hard exclusion")
    if (selected_season >= cutoff_year).any():
        raise RuntimeError("future rows escaped the cutoff")
    return selected


def select_recent_regime_inner_partitions(
    frame: pd.DataFrame,
    *,
    validation_year: int,
    validation_start_month: int,
    policy: RecentRegimePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a within-regime chronological stop split.

    The validation season is also the first new-regime season.  Rows before
    ``validation_start_month`` form the recent fit partition; later rows are
    the stop partition.  Under ``stable_r_plus_recent`` only historical R is
    prepended to the fit partition.  Historical F can enter neither side.
    """

    required = {"season", "game_month", "game_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"recent-regime inner split is missing columns: {missing}")
    if validation_year != policy.regime_start_year:
        raise ValueError("inner validation_year must equal regime_start_year")
    if isinstance(validation_start_month, bool) or not 2 <= validation_start_month <= 12:
        raise ValueError("validation_start_month must be an integer in [2, 12]")

    season = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(dtype=np.int64)
    game_type = frame["game_type"].astype("string").str.strip().str.upper()
    early_recent = (season == validation_year) & (month < validation_start_month)
    stop = (season == validation_year) & (month >= validation_start_month)
    if policy.history_policy == "recent_only":
        fit = early_recent
    else:
        historical_r = (season < validation_year) & game_type.eq(
            policy.stable_history_game_type
        ).to_numpy()
        fit = early_recent | historical_r

    fit_frame = frame.loc[fit].copy()
    stop_frame = frame.loc[stop].copy()
    if fit_frame.empty or stop_frame.empty:
        raise ValueError("recent-regime inner split produced an empty partition")
    fit_season = pd.to_numeric(fit_frame["season"], errors="raise")
    fit_type = fit_frame["game_type"].astype("string").str.strip().str.upper()
    if ((fit_season < validation_year) & fit_type.eq("F")).any():
        raise RuntimeError("obsolete F rows escaped the inner-fit hard exclusion")
    if not pd.to_numeric(stop_frame["season"], errors="raise").eq(validation_year).all():
        raise RuntimeError("inner stop contains rows from another season")
    return fit_frame, stop_frame


def recent_regime_weights(
    frame: pd.DataFrame,
    *,
    cutoff_year: int,
    policy: RecentRegimePolicy,
) -> np.ndarray:
    """Return normalized weights without reviving excluded historical F rows."""

    season = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
    game_type = frame["game_type"].astype("string").str.strip().str.upper()
    age = (int(cutoff_year) - 1) - season
    if (age < 0).any():
        raise ValueError("training rows must precede cutoff_year")
    obsolete_f = (season < policy.regime_start_year) & game_type.eq("F").to_numpy()
    if obsolete_f.any():
        raise ValueError("weights refuse obsolete pre-regime F rows")

    recent = season >= policy.regime_start_year
    decay = np.where(recent, policy.recent_decay, policy.historical_r_decay)
    weights = np.power(decay, age).astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("recent-regime weights must be finite and positive")
    return weights / float(weights.mean())


def recent_regime_audit(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    policy: RecentRegimePolicy,
) -> dict[str, Any]:
    """Summarise the exact labels exposed to preprocessing and optimisation."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) != len(frame):
        raise ValueError("audit weights are not row-aligned")
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    game_type = frame["game_type"].astype("string").str.strip().str.upper()
    obsolete_f = (season < policy.regime_start_year) & game_type.eq("F")
    by_season_game_type: dict[str, dict[str, dict[str, float | int]]] = {}
    for (year, group), positions in frame.groupby([season, game_type], sort=True).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        target_rate = None
        if "control_success" in frame.columns:
            target_rate = float(
                pd.to_numeric(frame.iloc[index]["control_success"], errors="raise").mean()
            )
        by_season_game_type.setdefault(str(int(year)), {})[str(group)] = {
            "rows": int(len(index)),
            "weight_sum": float(values[index].sum()),
            "target_rate": target_rate,
        }
    return {
        "policy": policy.to_dict(),
        "rows": int(len(frame)),
        "seasons": sorted(int(value) for value in season.unique()),
        "obsolete_pre_regime_f_rows": int(obsolete_f.sum()),
        "by_season_game_type": by_season_game_type,
    }


__all__ = (
    "HistoryPolicy",
    "RecentRegimePolicy",
    "recent_regime_audit",
    "recent_regime_weights",
    "select_recent_regime_inner_partitions",
    "select_recent_regime_training_rows",
)
