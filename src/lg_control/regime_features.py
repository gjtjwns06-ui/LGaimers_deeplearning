"""Leakage-safe recent-history and regime features for the V2 neural path.

The builder has two deliberately separate sources of information:

* official row-local ``asof_*`` columns, which are already available before
  the pitch represented by the row; and
* labelled *earlier seasons* supplied to :meth:`fit`, summarized into small
  sufficient-statistic tables.

For a row from season ``s``, target-derived features only use fitted records
whose season is strictly less than ``s``.  Same-season and future-season
labels are never consulted.  ``transform`` never groups, sorts, shifts, or
otherwise compares the rows being transformed, so singleton and batch
transforms are identical.

The official previous-1/3/5-game rates do not include their pitch counts.
Consequently they are not treated as binomial observations.  Beta-binomial
posterior features are only created where a real denominator exists:
``asof_pitcher_n``, ``asof_batter_n``, or earlier-season labelled row counts.
"""

from __future__ import annotations

import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import TARGET_COL
from .preprocessing import validate_preprocessing_schema
from .schema import SchemaError


REGIME_FEATURE_STATE_VERSION = 1
DEFAULT_REGIME_START_SEASON = 2023
PRIOR_SEASON_WINDOWS = (1, 3, 5)
RECENT_GAME_WINDOWS = (1, 3, 5)
DEFAULT_RECENT_MAX_DENOMINATORS = (160, 480, 800)
DEFAULT_RATIONAL_TOLERANCE = 5.1e-7

# alias, source rate, real pre-pitch denominator, prior family
CUMULATIVE_RATE_SPECS = (
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n", "success"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n", "neutral"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n", "neutral"),
    ("pitcher_ball", "asof_pitcher_ball_rate", "asof_pitcher_n", "neutral"),
    ("pitcher_strike", "asof_pitcher_strike_rate", "asof_pitcher_n", "neutral"),
    ("pitcher_fastball", "asof_pitcher_fastball_rate", "asof_pitcher_n", "neutral"),
    ("pitcher_breaking", "asof_pitcher_breaking_rate", "asof_pitcher_n", "neutral"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n", "success"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n", "neutral"),
)

RECENT_RATE_SPECS = {
    "success": {
        1: "asof_pitcher_prev1_game_success_rate",
        3: "asof_pitcher_prev3_game_success_rate",
        5: "asof_pitcher_prev5_game_success_rate",
    },
    "middle": {
        1: "asof_pitcher_prev1_game_middle_rate",
        3: "asof_pitcher_prev3_game_middle_rate",
        5: "asof_pitcher_prev5_game_middle_rate",
    },
}

REGIME_CONTEXT_FEATURE_COLUMNS = (
    "regime_season_offset",
    "regime_post_start",
    "regime_age",
    "regime_game_type_f",
    "regime_season_offset_x_f",
    "regime_post_start_x_f",
    "regime_age_x_f",
    "regime_season_progress",
    "regime_continuous_time",
    "regime_continuous_time_x_f",
)

ASOF_TARGET_FEATURE_COLUMNS = (
    "asof_global_target_rate_bb",
    "asof_global_target_reliability",
    "asof_game_type_target_rate_all_prior_bb",
    "asof_game_type_target_reliability_all_prior",
    *(
        feature
        for window in PRIOR_SEASON_WINDOWS
        for feature in (
            f"asof_game_type_target_rate_prev{window}_seasons_bb",
            f"asof_game_type_target_reliability_prev{window}_seasons",
        )
    ),
    "asof_game_type_target_rate_prev1_minus_prev5",
)

HISTORY_COUNT_FEATURE_COLUMNS = (
    "pitcher_history_log1p_n",
    "pitcher_history_reliability",
    "batter_history_log1p_n",
    "batter_history_reliability",
)

CUMULATIVE_BB_FEATURE_COLUMNS = tuple(
    feature
    for alias, _rate, _count, _prior in CUMULATIVE_RATE_SPECS
    for feature in (
        f"{alias}_bb_mean",
        f"{alias}_bb_std",
        f"{alias}_bb_reliability",
        f"{alias}_observed",
    )
)

RECENT_FEATURE_COLUMNS = tuple(
    feature
    for family in RECENT_RATE_SPECS
    for feature in (
        *(f"recent_{family}_prev{window}_rate" for window in RECENT_GAME_WINDOWS),
        *(f"recent_{family}_prev{window}_observed" for window in RECENT_GAME_WINDOWS),
        f"recent_{family}_observed_fraction",
        f"recent_{family}_weighted_rate",
        f"recent_{family}_dispersion",
        f"recent_{family}_prev1_minus_prev3",
        f"recent_{family}_prev3_minus_prev5",
        f"recent_{family}_prev1_minus_prev5",
        f"recent_{family}_weighted_minus_cumulative",
    )
)

RECENT_RATIONAL_FEATURE_COLUMNS = (
    *(
        feature
        for family in RECENT_RATE_SPECS
        for window in RECENT_GAME_WINDOWS
        for feature in (
            f"recent_{family}_prev{window}_rational_min_denominator",
            f"recent_{family}_prev{window}_rational_min_denominator_log1p",
            f"recent_{family}_prev{window}_rational_reliability_lower_bound",
            f"recent_{family}_prev{window}_rational_error_scaled",
            f"recent_{family}_prev{window}_rational_fit",
        )
    ),
    *(
        feature
        for window in RECENT_GAME_WINDOWS
        for feature in (
            f"recent_prev{window}_joint_rational_min_denominator",
            f"recent_prev{window}_joint_rational_min_denominator_log1p",
            f"recent_prev{window}_joint_rational_reliability_lower_bound",
            f"recent_prev{window}_joint_rational_fit",
        )
    ),
)

REGIME_FEATURE_COLUMNS = (
    *REGIME_CONTEXT_FEATURE_COLUMNS,
    *ASOF_TARGET_FEATURE_COLUMNS,
    *HISTORY_COUNT_FEATURE_COLUMNS,
    *CUMULATIVE_BB_FEATURE_COLUMNS,
    *RECENT_FEATURE_COLUMNS,
    *RECENT_RATIONAL_FEATURE_COLUMNS,
)

if len(REGIME_FEATURE_COLUMNS) != len(set(REGIME_FEATURE_COLUMNS)):
    raise RuntimeError("regime feature names must be unique")


class RegimeFeatureError(ValueError):
    """Raised when V2 feature state or input violates its contract."""


def _exact_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RegimeFeatureError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise RegimeFeatureError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RegimeFeatureError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RegimeFeatureError(f"{name} must be numeric") from error
    if not np.isfinite(result) or (positive and result <= 0.0):
        requirement = "positive and finite" if positive else "finite"
        raise RegimeFeatureError(f"{name} must be {requirement}")
    return result


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    source = frame[column]
    values = pd.to_numeric(source, errors="coerce")
    invalid = source.notna() & values.isna()
    if invalid.any():
        examples = source.loc[invalid].head(3).tolist()
        raise SchemaError(f"{column} contains non-numeric values such as {examples}")
    return values.to_numpy(dtype=np.float64, copy=True)


def _validated_source_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate official inputs and return season, month, and game type arrays."""

    validate_preprocessing_schema(frame)
    season = _numeric(frame, "season")
    if not np.isfinite(season).all() or not np.equal(season, np.floor(season)).all():
        raise RegimeFeatureError("season must contain finite integer years")
    month = _numeric(frame, "game_month")
    game_type = (
        frame["game_type"].astype("string").str.strip().str.upper().to_numpy(dtype=object)
    )
    if not np.isin(game_type, ["F", "R"]).all():
        examples = sorted({str(value) for value in game_type if value not in {"F", "R"}})
        raise RegimeFeatureError(f"game_type must contain only F or R; found {examples[:5]}")
    return season.astype(np.int64), month, game_type


def _beta_binomial(
    rate: np.ndarray,
    count: np.ndarray,
    prior: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return posterior mean/std, reliability, and observed flag row-wise."""

    observed = np.isfinite(rate)
    effective_count = np.where(observed, count, 0.0)
    safe_rate = np.where(observed, np.clip(rate, 0.0, 1.0), prior)
    alpha = effective_count * safe_rate + strength * prior
    beta = effective_count * (1.0 - safe_rate) + strength * (1.0 - prior)
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    std = np.sqrt(np.maximum(variance, 0.0))
    reliability = effective_count / (effective_count + strength)
    return mean, std, reliability, observed.astype(np.float64)


@lru_cache(maxsize=16_384)
def _rational_candidate(value: float, max_denominator: int) -> tuple[int, float]:
    """Return a cached bounded-denominator candidate and absolute error."""

    # The official CSV stores these rates as decimal text.  Converting the
    # short, stable decimal representation instead of the binary float avoids
    # inventing a very large denominator from IEEE-754 noise.
    fraction = Fraction(str(float(value))).limit_denominator(max_denominator)
    error = abs(float(fraction) - float(value))
    return int(fraction.denominator), float(error)


def _infer_rational_lower_bound(
    rate: np.ndarray,
    *,
    max_denominator: int,
    tolerance: float,
    reliability_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer a conservative minimum denominator from one rounded rate.

    A successful reconstruction is only a denominator lower bound: the true
    pitch count may be any multiple producing the same reduced fraction.
    Failed or missing reconstructions return denominator/reliability zero.
    """

    denominator = np.zeros(len(rate), dtype=np.float64)
    error_scaled = np.zeros(len(rate), dtype=np.float64)
    fit = np.zeros(len(rate), dtype=np.float64)
    observed_positions = np.flatnonzero(np.isfinite(rate))
    if observed_positions.size:
        unique, inverse = np.unique(rate[observed_positions], return_inverse=True)
        unique_denominator = np.zeros(len(unique), dtype=np.float64)
        unique_error = np.zeros(len(unique), dtype=np.float64)
        unique_fit = np.zeros(len(unique), dtype=np.float64)
        for index, value in enumerate(unique):
            candidate, error = _rational_candidate(float(value), max_denominator)
            accepted = error <= tolerance
            unique_denominator[index] = float(candidate) if accepted else 0.0
            unique_error[index] = min(error / tolerance, 10.0)
            unique_fit[index] = float(accepted)
        denominator[observed_positions] = unique_denominator[inverse]
        error_scaled[observed_positions] = unique_error[inverse]
        fit[observed_positions] = unique_fit[inverse]
    reliability = denominator / (denominator + reliability_strength)
    return denominator, np.log1p(denominator), reliability, error_scaled, fit


class RegimeFeatureBuilderV2:
    """Fit earlier-season sufficient statistics and create row-local V2 features.

    ``fit`` is target-dependent and must therefore receive only the training
    partition of a temporal fold.  The stored state contains counts and
    successes by season and game type, never individual labels.  During
    ``transform``, a row from season ``s`` can only see statistics from years
    less than ``s`` even if later years are present in the fitted state.
    """

    feature_columns = REGIME_FEATURE_COLUMNS

    def __init__(
        self,
        *,
        regime_start_season: int = DEFAULT_REGIME_START_SEASON,
        history_prior_strength: float = 64.0,
        target_prior_strength: float = 256.0,
        neutral_prior: float = 0.5,
        recent_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        recent_max_denominators: tuple[int, int, int] = DEFAULT_RECENT_MAX_DENOMINATORS,
        rational_tolerance: float = DEFAULT_RATIONAL_TOLERANCE,
    ) -> None:
        self.regime_start_season = _exact_int(
            regime_start_season, name="regime_start_season", minimum=1900
        )
        self.history_prior_strength = _finite_float(
            history_prior_strength, name="history_prior_strength", positive=True
        )
        self.target_prior_strength = _finite_float(
            target_prior_strength, name="target_prior_strength", positive=True
        )
        self.neutral_prior = _finite_float(neutral_prior, name="neutral_prior")
        if not 0.0 < self.neutral_prior < 1.0:
            raise RegimeFeatureError("neutral_prior must be strictly inside (0, 1)")
        if not isinstance(recent_weights, (tuple, list)) or len(recent_weights) != 3:
            raise RegimeFeatureError("recent_weights must contain three values for 1/3/5")
        weights = np.asarray(recent_weights, dtype=np.float64)
        if not np.isfinite(weights).all() or (weights <= 0.0).any():
            raise RegimeFeatureError("recent_weights must be positive and finite")
        self.recent_weights = tuple(float(value) for value in weights / weights.sum())
        if not isinstance(recent_max_denominators, (tuple, list)) or len(
            recent_max_denominators
        ) != 3:
            raise RegimeFeatureError(
                "recent_max_denominators must contain three values for 1/3/5"
            )
        denominators = tuple(
            _exact_int(value, name="recent maximum denominator", minimum=1)
            for value in recent_max_denominators
        )
        if tuple(sorted(denominators)) != denominators:
            raise RegimeFeatureError("recent_max_denominators must be non-decreasing")
        self.recent_max_denominators = denominators
        self.rational_tolerance = _finite_float(
            rational_tolerance, name="rational_tolerance", positive=True
        )
        self.target_statistics_: dict[tuple[int, str], tuple[int, int]] = {}
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "RegimeFeatureBuilderV2":
        """Store target sufficient statistics; callers must provide train rows only."""

        if TARGET_COL not in frame.columns:
            raise RegimeFeatureError(f"fit requires labelled column {TARGET_COL!r}")
        if len(frame) == 0:
            raise RegimeFeatureError("cannot fit regime features on an empty frame")
        season, _month, game_type = _validated_source_arrays(frame)
        target = _numeric(frame, TARGET_COL)
        if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
            raise RegimeFeatureError(f"{TARGET_COL} must contain only binary 0/1 values")

        table = pd.DataFrame(
            {"season": season, "game_type": game_type, "target": target.astype(np.int64)}
        )
        grouped = table.groupby(["season", "game_type"], sort=True, observed=True)[
            "target"
        ].agg(["size", "sum"])
        self.target_statistics_ = {
            (int(year), str(kind)): (int(row["size"]), int(row["sum"]))
            for (year, kind), row in grouped.iterrows()
        }
        self.is_fitted_ = True
        return self

    def _sum_target_statistics(
        self,
        *,
        before_season: int,
        game_type: str | None = None,
        first_season: int | None = None,
    ) -> tuple[int, int]:
        rows = 0
        successes = 0
        for (year, kind), (count, positive) in self.target_statistics_.items():
            if year >= before_season:
                continue
            if first_season is not None and year < first_season:
                continue
            if game_type is not None and kind != game_type:
                continue
            rows += count
            successes += positive
        return rows, successes

    def _asof_target_features(
        self, season: np.ndarray, game_type: np.ndarray
    ) -> dict[str, np.ndarray]:
        result = {
            column: np.empty(len(season), dtype=np.float64)
            for column in ASOF_TARGET_FEATURE_COLUMNS
        }
        for year in np.unique(season):
            year_mask = season == year
            global_rows, global_successes = self._sum_target_statistics(
                before_season=int(year)
            )
            global_rate = (
                global_successes + self.target_prior_strength * self.neutral_prior
            ) / (global_rows + self.target_prior_strength)
            global_reliability = global_rows / (global_rows + self.target_prior_strength)
            result["asof_global_target_rate_bb"][year_mask] = global_rate
            result["asof_global_target_reliability"][year_mask] = global_reliability

            for kind in ("F", "R"):
                mask = year_mask & (game_type == kind)
                if not mask.any():
                    continue
                all_rows, all_successes = self._sum_target_statistics(
                    before_season=int(year), game_type=kind
                )
                all_rate = (
                    all_successes + self.target_prior_strength * global_rate
                ) / (all_rows + self.target_prior_strength)
                all_reliability = all_rows / (all_rows + self.target_prior_strength)
                result["asof_game_type_target_rate_all_prior_bb"][mask] = all_rate
                result["asof_game_type_target_reliability_all_prior"][mask] = (
                    all_reliability
                )

                window_rates: dict[int, float] = {}
                for window in PRIOR_SEASON_WINDOWS:
                    rows, successes = self._sum_target_statistics(
                        before_season=int(year),
                        first_season=int(year) - window,
                        game_type=kind,
                    )
                    rate = (
                        successes + self.target_prior_strength * global_rate
                    ) / (rows + self.target_prior_strength)
                    reliability = rows / (rows + self.target_prior_strength)
                    result[
                        f"asof_game_type_target_rate_prev{window}_seasons_bb"
                    ][mask] = rate
                    result[
                        f"asof_game_type_target_reliability_prev{window}_seasons"
                    ][mask] = reliability
                    window_rates[window] = rate
                result["asof_game_type_target_rate_prev1_minus_prev5"][mask] = (
                    window_rates[1] - window_rates[5]
                )
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create finite features without reading labels or peer transform rows."""

        self._require_fitted()
        season, month, game_type = _validated_source_arrays(frame)
        row_count = len(frame)
        result: dict[str, np.ndarray] = {}

        offset = season.astype(np.float64) - float(self.regime_start_season)
        post = (season >= self.regime_start_season).astype(np.float64)
        age = np.maximum(offset, 0.0)
        game_type_f = (game_type == "F").astype(np.float64)
        progress = (month - 1.0) / 11.0
        continuous_time = offset + (month - 1.0) / 12.0
        result.update(
            {
                "regime_season_offset": offset,
                "regime_post_start": post,
                "regime_age": age,
                "regime_game_type_f": game_type_f,
                "regime_season_offset_x_f": offset * game_type_f,
                "regime_post_start_x_f": post * game_type_f,
                "regime_age_x_f": age * game_type_f,
                "regime_season_progress": progress,
                "regime_continuous_time": continuous_time,
                "regime_continuous_time_x_f": continuous_time * game_type_f,
            }
        )
        result.update(self._asof_target_features(season, game_type))
        success_prior = result["asof_global_target_rate_bb"]
        neutral_prior = np.full(row_count, self.neutral_prior, dtype=np.float64)

        pitcher_n = _numeric(frame, "asof_pitcher_n")
        batter_n = _numeric(frame, "asof_batter_n")
        for name, count in (("pitcher", pitcher_n), ("batter", batter_n)):
            if not np.isfinite(count).all() or (count < 0.0).any() or not np.equal(
                count, np.floor(count)
            ).all():
                raise RegimeFeatureError(f"asof_{name}_n must be finite non-negative integers")
            result[f"{name}_history_log1p_n"] = np.log1p(count)
            result[f"{name}_history_reliability"] = count / (
                count + self.history_prior_strength
            )

        for alias, rate_column, count_column, prior_family in CUMULATIVE_RATE_SPECS:
            rate = _numeric(frame, rate_column)
            finite = np.isfinite(rate)
            if (finite & ((rate < 0.0) | (rate > 1.0))).any():
                raise RegimeFeatureError(f"{rate_column} must be in [0, 1] when present")
            count = pitcher_n if count_column == "asof_pitcher_n" else batter_n
            prior = success_prior if prior_family == "success" else neutral_prior
            mean, std, reliability, observed = _beta_binomial(
                rate, count, prior, self.history_prior_strength
            )
            result[f"{alias}_bb_mean"] = mean
            result[f"{alias}_bb_std"] = std
            result[f"{alias}_bb_reliability"] = reliability
            result[f"{alias}_observed"] = observed

        cumulative_columns = {
            "success": "pitcher_success_bb_mean",
            "middle": "pitcher_middle_bb_mean",
        }
        weights = np.asarray(self.recent_weights, dtype=np.float64)
        for family, columns in RECENT_RATE_SPECS.items():
            raw_values: list[np.ndarray] = []
            observed_values: list[np.ndarray] = []
            prior = success_prior if family == "success" else neutral_prior
            for window in RECENT_GAME_WINDOWS:
                raw = _numeric(frame, columns[window])
                observed = np.isfinite(raw)
                if (observed & ((raw < 0.0) | (raw > 1.0))).any():
                    raise RegimeFeatureError(
                        f"{columns[window]} must be in [0, 1] when present"
                    )
                filled = np.where(observed, raw, prior)
                result[f"recent_{family}_prev{window}_rate"] = filled
                result[f"recent_{family}_prev{window}_observed"] = observed.astype(
                    np.float64
                )
                raw_values.append(filled)
                observed_values.append(observed.astype(np.float64))

            values = np.column_stack(raw_values)
            observed = np.column_stack(observed_values)
            weighted_observation = observed * weights
            weight_sum = weighted_observation.sum(axis=1)
            cumulative = result[cumulative_columns[family]]
            weighted = np.divide(
                (values * weighted_observation).sum(axis=1),
                weight_sum,
                out=cumulative.copy(),
                where=weight_sum > 0.0,
            )
            observed_count = observed.sum(axis=1)
            observed_min = np.min(np.where(observed > 0.0, values, np.inf), axis=1)
            observed_max = np.max(np.where(observed > 0.0, values, -np.inf), axis=1)
            dispersion = np.where(observed_count > 1.0, observed_max - observed_min, 0.0)

            result[f"recent_{family}_observed_fraction"] = observed_count / 3.0
            result[f"recent_{family}_weighted_rate"] = weighted
            result[f"recent_{family}_dispersion"] = dispersion
            for left, right in ((1, 3), (3, 5), (1, 5)):
                left_index = RECENT_GAME_WINDOWS.index(left)
                right_index = RECENT_GAME_WINDOWS.index(right)
                both = (observed[:, left_index] > 0.0) & (
                    observed[:, right_index] > 0.0
                )
                delta = np.where(
                    both, values[:, left_index] - values[:, right_index], 0.0
                )
                result[f"recent_{family}_prev{left}_minus_prev{right}"] = delta
            result[f"recent_{family}_weighted_minus_cumulative"] = weighted - cumulative

        # Infer only a conservative denominator lower bound from each rounded
        # decimal.  It is intentionally not fed into the beta-binomial means:
        # the actual pitch count is not present in the official schema.
        inferred_denominators: dict[tuple[str, int], np.ndarray] = {}
        inferred_fits: dict[tuple[str, int], np.ndarray] = {}
        for family, columns in RECENT_RATE_SPECS.items():
            for position, window in enumerate(RECENT_GAME_WINDOWS):
                raw = _numeric(frame, columns[window])
                denominator, log_denominator, reliability, error, fit = (
                    _infer_rational_lower_bound(
                        raw,
                        max_denominator=self.recent_max_denominators[position],
                        tolerance=self.rational_tolerance,
                        reliability_strength=self.history_prior_strength,
                    )
                )
                prefix = f"recent_{family}_prev{window}_rational"
                result[f"{prefix}_min_denominator"] = denominator
                result[f"{prefix}_min_denominator_log1p"] = log_denominator
                result[f"{prefix}_reliability_lower_bound"] = reliability
                result[f"{prefix}_error_scaled"] = error
                result[f"{prefix}_fit"] = fit
                inferred_denominators[(family, window)] = denominator
                inferred_fits[(family, window)] = fit

        for window in RECENT_GAME_WINDOWS:
            left = inferred_denominators[("success", window)].astype(np.int64)
            right = inferred_denominators[("middle", window)].astype(np.int64)
            both = (inferred_fits[("success", window)] > 0.0) & (
                inferred_fits[("middle", window)] > 0.0
            )
            joint = np.zeros(row_count, dtype=np.float64)
            joint[both] = np.lcm(left[both], right[both]).astype(np.float64)
            max_denominator = self.recent_max_denominators[
                RECENT_GAME_WINDOWS.index(window)
            ]
            accepted = both & (joint <= float(max_denominator))
            joint = np.where(accepted, joint, 0.0)
            prefix = f"recent_prev{window}_joint_rational"
            result[f"{prefix}_min_denominator"] = joint
            result[f"{prefix}_min_denominator_log1p"] = np.log1p(joint)
            result[f"{prefix}_reliability_lower_bound"] = joint / (
                joint + self.history_prior_strength
            )
            result[f"{prefix}_fit"] = accepted.astype(np.float64)

        output = pd.DataFrame(result, index=frame.index)
        missing = sorted(set(REGIME_FEATURE_COLUMNS).difference(output.columns))
        unexpected = sorted(set(output.columns).difference(REGIME_FEATURE_COLUMNS))
        if missing or unexpected:
            raise RuntimeError(
                "regime feature implementation drifted from the public contract; "
                f"missing={missing}, unexpected={unexpected}"
            )
        output = output.loc[:, REGIME_FEATURE_COLUMNS]
        values = output.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise RuntimeError("regime feature transform produced non-finite values")
        return output

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    @property
    def fit_seasons(self) -> tuple[int, ...]:
        self._require_fitted()
        return tuple(sorted({year for year, _kind in self.target_statistics_}))

    def to_state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_version": REGIME_FEATURE_STATE_VERSION,
            "builder_class": type(self).__name__,
            "regime_start_season": self.regime_start_season,
            "history_prior_strength": self.history_prior_strength,
            "target_prior_strength": self.target_prior_strength,
            "neutral_prior": self.neutral_prior,
            "recent_weights": list(self.recent_weights),
            "recent_max_denominators": list(self.recent_max_denominators),
            "rational_tolerance": self.rational_tolerance,
            "feature_columns": list(REGIME_FEATURE_COLUMNS),
            "target_statistics": [
                {
                    "season": year,
                    "game_type": kind,
                    "rows": rows,
                    "successes": successes,
                }
                for (year, kind), (rows, successes) in sorted(
                    self.target_statistics_.items()
                )
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "RegimeFeatureBuilderV2":
        if not isinstance(payload, dict):
            raise TypeError("regime feature state must be a dictionary")
        if payload.get("state_version") != REGIME_FEATURE_STATE_VERSION:
            raise RegimeFeatureError("unsupported regime feature state version")
        if payload.get("builder_class") != cls.__name__:
            raise RegimeFeatureError("serialized regime feature builder class does not match")
        if tuple(payload.get("feature_columns", ())) != REGIME_FEATURE_COLUMNS:
            raise RegimeFeatureError("serialized regime feature columns do not match code")
        raw_weights = payload.get("recent_weights")
        if not isinstance(raw_weights, list):
            raise RegimeFeatureError("serialized recent_weights must be a list")
        obj = cls(
            regime_start_season=_exact_int(
                payload.get("regime_start_season"), name="regime_start_season", minimum=1900
            ),
            history_prior_strength=_finite_float(
                payload.get("history_prior_strength"),
                name="history_prior_strength",
                positive=True,
            ),
            target_prior_strength=_finite_float(
                payload.get("target_prior_strength"),
                name="target_prior_strength",
                positive=True,
            ),
            neutral_prior=_finite_float(
                payload.get("neutral_prior"), name="neutral_prior"
            ),
            recent_weights=tuple(raw_weights),
            recent_max_denominators=tuple(payload.get("recent_max_denominators", ())),
            rational_tolerance=_finite_float(
                payload.get("rational_tolerance"),
                name="rational_tolerance",
                positive=True,
            ),
        )
        raw_statistics = payload.get("target_statistics")
        if not isinstance(raw_statistics, list) or not raw_statistics:
            raise RegimeFeatureError("serialized target statistics must be a non-empty list")
        restored: dict[tuple[int, str], tuple[int, int]] = {}
        for row in raw_statistics:
            if not isinstance(row, dict):
                raise RegimeFeatureError("serialized target statistic is malformed")
            year = _exact_int(row.get("season"), name="target statistic season", minimum=1900)
            kind = row.get("game_type")
            if kind not in {"F", "R"}:
                raise RegimeFeatureError("serialized target statistic game_type must be F or R")
            rows = _exact_int(row.get("rows"), name="target statistic rows", minimum=1)
            successes = _exact_int(
                row.get("successes"), name="target statistic successes", minimum=0
            )
            if successes > rows:
                raise RegimeFeatureError("serialized successes cannot exceed rows")
            key = (year, str(kind))
            if key in restored:
                raise RegimeFeatureError("serialized target statistics contain duplicates")
            restored[key] = (rows, successes)
        obj.target_statistics_ = restored
        obj.is_fitted_ = True
        return obj

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_state_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegimeFeatureBuilderV2":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("RegimeFeatureBuilderV2 must be fitted before transform")


__all__ = [
    "REGIME_FEATURE_STATE_VERSION",
    "DEFAULT_REGIME_START_SEASON",
    "PRIOR_SEASON_WINDOWS",
    "RECENT_GAME_WINDOWS",
    "DEFAULT_RECENT_MAX_DENOMINATORS",
    "DEFAULT_RATIONAL_TOLERANCE",
    "CUMULATIVE_RATE_SPECS",
    "RECENT_RATE_SPECS",
    "REGIME_FEATURE_COLUMNS",
    "RegimeFeatureError",
    "RegimeFeatureBuilderV2",
]
