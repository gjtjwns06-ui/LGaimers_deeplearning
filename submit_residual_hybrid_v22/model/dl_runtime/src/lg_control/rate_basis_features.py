"""Compact row-local nonlinear rate bases for the neural tabular path.

The official ``asof_*`` rates are available before the represented pitch, but
their useful relationships are nonlinear and depend on history size.  This
module adds a deliberately small deterministic basis without using labels,
entity identifiers, other transform rows, or inferred recent-game counts.

Only the outer standardisation is fitted.  Consequently all learned state
comes from the caller-supplied temporal training fold and transforming a row
alone is identical to transforming it in any batch or order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .model import TabularFeatureLayout
from .preprocessing import DeepTabularPreprocessor, validate_preprocessing_schema


RATE_BASIS_STATE_VERSION = 1
_RATE_EPSILON = 1.0e-4

# alias, rate column, real cumulative count column
CUMULATIVE_RATE_SPECS = (
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("pitcher_ball", "asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("pitcher_strike", "asof_pitcher_strike_rate", "asof_pitcher_n"),
    (
        "pitcher_fastball",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_pitchmix_n",
    ),
    (
        "pitcher_breaking",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_pitchmix_n",
    ),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
)

_RECENT_SPECS = {
    "success": (
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_success_rate",
    ),
    "middle": (
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_pitcher_middle_rate",
    ),
}

RATE_BASIS_FEATURE_COLUMNS = (
    *(
        name
        for alias, _rate, _count in CUMULATIVE_RATE_SPECS
        for name in (
            f"rb_{alias}_logit",
            f"rb_{alias}_centered_square",
            f"rb_{alias}_uncertainty",
            f"rb_{alias}_reliable_deviation",
        )
    ),
    "rb_pitcher_reliability_short",
    "rb_pitcher_reliability_long",
    "rb_batter_reliability_short",
    "rb_batter_reliability_long",
    *(
        name
        for family in _RECENT_SPECS
        for name in (
            f"rb_pitcher_recent_{family}_weighted",
            f"rb_pitcher_recent_{family}_prev1_minus_prev5",
            f"rb_pitcher_recent_{family}_curvature",
            f"rb_pitcher_recent_{family}_dispersion",
            f"rb_pitcher_recent_{family}_weighted_minus_cumulative",
        )
    ),
    "rb_compat_success_product",
    "rb_compat_success_mean",
    "rb_compat_middle_product",
    "rb_compat_pitcher_control_composite",
    "rb_compat_pitch_mix_entropy",
    "rb_game_f_x_pitcher_success",
    "rb_game_f_x_batter_success",
    "rb_game_f_x_recent_success_delta",
    "rb_game_f_x_pitcher_control",
    "rb_game_f_x_pitch_mix_entropy",
)

if len(RATE_BASIS_FEATURE_COLUMNS) != 60:
    raise RuntimeError("compact rate basis must contain exactly 60 features")
if len(set(RATE_BASIS_FEATURE_COLUMNS)) != len(RATE_BASIS_FEATURE_COLUMNS):
    raise RuntimeError("compact rate-basis feature names must be unique")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )


def _safe_rate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(values)
    safe = np.where(observed, np.clip(values, 0.0, 1.0), 0.5)
    return safe, observed


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CompactRateBasisBuilder:
    """Create 60 deterministic nonlinear features from pre-pitch columns."""

    feature_columns = RATE_BASIS_FEATURE_COLUMNS

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_preprocessing_schema(frame)
        output: dict[str, np.ndarray] = {}

        pitcher_n = np.maximum(_numeric(frame, "asof_pitcher_n"), 0.0)
        batter_n = np.maximum(_numeric(frame, "asof_batter_n"), 0.0)
        for alias, rate_column, count_column in CUMULATIVE_RATE_SPECS:
            raw = _numeric(frame, rate_column)
            rate, observed = _safe_rate(raw)
            count = np.maximum(_numeric(frame, count_column), 0.0)
            effective_count = np.where(observed, count, 0.0)
            clipped = np.clip(rate, _RATE_EPSILON, 1.0 - _RATE_EPSILON)
            output[f"rb_{alias}_logit"] = np.where(
                observed, np.log(clipped / (1.0 - clipped)), 0.0
            )
            output[f"rb_{alias}_centered_square"] = np.where(
                observed, np.square(rate - 0.5), 0.0
            )
            output[f"rb_{alias}_uncertainty"] = np.where(
                observed,
                np.sqrt(rate * (1.0 - rate) / (effective_count + 1.0)),
                0.0,
            )
            output[f"rb_{alias}_reliable_deviation"] = (
                effective_count / (effective_count + 50.0) * (rate - 0.5)
            )

        output["rb_pitcher_reliability_short"] = pitcher_n / (pitcher_n + 20.0)
        output["rb_pitcher_reliability_long"] = pitcher_n / (pitcher_n + 100.0)
        output["rb_batter_reliability_short"] = batter_n / (batter_n + 20.0)
        output["rb_batter_reliability_long"] = batter_n / (batter_n + 100.0)

        recent_values: dict[str, dict[str, np.ndarray]] = {}
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        for family, (prev1_col, prev3_col, prev5_col, cumulative_col) in _RECENT_SPECS.items():
            raw = np.column_stack(
                [_numeric(frame, prev1_col), _numeric(frame, prev3_col), _numeric(frame, prev5_col)]
            )
            observed = np.isfinite(raw)
            safe = np.where(observed, np.clip(raw, 0.0, 1.0), 0.0)
            denominator = (observed * weights).sum(axis=1)
            weighted = np.divide(
                (safe * weights).sum(axis=1),
                denominator,
                out=np.full(len(frame), 0.5, dtype=np.float64),
                where=denominator > 0.0,
            )
            prev1_minus_prev5 = np.where(
                observed[:, 0] & observed[:, 2], safe[:, 0] - safe[:, 2], 0.0
            )
            curvature = np.where(
                observed.all(axis=1), safe[:, 0] - 2.0 * safe[:, 1] + safe[:, 2], 0.0
            )
            observed_count = observed.sum(axis=1)
            observed_mean = np.divide(
                safe.sum(axis=1),
                observed_count,
                out=np.zeros(len(frame), dtype=np.float64),
                where=observed_count > 0,
            )
            squared = np.where(observed, np.square(safe - observed_mean[:, None]), 0.0)
            dispersion = np.sqrt(
                np.divide(
                    squared.sum(axis=1),
                    observed_count,
                    out=np.zeros(len(frame), dtype=np.float64),
                    where=observed_count > 0,
                )
            )
            cumulative, cumulative_observed = _safe_rate(_numeric(frame, cumulative_col))
            weighted_minus_cumulative = np.where(
                (denominator > 0.0) & cumulative_observed,
                weighted - cumulative,
                0.0,
            )
            prefix = f"rb_pitcher_recent_{family}"
            output[f"{prefix}_weighted"] = weighted
            output[f"{prefix}_prev1_minus_prev5"] = prev1_minus_prev5
            output[f"{prefix}_curvature"] = curvature
            output[f"{prefix}_dispersion"] = dispersion
            output[f"{prefix}_weighted_minus_cumulative"] = weighted_minus_cumulative
            recent_values[family] = {
                "weighted": weighted,
                "delta": prev1_minus_prev5,
            }

        def rate(column: str) -> np.ndarray:
            return _safe_rate(_numeric(frame, column))[0]

        pitcher_success = rate("asof_pitcher_success_rate")
        batter_success = rate("asof_batter_success_rate")
        pitcher_middle = rate("asof_pitcher_middle_rate")
        batter_middle = rate("asof_batter_middle_rate")
        pitcher_reverse = rate("asof_pitcher_reverse_rate")
        fastball, fastball_observed = _safe_rate(
            _numeric(frame, "asof_pitcher_fastball_rate")
        )
        breaking, breaking_observed = _safe_rate(
            _numeric(frame, "asof_pitcher_breaking_rate")
        )
        offspeed = np.clip(1.0 - fastball - breaking, 0.0, 1.0)
        mix = np.column_stack([fastball, breaking, offspeed])
        mix_entropy = -(mix * np.log(np.clip(mix, _RATE_EPSILON, 1.0))).sum(axis=1)
        mix_entropy = np.where(fastball_observed & breaking_observed, mix_entropy, 0.0)
        control = pitcher_success * np.clip(1.0 - pitcher_reverse - pitcher_middle, 0.0, 1.0)

        output["rb_compat_success_product"] = pitcher_success * batter_success
        output["rb_compat_success_mean"] = 0.5 * (pitcher_success + batter_success)
        output["rb_compat_middle_product"] = pitcher_middle * batter_middle
        output["rb_compat_pitcher_control_composite"] = control
        output["rb_compat_pitch_mix_entropy"] = mix_entropy

        game_f = (
            frame["game_type"].astype("string").str.strip().str.upper().eq("F")
        ).to_numpy(dtype=np.float64)
        output["rb_game_f_x_pitcher_success"] = game_f * (pitcher_success - 0.5)
        output["rb_game_f_x_batter_success"] = game_f * (batter_success - 0.5)
        output["rb_game_f_x_recent_success_delta"] = game_f * recent_values["success"]["delta"]
        output["rb_game_f_x_pitcher_control"] = game_f * control
        output["rb_game_f_x_pitch_mix_entropy"] = game_f * mix_entropy

        result = pd.DataFrame(output, index=frame.index).loc[:, self.feature_columns]
        if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
            raise RuntimeError("compact rate basis produced non-finite values")
        return result


class RateBasisAugmentedPreprocessor:
    """Canonical preprocessor plus standardized compact nonlinear rate bases."""

    categorical_columns = DeepTabularPreprocessor.categorical_columns
    numerical_columns = (
        *DeepTabularPreprocessor.numerical_columns,
        *RATE_BASIS_FEATURE_COLUMNS,
    )

    def __init__(self, *, min_category_frequency: int = 2, clip_value: float = 10.0) -> None:
        self.base = DeepTabularPreprocessor(
            min_category_frequency=min_category_frequency, clip_value=clip_value
        )
        self.builder = CompactRateBasisBuilder()
        self.extra_means_: dict[str, float] = {}
        self.extra_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "RateBasisAugmentedPreprocessor":
        if len(frame) == 0:
            raise ValueError("cannot fit an empty rate-basis preprocessor")
        self.base.fit(frame)
        extra = self.builder.transform(frame)
        self.extra_means_ = {}
        self.extra_scales_ = {}
        for column in RATE_BASIS_FEATURE_COLUMNS:
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(mean):
                raise ValueError(f"rate-basis feature {column} has a non-finite mean")
            if not np.isfinite(scale) or scale < 1.0e-8:
                scale = 1.0
            self.extra_means_[column] = mean
            self.extra_scales_[column] = scale
        self.category_cardinalities_ = list(self.base.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        result = self.base.transform(frame)
        extra = self.builder.transform(frame)
        array = np.empty((len(frame), len(RATE_BASIS_FEATURE_COLUMNS)), dtype=np.float32)
        for position, column in enumerate(RATE_BASIS_FEATURE_COLUMNS):
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            standardized = (values - self.extra_means_[column]) / self.extra_scales_[column]
            array[:, position] = np.clip(
                standardized, -self.base.clip_value, self.base.clip_value
            ).astype(np.float32)
        result["numerical"] = np.concatenate([result["numerical"], array], axis=1)
        return result

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    def feature_layout(self) -> TabularFeatureLayout:
        self._require_fitted()
        base_layout = TabularFeatureLayout.from_preprocessor(self.base)
        numerical_names = tuple(self.numerical_columns)
        pitcher_names = tuple(
            name for name in numerical_names
            if name in base_layout.pitcher_numerical_names or name.startswith("rb_pitcher_")
        )
        batter_names = tuple(
            name for name in numerical_names
            if name in base_layout.batter_numerical_names or name.startswith("rb_batter_")
        )
        return TabularFeatureLayout(
            categorical_names=base_layout.categorical_names,
            categorical_cardinalities=base_layout.categorical_cardinalities,
            numerical_names=numerical_names,
            pitcher_numerical_names=pitcher_names,
            batter_numerical_names=batter_names,
            preprocessor_fingerprint=_fingerprint(self.to_state_dict()),
        )

    def to_state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_version": RATE_BASIS_STATE_VERSION,
            "base": self.base.to_state_dict(),
            "feature_columns": list(RATE_BASIS_FEATURE_COLUMNS),
            "extra_means": dict(self.extra_means_),
            "extra_scales": dict(self.extra_scales_),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "RateBasisAugmentedPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("rate-basis state must be a mapping")
        if payload.get("state_version") != RATE_BASIS_STATE_VERSION:
            raise ValueError("unsupported rate-basis state version")
        if tuple(payload.get("feature_columns", ())) != RATE_BASIS_FEATURE_COLUMNS:
            raise ValueError("serialized rate-basis feature order differs from code")
        base = DeepTabularPreprocessor.from_state_dict(dict(payload["base"]))
        raw_means = payload.get("extra_means")
        raw_scales = payload.get("extra_scales")
        if not isinstance(raw_means, Mapping) or not isinstance(raw_scales, Mapping):
            raise ValueError("serialized rate-basis scaling state is missing")
        if set(raw_means) != set(RATE_BASIS_FEATURE_COLUMNS) or set(raw_scales) != set(RATE_BASIS_FEATURE_COLUMNS):
            raise ValueError("serialized rate-basis scaling keys differ from code")
        obj = cls(
            min_category_frequency=base.min_category_frequency,
            clip_value=base.clip_value,
        )
        obj.base = base
        obj.extra_means_ = {name: float(raw_means[name]) for name in RATE_BASIS_FEATURE_COLUMNS}
        obj.extra_scales_ = {name: float(raw_scales[name]) for name in RATE_BASIS_FEATURE_COLUMNS}
        if not all(np.isfinite(value) for value in obj.extra_means_.values()):
            raise ValueError("serialized rate-basis means must be finite")
        if not all(np.isfinite(value) and value > 0.0 for value in obj.extra_scales_.values()):
            raise ValueError("serialized rate-basis scales must be positive and finite")
        obj.category_cardinalities_ = list(base.category_cardinalities_)
        obj.is_fitted_ = True
        return obj

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("RateBasisAugmentedPreprocessor must be fitted before use")


__all__ = [
    "CUMULATIVE_RATE_SPECS",
    "CompactRateBasisBuilder",
    "RATE_BASIS_FEATURE_COLUMNS",
    "RATE_BASIS_STATE_VERSION",
    "RateBasisAugmentedPreprocessor",
]
