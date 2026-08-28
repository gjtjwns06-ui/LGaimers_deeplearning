"""Transparent, row-local preprocessing for neural tabular inputs.

This module deliberately stops at preprocessing.  It contains no split,
validation, model, optimisation, inference, or submission logic.  ``fit``
learns only category vocabularies and numeric scaling statistics from the
reference rows explicitly supplied by the caller.  ``transform`` never uses
other rows in the frame to create a feature, so transforming one evaluation
row is identical to transforming that row inside a batch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ID_COL, RAW_FEATURE_COLUMNS, TARGET_COL
from .schema import SchemaError, validate_input_frame


MISSING_CATEGORY_INDEX = 0
OOV_CATEGORY_INDEX = 1
DEEP_PREPROCESSOR_STATE_VERSION = 1


def _serialized_int(value: Any, *, name: str, minimum: int = 0) -> int:
    """Decode an exact JSON-style integer without lossy coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"serialized {name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"serialized {name} must be >= {minimum}")
    return result

# Each entry is exactly recoverable from retained raw context.  Keeping this
# list public makes the information-removal policy auditable.
DROPPED_DETERMINISTIC_COLUMNS = (
    "run_total_before",  # run_top_before + run_bot_before
    "score_diff_home",  # run_bot_before - run_top_before
    "runner_on_1b",  # encoded by base_state
    "runner_on_2b",  # encoded by base_state
    "runner_on_3b",  # encoded by base_state
    "num_runners_on",  # number of occupied bases
    "away_win_expectancy",  # rounded complement of home_win_expectancy
    "asof_pitcher_pitchmix_n",  # equal to asof_pitcher_n
    "asof_pitcher_offspeed_rate",  # 1 - fastball_rate - breaking_rate
)

RAW_RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
)

CATEGORICAL_FEATURE_COLUMNS = (
    "season",
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "base_out_state",
    "inning_bucket",
    "platoon",
    "score_state",
)

NUMERICAL_FEATURE_COLUMNS = (
    "season_numeric",
    "month_sin",
    "month_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "two_strikes",
    "three_balls",
    "full_count",
    "late_inning",
    "extra_innings",
    "run_top_log1p",
    "run_bot_log1p",
    "score_diff_pitcher_team",
    "score_diff_abs",
    "pitcher_team_leading",
    "pitcher_team_trailing",
    "close_game",
    "runner_count",
    "runner_scoring_position",
    "bases_loaded",
    "home_win_expectancy",
    "li_log1p",
    "pitcher_n_log1p",
    "batter_n_log1p",
    "pitcher_cold_start",
    "batter_cold_start",
    *RAW_RATE_COLUMNS,
    *(f"{column}__missing" for column in RAW_RATE_COLUMNS),
)

_STRUCTURAL_INTEGER_RANGES: dict[str, tuple[int | None, int | None]] = {
    "season": (1900, None),
    "game_month": (1, 12),
    "game_dayofweek": (0, 6),
    "inning": (1, None),
    "balls_before": (0, 3),
    "strikes_before": (0, 2),
    "outs_before": (0, 2),
    "run_top_before": (0, None),
    "run_bot_before": (0, None),
    "run_total_before": (0, None),
    "score_diff_home": (None, None),
    "score_diff_pitcher_team": (None, None),
    "runner_on_1b": (0, 1),
    "runner_on_2b": (0, 1),
    "runner_on_3b": (0, 1),
    "num_runners_on": (0, 3),
    "asof_pitcher_n": (0, None),
    "asof_batter_n": (0, None),
    "asof_pitcher_pitchmix_n": (0, None),
}

_BASE_STATE_FROM_FLAGS = {
    (0, 0, 0): "___",
    (1, 0, 0): "1__",
    (0, 1, 0): "_2_",
    (0, 0, 1): "__3",
    (1, 1, 0): "12_",
    (1, 0, 1): "1_3",
    (0, 1, 1): "_23",
    (1, 1, 1): "123",
}


def _canonical_tokens(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    """Return whitespace-normalised strings while retaining true missingness."""

    tokens = series.astype("string").str.strip()
    tokens = tokens.mask(tokens.eq(""), pd.NA)
    # CSV inference can turn an otherwise integral ID into ``123.0`` when a
    # missing value is present.  Canonicalising it prevents false OOV tokens.
    tokens = tokens.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)
    if uppercase:
        tokens = tokens.str.upper()
    return tokens


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column]
    values = pd.to_numeric(source, errors="coerce")
    invalid = source.notna() & values.isna()
    if invalid.any():
        examples = source.loc[invalid].head(3).tolist()
        raise SchemaError(f"{column} contains non-numeric values such as {examples}")
    return values.astype(float)


def _raise_on_mask(mask: pd.Series | np.ndarray, message: str) -> None:
    values = np.asarray(mask, dtype=bool)
    if values.any():
        examples = np.flatnonzero(values)[:5].tolist()
        raise SchemaError(f"{message}; row positions include {examples}")


def validate_preprocessing_schema(frame: pd.DataFrame) -> None:
    """Validate official schema, domains, and deterministic relationships.

    The relationship checks justify the columns removed by
    :data:`DROPPED_DETERMINISTIC_COLUMNS`.  No target-derived statistic is
    computed here.
    """

    validate_input_frame(frame, require_target=False, allow_target=True)

    numeric: dict[str, pd.Series] = {}
    for column, (lower, upper) in _STRUCTURAL_INTEGER_RANGES.items():
        values = _numeric(frame, column)
        _raise_on_mask(values.isna(), f"{column} contains missing values")
        _raise_on_mask(~np.isfinite(values), f"{column} contains non-finite values")
        _raise_on_mask(~np.isclose(values, np.round(values)), f"{column} must be integral")
        if lower is not None:
            _raise_on_mask(values < lower, f"{column} must be >= {lower}")
        if upper is not None:
            _raise_on_mask(values > upper, f"{column} must be <= {upper}")
        numeric[column] = values

    top_bottom = _canonical_tokens(frame["top_bottom"], uppercase=True)
    _raise_on_mask(~top_bottom.isin(["T", "B"]), "top_bottom must contain only T or B")
    for column in ("game_type", "base_state"):
        tokens = _canonical_tokens(frame[column], uppercase=column == "game_type")
        _raise_on_mask(tokens.isna(), f"{column} contains missing/blank values")

    for column in ("home_win_expectancy", "away_win_expectancy"):
        values = _numeric(frame, column)
        _raise_on_mask(values.isna() | ~np.isfinite(values), f"{column} must be finite")
        _raise_on_mask((values < 0.0) | (values > 100.0), f"{column} must be in [0, 100]")

    leverage = _numeric(frame, "li")
    _raise_on_mask(leverage.isna() | ~np.isfinite(leverage), "li must be finite")
    _raise_on_mask(leverage < 0.0, "li must be non-negative")

    for column in RAW_RATE_COLUMNS + ("asof_pitcher_offspeed_rate",):
        values = _numeric(frame, column)
        finite = values.notna()
        _raise_on_mask(
            finite & ((values < 0.0) | (values > 1.0)),
            f"{column} must be in [0, 1] when present",
        )

    _raise_on_mask(
        numeric["run_total_before"]
        != numeric["run_top_before"] + numeric["run_bot_before"],
        "run_total_before is inconsistent with team runs",
    )
    expected_home_diff = numeric["run_bot_before"] - numeric["run_top_before"]
    _raise_on_mask(
        numeric["score_diff_home"] != expected_home_diff,
        "score_diff_home is inconsistent with team runs",
    )
    expected_pitcher_diff = pd.Series(
        np.where(top_bottom.eq("T"), expected_home_diff, -expected_home_diff),
        index=frame.index,
    )
    _raise_on_mask(
        numeric["score_diff_pitcher_team"] != expected_pitcher_diff,
        "score_diff_pitcher_team is inconsistent with inning half",
    )

    runner_count = (
        numeric["runner_on_1b"]
        + numeric["runner_on_2b"]
        + numeric["runner_on_3b"]
    )
    _raise_on_mask(
        numeric["num_runners_on"] != runner_count,
        "num_runners_on is inconsistent with runner flags",
    )
    expected_base = pd.Series(
        [
            _BASE_STATE_FROM_FLAGS[(int(first), int(second), int(third))]
            for first, second, third in zip(
                numeric["runner_on_1b"],
                numeric["runner_on_2b"],
                numeric["runner_on_3b"],
                strict=True,
            )
        ],
        index=frame.index,
        dtype="string",
    )
    actual_base = _canonical_tokens(frame["base_state"])
    _raise_on_mask(actual_base.ne(expected_base), "base_state is inconsistent with runner flags")

    home = _numeric(frame, "home_win_expectancy")
    away = _numeric(frame, "away_win_expectancy")
    _raise_on_mask(
        ~np.isclose(home + away, 100.0, atol=0.100001, rtol=0.0),
        "home and away win expectancy must sum to 100 within source rounding",
    )
    _raise_on_mask(
        numeric["asof_pitcher_pitchmix_n"] != numeric["asof_pitcher_n"],
        "pitchmix count is inconsistent with pitcher history count",
    )

    fastball = _numeric(frame, "asof_pitcher_fastball_rate")
    breaking = _numeric(frame, "asof_pitcher_breaking_rate")
    offspeed = _numeric(frame, "asof_pitcher_offspeed_rate")
    present = fastball.notna() & breaking.notna() & offspeed.notna()
    _raise_on_mask(
        present
        & ~np.isclose(fastball + breaking + offspeed, 1.0, atol=1.1e-6, rtol=0.0),
        "pitch-mix rates must sum to 1 when present",
    )


def _inning_bucket(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    buckets = np.select(
        [array <= 3, array <= 6, array <= 9],
        ["early", "middle", "late"],
        default="extra",
    )
    return pd.Series(buckets, index=values.index, dtype="string")


class DeepTabularPreprocessor:
    """Fit reusable numeric statistics and categorical vocabularies.

    The caller controls which rows are used by :meth:`fit`; state is never
    updated by :meth:`transform`.  Missing categorical values and unseen
    categories use separate reserved indices (0 and 1 respectively).
    """

    categorical_columns = CATEGORICAL_FEATURE_COLUMNS
    numerical_columns = NUMERICAL_FEATURE_COLUMNS
    dropped_columns = DROPPED_DETERMINISTIC_COLUMNS

    def __init__(self, *, min_category_frequency: int = 1, clip_value: float = 10.0):
        if min_category_frequency < 1:
            raise ValueError("min_category_frequency must be >= 1")
        if not np.isfinite(clip_value) or clip_value <= 0:
            raise ValueError("clip_value must be a positive finite number")
        self.min_category_frequency = int(min_category_frequency)
        self.clip_value = float(clip_value)
        self.category_vocabs_: dict[str, dict[str, int]] = {}
        self.category_cardinalities_: list[int] = []
        self.numeric_imputation_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_scales_: dict[str, float] = {}
        self.is_fitted_ = False

    def clean_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return an auditable, unscaled feature table with categories intact."""

        validate_preprocessing_schema(frame)
        result = pd.DataFrame(index=frame.index)
        result[ID_COL] = frame[ID_COL].copy()

        season = _numeric(frame, "season")
        month = _numeric(frame, "game_month")
        weekday = _numeric(frame, "game_dayofweek")
        inning = _numeric(frame, "inning")
        balls = _numeric(frame, "balls_before")
        strikes = _numeric(frame, "strikes_before")
        outs = _numeric(frame, "outs_before")
        run_top = _numeric(frame, "run_top_before")
        run_bot = _numeric(frame, "run_bot_before")
        pitcher_diff = _numeric(frame, "score_diff_pitcher_team")
        first = _numeric(frame, "runner_on_1b")
        second = _numeric(frame, "runner_on_2b")
        third = _numeric(frame, "runner_on_3b")
        pitcher_n = _numeric(frame, "asof_pitcher_n")
        batter_n = _numeric(frame, "asof_batter_n")

        base_categories = {
            "season": _canonical_tokens(frame["season"]),
            "game_month": _canonical_tokens(frame["game_month"]),
            "game_dayofweek": _canonical_tokens(frame["game_dayofweek"]),
            "top_bottom": _canonical_tokens(frame["top_bottom"], uppercase=True),
            "game_type": _canonical_tokens(frame["game_type"], uppercase=True),
            "base_state": _canonical_tokens(frame["base_state"]),
            "pitcher_id": _canonical_tokens(frame["pitcher_id"]),
            "batter_id": _canonical_tokens(frame["batter_id"]),
            "pitcher_hand": _canonical_tokens(frame["pitcher_hand"]),
            "batter_hand": _canonical_tokens(frame["batter_hand"]),
            "pitcher_team_id": _canonical_tokens(frame["pitcher_team_id"]),
            "batter_team_id": _canonical_tokens(frame["batter_team_id"]),
            "count_state": (
                "B" + balls.astype(int).astype(str) + "_S" + strikes.astype(int).astype(str)
            ).astype("string"),
            "base_out_state": (
                _canonical_tokens(frame["base_state"]) + "_O" + outs.astype(int).astype(str)
            ).astype("string"),
            "inning_bucket": _inning_bucket(inning),
            "platoon": (
                _canonical_tokens(frame["pitcher_hand"])
                + "_vs_"
                + _canonical_tokens(frame["batter_hand"])
            ).astype("string"),
            "score_state": pd.Series(
                np.select(
                    [pitcher_diff.to_numpy() > 0, pitcher_diff.to_numpy() < 0],
                    ["leading", "trailing"],
                    default="tied",
                ),
                index=frame.index,
                dtype="string",
            ),
        }
        for column in self.categorical_columns:
            result[column] = base_categories[column]

        numerical: dict[str, pd.Series | np.ndarray] = {
            "season_numeric": season,
            "month_sin": np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            "month_cos": np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            "dayofweek_sin": np.sin(2.0 * np.pi * weekday / 7.0),
            "dayofweek_cos": np.cos(2.0 * np.pi * weekday / 7.0),
            "inning": inning,
            "balls_before": balls,
            "strikes_before": strikes,
            "outs_before": outs,
            "two_strikes": strikes.eq(2).astype(float),
            "three_balls": balls.eq(3).astype(float),
            "full_count": (balls.eq(3) & strikes.eq(2)).astype(float),
            "late_inning": inning.ge(7).astype(float),
            "extra_innings": inning.ge(10).astype(float),
            "run_top_log1p": np.log1p(run_top),
            "run_bot_log1p": np.log1p(run_bot),
            "score_diff_pitcher_team": pitcher_diff,
            "score_diff_abs": pitcher_diff.abs(),
            "pitcher_team_leading": pitcher_diff.gt(0).astype(float),
            "pitcher_team_trailing": pitcher_diff.lt(0).astype(float),
            "close_game": pitcher_diff.abs().le(1).astype(float),
            "runner_count": first + second + third,
            "runner_scoring_position": (second.gt(0) | third.gt(0)).astype(float),
            "bases_loaded": (first.gt(0) & second.gt(0) & third.gt(0)).astype(float),
            "home_win_expectancy": _numeric(frame, "home_win_expectancy"),
            "li_log1p": np.log1p(_numeric(frame, "li")),
            "pitcher_n_log1p": np.log1p(pitcher_n),
            "batter_n_log1p": np.log1p(batter_n),
            "pitcher_cold_start": pitcher_n.eq(0).astype(float),
            "batter_cold_start": batter_n.eq(0).astype(float),
        }
        for column in RAW_RATE_COLUMNS:
            values = _numeric(frame, column)
            numerical[column] = values
        for column in RAW_RATE_COLUMNS:
            numerical[f"{column}__missing"] = _numeric(frame, column).isna().astype(float)

        for column in self.numerical_columns:
            result[column] = pd.to_numeric(numerical[column], errors="coerce").astype(float)

        if TARGET_COL in frame.columns:
            result[TARGET_COL] = frame[TARGET_COL].copy()
        return result[[ID_COL, *self.categorical_columns, *self.numerical_columns, *(
            [TARGET_COL] if TARGET_COL in frame.columns else []
        )]]

    def fit(self, frame: pd.DataFrame) -> "DeepTabularPreprocessor":
        """Learn state from only ``frame``; the target column is ignored."""

        if len(frame) == 0:
            raise ValueError("cannot fit on an empty frame")
        clean = self.clean_frame(frame)

        self.category_vocabs_ = {}
        self.category_cardinalities_ = []
        for column in self.categorical_columns:
            counts = clean[column].value_counts(dropna=True)
            kept = sorted(
                str(token)
                for token, count in counts.items()
                if int(count) >= self.min_category_frequency
            )
            mapping = {token: index + 2 for index, token in enumerate(kept)}
            self.category_vocabs_[column] = mapping
            self.category_cardinalities_.append(len(mapping) + 2)

        self.numeric_imputation_ = {}
        self.numeric_means_ = {}
        self.numeric_scales_ = {}
        for column in self.numerical_columns:
            values = clean[column].to_numpy(dtype=np.float64, copy=True)
            finite = np.isfinite(values)
            if finite.any():
                imputation = float(np.median(values[finite]))
            else:
                imputation = 0.0
            filled = np.where(finite, values, imputation)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            self.numeric_imputation_[column] = imputation
            self.numeric_means_[column] = mean
            self.numeric_scales_[column] = scale

        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Encode a frame without fitting or aggregating transform rows."""

        self._require_fitted()
        clean = self.clean_frame(frame)

        categorical = np.empty((len(clean), len(self.categorical_columns)), dtype=np.int64)
        for index, column in enumerate(self.categorical_columns):
            tokens = clean[column]
            mapped = tokens.map(self.category_vocabs_[column])
            encoded = mapped.fillna(OOV_CATEGORY_INDEX).to_numpy(dtype=np.int64)
            encoded[tokens.isna().to_numpy()] = MISSING_CATEGORY_INDEX
            categorical[:, index] = encoded

        numerical = np.empty((len(clean), len(self.numerical_columns)), dtype=np.float32)
        for index, column in enumerate(self.numerical_columns):
            values = clean[column].to_numpy(dtype=np.float64, copy=True)
            finite = np.isfinite(values)
            values[~finite] = self.numeric_imputation_[column]
            values = (values - self.numeric_means_[column]) / self.numeric_scales_[column]
            numerical[:, index] = np.clip(
                values, -self.clip_value, self.clip_value
            ).astype(np.float32)

        result: dict[str, np.ndarray] = {
            "row_id": clean[ID_COL].to_numpy(copy=True),
            "categorical": categorical,
            "numerical": numerical,
        }
        if TARGET_COL in clean.columns:
            result["target"] = clean[TARGET_COL].to_numpy(dtype=np.float32, copy=True)
        return result

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    def to_state_dict(self) -> dict[str, Any]:
        """Return the complete target-independent fitted state."""

        self._require_fitted()
        return {
            "state_version": DEEP_PREPROCESSOR_STATE_VERSION,
            "preprocessor_class": type(self).__name__,
            "min_category_frequency": self.min_category_frequency,
            "clip_value": self.clip_value,
            "categorical_columns": list(self.categorical_columns),
            "numerical_columns": list(self.numerical_columns),
            "dropped_columns": list(self.dropped_columns),
            "category_vocabs": {
                column: dict(mapping)
                for column, mapping in self.category_vocabs_.items()
            },
            "category_cardinalities": list(self.category_cardinalities_),
            "numeric_imputation": dict(self.numeric_imputation_),
            "numeric_means": dict(self.numeric_means_),
            "numeric_scales": dict(self.numeric_scales_),
        }

    def save(self, path: str | Path) -> None:
        """Persist the fitted state as deterministic, inspectable JSON."""

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
    def from_state_dict(cls, payload: dict[str, Any]) -> "DeepTabularPreprocessor":
        """Restore and strictly validate a serialized fitted state."""

        if not isinstance(payload, dict):
            raise TypeError("preprocessor state must be a dictionary")
        raw_version = payload.get("state_version")
        if (
            isinstance(raw_version, (bool, np.bool_))
            or not isinstance(raw_version, (int, np.integer))
            or int(raw_version) != DEEP_PREPROCESSOR_STATE_VERSION
        ):
            raise ValueError(
                "unsupported DeepTabularPreprocessor state version: "
                f"{raw_version}"
            )
        if payload.get("preprocessor_class") != cls.__name__:
            raise ValueError("serialized preprocessor class does not match")
        if tuple(payload.get("categorical_columns", ())) != cls.categorical_columns:
            raise ValueError("serialized categorical feature order does not match code")
        if tuple(payload.get("numerical_columns", ())) != cls.numerical_columns:
            raise ValueError("serialized numerical feature order does not match code")
        if tuple(payload.get("dropped_columns", ())) != cls.dropped_columns:
            raise ValueError("serialized deterministic-drop policy does not match code")

        obj = cls(
            min_category_frequency=_serialized_int(
                payload["min_category_frequency"],
                name="min_category_frequency",
                minimum=1,
            ),
            clip_value=float(payload["clip_value"]),
        )
        raw_vocabs = payload.get("category_vocabs")
        raw_cardinalities = payload.get("category_cardinalities")
        if not isinstance(raw_vocabs, dict) or not isinstance(
            raw_cardinalities, list
        ):
            raise ValueError("serialized categorical state is malformed")
        if set(raw_vocabs) != set(cls.categorical_columns):
            raise ValueError("serialized category vocab columns do not match")
        if len(raw_cardinalities) != len(cls.categorical_columns):
            raise ValueError("serialized category cardinality count does not match")

        obj.category_vocabs_ = {}
        obj.category_cardinalities_ = []
        for column, raw_cardinality in zip(
            cls.categorical_columns, raw_cardinalities, strict=True
        ):
            raw_mapping = raw_vocabs[column]
            if not isinstance(raw_mapping, dict):
                raise ValueError(f"serialized vocabulary for {column} is malformed")
            if not all(isinstance(token, str) for token in raw_mapping):
                raise ValueError(f"serialized vocabulary keys for {column} must be strings")
            mapping = {
                token: _serialized_int(
                    index,
                    name=f"vocabulary index for {column}",
                    minimum=2,
                )
                for token, index in raw_mapping.items()
            }
            expected_indices = list(range(2, len(mapping) + 2))
            if sorted(mapping.values()) != expected_indices:
                raise ValueError(
                    f"serialized vocabulary indices for {column} are not contiguous from 2"
                )
            cardinality = _serialized_int(
                raw_cardinality,
                name=f"cardinality for {column}",
                minimum=2,
            )
            if cardinality != len(mapping) + 2:
                raise ValueError(
                    f"serialized cardinality for {column} does not match vocabulary"
                )
            obj.category_vocabs_[column] = mapping
            obj.category_cardinalities_.append(cardinality)

        numeric_payloads = {
            "numeric_imputation": payload.get("numeric_imputation"),
            "numeric_means": payload.get("numeric_means"),
            "numeric_scales": payload.get("numeric_scales"),
        }
        restored_numeric: dict[str, dict[str, float]] = {}
        for state_name, raw_values in numeric_payloads.items():
            if not isinstance(raw_values, dict) or set(raw_values) != set(
                cls.numerical_columns
            ):
                raise ValueError(f"serialized {state_name} columns do not match")
            if any(isinstance(value, (bool, np.bool_)) for value in raw_values.values()):
                raise ValueError(f"serialized {state_name} contains boolean values")
            values = {str(key): float(value) for key, value in raw_values.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"serialized {state_name} contains non-finite values")
            restored_numeric[state_name] = values
        if any(value <= 0.0 for value in restored_numeric["numeric_scales"].values()):
            raise ValueError("serialized numeric scales must be positive")

        obj.numeric_imputation_ = restored_numeric["numeric_imputation"]
        obj.numeric_means_ = restored_numeric["numeric_means"]
        obj.numeric_scales_ = restored_numeric["numeric_scales"]
        obj.is_fitted_ = True
        return obj

    @classmethod
    def load(cls, path: str | Path) -> "DeepTabularPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(payload)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.categorical_columns, *self.numerical_columns)

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("DeepTabularPreprocessor must be fitted before transform")


def raw_columns_not_used_as_features() -> tuple[str, ...]:
    """Return official raw columns intentionally excluded from model inputs."""

    retained = set(CATEGORICAL_FEATURE_COLUMNS) | {
        "inning",
        "balls_before",
        "strikes_before",
        "outs_before",
        "score_diff_pitcher_team",
        "home_win_expectancy",
        *RAW_RATE_COLUMNS,
    }
    excluded: Iterable[str] = (
        column for column in RAW_FEATURE_COLUMNS if column not in retained
    )
    return tuple(excluded)
