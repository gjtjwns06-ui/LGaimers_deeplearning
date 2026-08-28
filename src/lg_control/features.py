"""Row-local, train-fitted feature preprocessing.

No operation in this module groups, sorts, shifts, or aggregates evaluation
rows.  Every transform is a pure function of one row plus parameters learned
from the training split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    CATEGORICAL_COLUMNS,
    RATE_SMOOTHING_SPECS,
    RECENT_RATE_COLUMNS,
    TARGET_COL,
)
from .schema import validate_input_frame


STATE_VERSION = 1
MISSING_INDEX = 0
OOV_INDEX = 1


def _tokens(series: pd.Series) -> pd.Series:
    return series.astype("string")


class FeaturePreprocessor:
    """Fit category vocabularies and numeric statistics on training rows only."""

    def __init__(
        self,
        *,
        min_category_frequency: int = 1,
        smoothing_alpha: float = 50.0,
        extra_numeric_columns: Sequence[str] = (),
    ):
        if min_category_frequency < 1:
            raise ValueError("min_category_frequency must be >= 1")
        if smoothing_alpha < 0:
            raise ValueError("smoothing_alpha must be >= 0")
        self.min_category_frequency = int(min_category_frequency)
        self.smoothing_alpha = float(smoothing_alpha)
        self.extra_numeric_columns = list(dict.fromkeys(str(value) for value in extra_numeric_columns))
        if any(column in {"row_id", TARGET_COL} for column in self.extra_numeric_columns):
            raise ValueError("extra numeric features cannot include row_id or target")
        self.categorical_columns = list(CATEGORICAL_COLUMNS)
        self.numerical_columns: list[str] = []
        self.feature_names: list[str] = []
        self.category_cardinalities: list[int] = []
        self.category_maps: dict[str, dict[str, int]] = {}
        self.rate_priors: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.target_prior = 0.5
        self._is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        validate_input_frame(
            frame,
            require_target=TARGET_COL in frame.columns,
            allow_target=True,
            allowed_extra_columns=self.extra_numeric_columns,
        )
        if len(frame) == 0:
            raise ValueError("cannot fit FeaturePreprocessor on an empty frame")

        if TARGET_COL in frame.columns:
            self.target_prior = float(frame[TARGET_COL].mean())

        for rate_column, _ in RATE_SMOOTHING_SPECS:
            if rate_column in {"asof_pitcher_success_rate", "asof_batter_success_rate"}:
                prior = self.target_prior
            else:
                prior = float(pd.to_numeric(frame[rate_column], errors="coerce").mean())
            self.rate_priors[rate_column] = prior if math.isfinite(prior) else 0.5
        for column in RECENT_RATE_COLUMNS:
            prior = float(pd.to_numeric(frame[column], errors="coerce").mean())
            self.rate_priors[column] = prior if math.isfinite(prior) else self.target_prior

        categorical = self._categorical_frame(frame)
        self.category_maps = {}
        self.category_cardinalities = []
        for column in self.categorical_columns:
            counts = _tokens(categorical[column]).value_counts(dropna=True)
            kept = sorted(str(value) for value, count in counts.items() if count >= self.min_category_frequency)
            mapping = {token: index + 2 for index, token in enumerate(kept)}
            self.category_maps[column] = mapping
            self.category_cardinalities.append(len(mapping) + 2)

        numerical = self._numerical_frame(frame)
        self.numerical_columns = list(numerical.columns)
        self.numeric_means = {}
        self.numeric_stds = {}
        for column in self.numerical_columns:
            values = pd.to_numeric(numerical[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            mean = float(values[finite].mean()) if finite.any() else 0.0
            std = float(values[finite].std()) if finite.any() else 1.0
            if not math.isfinite(std) or std < 1e-8:
                std = 1.0
            self.numeric_means[column] = mean
            self.numeric_stds[column] = std

        self.feature_names = [*self.categorical_columns, *self.numerical_columns]
        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        validate_input_frame(
            frame,
            require_target=False,
            allow_target=True,
            allowed_extra_columns=self.extra_numeric_columns,
        )
        categorical_frame = self._categorical_frame(frame)
        categorical = np.empty((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for index, column in enumerate(self.categorical_columns):
            token_series = _tokens(categorical_frame[column])
            mapped = token_series.map(self.category_maps[column])
            values = mapped.fillna(OOV_INDEX).to_numpy(dtype=np.int64)
            values[token_series.isna().to_numpy()] = MISSING_INDEX
            categorical[:, index] = values

        numerical_frame = self._numerical_frame(frame)
        if list(numerical_frame.columns) != self.numerical_columns:
            raise RuntimeError("numeric feature order changed after fitting")
        numerical = np.empty((len(frame), len(self.numerical_columns)), dtype=np.float32)
        for index, column in enumerate(self.numerical_columns):
            values = pd.to_numeric(numerical_frame[column], errors="coerce").to_numpy(dtype=np.float64)
            mean = self.numeric_means[column]
            std = self.numeric_stds[column]
            values = np.nan_to_num(values, nan=mean, posinf=mean, neginf=mean)
            numerical[:, index] = np.clip((values - mean) / std, -10.0, 10.0).astype(np.float32)

        return {"categorical": categorical, "numerical": numerical}

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_version": STATE_VERSION,
            "min_category_frequency": self.min_category_frequency,
            "smoothing_alpha": self.smoothing_alpha,
            "extra_numeric_columns": self.extra_numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "category_cardinalities": self.category_cardinalities,
            "category_maps": self.category_maps,
            "rate_priors": self.rate_priors,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "target_prior": self.target_prior,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("state_version") != STATE_VERSION:
            raise ValueError(f"unsupported preprocessor state version: {payload.get('state_version')}")
        obj = cls(
            min_category_frequency=int(payload["min_category_frequency"]),
            smoothing_alpha=float(payload["smoothing_alpha"]),
            extra_numeric_columns=payload.get("extra_numeric_columns", []),
        )
        obj.categorical_columns = list(payload["categorical_columns"])
        obj.numerical_columns = list(payload["numerical_columns"])
        obj.feature_names = [*obj.categorical_columns, *obj.numerical_columns]
        obj.category_cardinalities = [int(value) for value in payload["category_cardinalities"]]
        obj.category_maps = {
            str(column): {str(token): int(index) for token, index in mapping.items()}
            for column, mapping in payload["category_maps"].items()
        }
        obj.rate_priors = {str(key): float(value) for key, value in payload["rate_priors"].items()}
        obj.numeric_means = {str(key): float(value) for key, value in payload["numeric_means"].items()}
        obj.numeric_stds = {str(key): float(value) for key, value in payload["numeric_stds"].items()}
        obj.target_prior = float(payload["target_prior"])
        obj._is_fitted = True
        return obj

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before use")

    @staticmethod
    def _categorical_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[
            [
                "top_bottom",
                "game_type",
                "base_state",
                "pitcher_id",
                "batter_id",
                "pitcher_hand",
                "batter_hand",
                "pitcher_team_id",
                "batter_team_id",
                "game_month",
                "game_dayofweek",
            ]
        ].copy()
        balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
        strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
        outs = pd.to_numeric(frame["outs_before"], errors="coerce").fillna(-1).astype(int)
        inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(-1).astype(int)
        result["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
        result["base_out_state"] = frame["base_state"].astype("string") + "-" + outs.astype(str)
        result["inning_bucket"] = inning.clip(lower=-1, upper=10).astype(str)
        result["platoon"] = (
            frame["pitcher_hand"].astype("string") + "-" + frame["batter_hand"].astype("string")
        )
        return result[CATEGORICAL_COLUMNS]

    def _numerical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        def num(column: str) -> pd.Series:
            return pd.to_numeric(frame[column], errors="coerce").astype(float)

        season = num("season")
        month = num("game_month")
        day = num("game_dayofweek")
        inning = num("inning")
        balls = num("balls_before")
        strikes = num("strikes_before")
        outs = num("outs_before")
        run_top = num("run_top_before")
        run_bot = num("run_bot_before")
        score_diff = num("score_diff_pitcher_team")
        li = num("li").clip(lower=0)
        n_pitcher = num("asof_pitcher_n").clip(lower=0)
        n_batter = num("asof_batter_n").clip(lower=0)
        game_type_f = frame["game_type"].astype("string").eq("F").astype(float)

        data: dict[str, pd.Series | np.ndarray] = {
            "season_offset": season - 2019.0,
            "game_type_f": game_type_f,
            "season_x_game_type_f": (season - 2019.0) * game_type_f,
            "month_sin": np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            "month_cos": np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            "dayofweek_sin": np.sin(2.0 * np.pi * day / 7.0),
            "dayofweek_cos": np.cos(2.0 * np.pi * day / 7.0),
            "inning": inning,
            "late_inning": (inning >= 7).astype(float),
            "balls_before": balls,
            "strikes_before": strikes,
            "outs_before": outs,
            "two_strikes": (strikes == 2).astype(float),
            "three_balls": (balls == 3).astype(float),
            "full_count": ((balls == 3) & (strikes == 2)).astype(float),
            "run_top_log1p": np.log1p(run_top.clip(lower=0)),
            "run_bot_log1p": np.log1p(run_bot.clip(lower=0)),
            "run_total_log1p": np.log1p((run_top + run_bot).clip(lower=0)),
            "score_diff_pitcher_team": score_diff,
            "score_diff_abs": score_diff.abs(),
            "pitcher_team_leading": (score_diff > 0).astype(float),
            "close_game": (score_diff.abs() <= 1).astype(float),
            "runner_scoring_position": (
                (num("runner_on_2b") > 0) | (num("runner_on_3b") > 0)
            ).astype(float),
            "bases_loaded": (
                (num("runner_on_1b") > 0)
                & (num("runner_on_2b") > 0)
                & (num("runner_on_3b") > 0)
            ).astype(float),
            "home_win_expectancy": num("home_win_expectancy"),
            "li_log1p": np.log1p(li),
            "pitcher_n_log1p": np.log1p(n_pitcher),
            "batter_n_log1p": np.log1p(n_batter),
            "pitcher_history_reliability": n_pitcher / (n_pitcher + self.smoothing_alpha),
            "batter_history_reliability": n_batter / (n_batter + self.smoothing_alpha),
        }

        for rate_column, count_column in RATE_SMOOTHING_SPECS:
            raw = num(rate_column)
            count = num(count_column).fillna(0.0).clip(lower=0.0)
            prior = float(self.rate_priors.get(rate_column, 0.5))
            filled = raw.fillna(prior).clip(lower=0.0, upper=1.0)
            denominator = count + self.smoothing_alpha
            if self.smoothing_alpha == 0:
                smoothed = filled
            else:
                smoothed = (count * filled + self.smoothing_alpha * prior) / denominator
            data[rate_column] = filled
            data[f"{rate_column}_smoothed"] = smoothed
            data[f"{rate_column}_missing"] = raw.isna().astype(float)

        for rate_column in RECENT_RATE_COLUMNS:
            raw = num(rate_column)
            prior = float(self.rate_priors.get(rate_column, self.target_prior))
            data[rate_column] = raw.fillna(prior).clip(lower=0.0, upper=1.0)
            data[f"{rate_column}_missing"] = raw.isna().astype(float)

        for column in self.extra_numeric_columns:
            raw = num(column)
            if column.endswith("_count"):
                data[f"{column}_log1p"] = np.log1p(raw.clip(lower=0.0))
            else:
                data[column] = raw
            data[f"{column}_missing"] = raw.isna().astype(float)

        return pd.DataFrame(data, index=frame.index)
