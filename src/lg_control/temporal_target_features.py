"""Strictly prior-season entity target encodings for neural row inputs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import TARGET_COL
from .model import TabularFeatureLayout
from .preprocessing import DeepTabularPreprocessor, validate_preprocessing_schema


TEMPORAL_TARGET_STATE_VERSION = 1
TEMPORAL_WINDOWS = (1, 2)


@dataclass(frozen=True)
class TargetEncodingSpec:
    name: str
    columns: tuple[str, ...]
    strength: float


TARGET_ENCODING_SPECS = (
    TargetEncodingSpec("pitcher", ("pitcher_id",), 100.0),
    TargetEncodingSpec("pitcher_batter_hand", ("pitcher_id", "batter_hand"), 100.0),
    TargetEncodingSpec("pitcher_game_type", ("pitcher_id", "game_type"), 100.0),
    TargetEncodingSpec(
        "pitcher_count",
        ("pitcher_id", "balls_before", "strikes_before"),
        150.0,
    ),
    TargetEncodingSpec("batter", ("batter_id",), 200.0),
    TargetEncodingSpec("batter_pitcher_hand", ("batter_id", "pitcher_hand"), 200.0),
    TargetEncodingSpec(
        "hand_count",
        ("pitcher_hand", "batter_hand", "balls_before", "strikes_before"),
        500.0,
    ),
)

TEMPORAL_TARGET_FEATURE_COLUMNS = tuple(
    feature
    for spec in TARGET_ENCODING_SPECS
    for feature in (
        *(f"te_{spec.name}_prev{window}_rate" for window in TEMPORAL_WINDOWS),
        *(f"te_{spec.name}_prev{window}_reliability" for window in TEMPORAL_WINDOWS),
        *(f"te_{spec.name}_prev{window}_log1p_count" for window in TEMPORAL_WINDOWS),
        f"te_{spec.name}_prev1_minus_prev2",
    )
)


def _tokens(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)


def _canonical_key_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({column: _tokens(frame[column]) for column in columns}, index=frame.index)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class PriorSeasonTargetEncoder:
    """Store season-level sufficient statistics and expose only years < row year."""

    feature_columns = TEMPORAL_TARGET_FEATURE_COLUMNS

    def __init__(self) -> None:
        self.group_statistics_: dict[
            tuple[str, int], dict[tuple[str, ...], tuple[int, int]]
        ] = {}
        self.game_type_statistics_: dict[tuple[int, str], tuple[int, int]] = {}
        self.global_statistics_: dict[int, tuple[int, int]] = {}
        self.fit_seasons_: tuple[int, ...] = ()
        self.is_fitted_ = False
        self._combined_cache: dict[
            tuple[str, int, int], dict[tuple[str, ...], tuple[int, int]]
        ] = {}

    def fit(self, frame: pd.DataFrame) -> "PriorSeasonTargetEncoder":
        if TARGET_COL not in frame.columns or len(frame) == 0:
            raise ValueError("target encoder fit requires a non-empty labelled frame")
        validate_preprocessing_schema(frame)
        season = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
        target = pd.to_numeric(frame[TARGET_COL], errors="raise").to_numpy(dtype=np.float64)
        if not np.isin(target, [0.0, 1.0]).all():
            raise ValueError("target encoder requires binary labels")
        working = frame.copy(deep=False)
        self.group_statistics_ = {}
        for spec in TARGET_ENCODING_SPECS:
            keys = _canonical_key_frame(working, spec.columns)
            table = keys.assign(season=season, target=target.astype(np.int64))
            grouped = table.groupby(
                ["season", *spec.columns], sort=True, dropna=False, observed=True
            )["target"].agg(["size", "sum"])
            by_season: dict[int, dict[tuple[str, ...], tuple[int, int]]] = defaultdict(dict)
            for index, row in grouped.iterrows():
                year = int(index[0])
                raw_key = index[1:]
                key = tuple(str(value) for value in raw_key)
                by_season[year][key] = (int(row["size"]), int(row["sum"]))
            for year, mapping in by_season.items():
                self.group_statistics_[(spec.name, year)] = mapping

        summary = pd.DataFrame(
            {
                "season": season,
                "game_type": _tokens(frame["game_type"]).str.upper(),
                "target": target.astype(np.int64),
            }
        )
        global_grouped = summary.groupby("season")["target"].agg(["size", "sum"])
        self.global_statistics_ = {
            int(year): (int(row["size"]), int(row["sum"]))
            for year, row in global_grouped.iterrows()
        }
        type_grouped = summary.groupby(["season", "game_type"])["target"].agg(
            ["size", "sum"]
        )
        self.game_type_statistics_ = {
            (int(year), str(kind)): (int(row["size"]), int(row["sum"]))
            for (year, kind), row in type_grouped.iterrows()
        }
        self.fit_seasons_ = tuple(sorted(self.global_statistics_))
        self._combined_cache = {}
        self.is_fitted_ = True
        return self

    def _years(self, target_season: int, window: int) -> range:
        return range(target_season - window, target_season)

    def _prior(self, target_season: int, window: int, game_type: str) -> float:
        rows = successes = 0
        for year in self._years(target_season, window):
            count, positive = self.game_type_statistics_.get((year, game_type), (0, 0))
            rows += count
            successes += positive
        if rows:
            return successes / rows
        for year in self._years(target_season, window):
            count, positive = self.global_statistics_.get(year, (0, 0))
            rows += count
            successes += positive
        return successes / rows if rows else 0.5

    def _combined(
        self, spec: TargetEncodingSpec, target_season: int, window: int
    ) -> dict[tuple[str, ...], tuple[int, int]]:
        cache_key = (spec.name, target_season, window)
        cached = self._combined_cache.get(cache_key)
        if cached is not None:
            return cached
        totals: dict[tuple[str, ...], list[int]] = {}
        for year in self._years(target_season, window):
            for key, (count, positive) in self.group_statistics_.get(
                (spec.name, year), {}
            ).items():
                current = totals.setdefault(key, [0, 0])
                current[0] += count
                current[1] += positive
        result = {key: (values[0], values[1]) for key, values in totals.items()}
        self._combined_cache[cache_key] = result
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("PriorSeasonTargetEncoder must be fitted before transform")
        validate_preprocessing_schema(frame)
        season = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
        game_type = _tokens(frame["game_type"]).str.upper().to_numpy(dtype=object)
        output = {
            name: np.zeros(len(frame), dtype=np.float64)
            for name in TEMPORAL_TARGET_FEATURE_COLUMNS
        }
        for spec in TARGET_ENCODING_SPECS:
            key_frame = _canonical_key_frame(frame, spec.columns)
            if len(spec.columns) == 1:
                row_keys = key_frame.iloc[:, 0].map(lambda value: (str(value),))
            else:
                row_keys = pd.Series(
                    list(map(tuple, key_frame.astype(str).to_numpy())), index=frame.index
                )
            rates: dict[int, np.ndarray] = {}
            for target_season in np.unique(season):
                season_mask = season == target_season
                positions = np.flatnonzero(season_mask)
                selected_keys = row_keys.iloc[positions]
                selected_types = game_type[positions]
                for window in TEMPORAL_WINDOWS:
                    lookup = self._combined(spec, int(target_season), window)
                    counts = selected_keys.map(
                        {key: value[0] for key, value in lookup.items()}
                    ).fillna(0.0).to_numpy(dtype=np.float64)
                    positives = selected_keys.map(
                        {key: value[1] for key, value in lookup.items()}
                    ).fillna(0.0).to_numpy(dtype=np.float64)
                    priors = np.array(
                        [self._prior(int(target_season), window, str(kind)) for kind in selected_types],
                        dtype=np.float64,
                    )
                    posterior = (positives + spec.strength * priors) / (
                        counts + spec.strength
                    )
                    output[f"te_{spec.name}_prev{window}_rate"][positions] = posterior
                    output[f"te_{spec.name}_prev{window}_reliability"][positions] = counts / (
                        counts + spec.strength
                    )
                    output[f"te_{spec.name}_prev{window}_log1p_count"][positions] = np.log1p(
                        counts
                    )
                    rates.setdefault(window, np.zeros(len(frame), dtype=np.float64))[
                        positions
                    ] = posterior
            output[f"te_{spec.name}_prev1_minus_prev2"] = rates[1] - rates[2]
        result = pd.DataFrame(output, index=frame.index).loc[
            :, TEMPORAL_TARGET_FEATURE_COLUMNS
        ]
        if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
            raise RuntimeError("temporal target encoding produced non-finite values")
        return result

    def to_state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("cannot serialize an unfitted target encoder")
        groups = []
        for (name, year), mapping in sorted(self.group_statistics_.items()):
            for key, (count, positive) in sorted(mapping.items()):
                groups.append(
                    {
                        "spec": name,
                        "season": year,
                        "key": list(key),
                        "count": count,
                        "successes": positive,
                    }
                )
        return {
            "state_version": TEMPORAL_TARGET_STATE_VERSION,
            "feature_columns": list(TEMPORAL_TARGET_FEATURE_COLUMNS),
            "groups": groups,
            "global": [
                {"season": year, "count": value[0], "successes": value[1]}
                for year, value in sorted(self.global_statistics_.items())
            ],
            "game_type": [
                {
                    "season": year,
                    "game_type": kind,
                    "count": value[0],
                    "successes": value[1],
                }
                for (year, kind), value in sorted(self.game_type_statistics_.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "PriorSeasonTargetEncoder":
        """Restore an encoder without refitting or accessing labelled rows."""

        if not isinstance(payload, Mapping):
            raise TypeError("target encoder state must be a mapping")
        if payload.get("state_version") != TEMPORAL_TARGET_STATE_VERSION:
            raise ValueError("unsupported target encoder state version")
        if tuple(payload.get("feature_columns", ())) != TEMPORAL_TARGET_FEATURE_COLUMNS:
            raise ValueError("serialized target encoding feature order differs from code")

        specs = {spec.name: spec for spec in TARGET_ENCODING_SPECS}
        obj = cls()
        for record in payload.get("global", ()):  # type: ignore[union-attr]
            year = int(record["season"])
            count = int(record["count"])
            successes = int(record["successes"])
            if count < 0 or not 0 <= successes <= count or year in obj.global_statistics_:
                raise ValueError("invalid or duplicate global target encoding statistic")
            obj.global_statistics_[year] = (count, successes)
        if not obj.global_statistics_:
            raise ValueError("serialized target encoder has no global statistics")

        for record in payload.get("game_type", ()):  # type: ignore[union-attr]
            key = (int(record["season"]), str(record["game_type"]))
            count = int(record["count"])
            successes = int(record["successes"])
            if count < 0 or not 0 <= successes <= count or key in obj.game_type_statistics_:
                raise ValueError("invalid or duplicate game-type target encoding statistic")
            obj.game_type_statistics_[key] = (count, successes)

        for record in payload.get("groups", ()):  # type: ignore[union-attr]
            name = str(record["spec"])
            spec = specs.get(name)
            if spec is None:
                raise ValueError(f"unknown target encoding spec: {name}")
            raw_key = tuple(str(value) for value in record["key"])
            if len(raw_key) != len(spec.columns):
                raise ValueError(f"invalid target encoding key width for {name}")
            group_key = (name, int(record["season"]))
            mapping = obj.group_statistics_.setdefault(group_key, {})
            count = int(record["count"])
            successes = int(record["successes"])
            if count < 0 or not 0 <= successes <= count or raw_key in mapping:
                raise ValueError("invalid or duplicate grouped target encoding statistic")
            mapping[raw_key] = (count, successes)

        obj.fit_seasons_ = tuple(sorted(obj.global_statistics_))
        obj._combined_cache = {}
        obj.is_fitted_ = True
        return obj


class TemporalTargetAugmentedPreprocessor:
    """Canonical preprocessor plus compact strictly-prior entity encodings."""

    categorical_columns = DeepTabularPreprocessor.categorical_columns
    numerical_columns = (
        *DeepTabularPreprocessor.numerical_columns,
        *TEMPORAL_TARGET_FEATURE_COLUMNS,
    )

    def __init__(self, *, min_category_frequency: int = 2, clip_value: float = 10.0) -> None:
        self.base = DeepTabularPreprocessor(
            min_category_frequency=min_category_frequency, clip_value=clip_value
        )
        self.encoder = PriorSeasonTargetEncoder()
        self.extra_means_: dict[str, float] = {}
        self.extra_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "TemporalTargetAugmentedPreprocessor":
        self.base.fit(frame)
        self.encoder.fit(frame)
        extra = self.encoder.transform(frame)
        for column in TEMPORAL_TARGET_FEATURE_COLUMNS:
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            self.extra_means_[column] = float(values.mean())
            scale = float(values.std())
            self.extra_scales_[column] = scale if np.isfinite(scale) and scale >= 1e-8 else 1.0
        self.category_cardinalities_ = list(self.base.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("TemporalTargetAugmentedPreprocessor must be fitted")
        result = self.base.transform(frame)
        extra = self.encoder.transform(frame)
        array = np.empty((len(frame), len(TEMPORAL_TARGET_FEATURE_COLUMNS)), dtype=np.float32)
        for position, column in enumerate(TEMPORAL_TARGET_FEATURE_COLUMNS):
            values = (extra[column].to_numpy(dtype=np.float64) - self.extra_means_[column]) / self.extra_scales_[column]
            array[:, position] = np.clip(values, -self.base.clip_value, self.base.clip_value).astype(np.float32)
        result["numerical"] = np.concatenate([result["numerical"], array], axis=1)
        return result

    def feature_layout(self) -> TabularFeatureLayout:
        if not self.is_fitted_:
            raise RuntimeError("TemporalTargetAugmentedPreprocessor must be fitted")
        base_layout = TabularFeatureLayout.from_preprocessor(self.base)
        numerical = tuple(self.numerical_columns)
        return TabularFeatureLayout(
            categorical_names=base_layout.categorical_names,
            categorical_cardinalities=base_layout.categorical_cardinalities,
            numerical_names=numerical,
            pitcher_numerical_names=tuple(
                name
                for name in numerical
                if name in base_layout.pitcher_numerical_names
                or name.startswith("te_pitcher")
            ),
            batter_numerical_names=tuple(
                name
                for name in numerical
                if name in base_layout.batter_numerical_names
                or name.startswith("te_batter")
            ),
            preprocessor_fingerprint=_fingerprint(self.to_state_dict()),
        )

    def to_state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("cannot serialize an unfitted augmented preprocessor")
        return {
            "state_version": TEMPORAL_TARGET_STATE_VERSION,
            "base": self.base.to_state_dict(),
            "encoder": self.encoder.to_state_dict(),
            "extra_means": self.extra_means_,
            "extra_scales": self.extra_scales_,
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TemporalTargetAugmentedPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("temporal augmented state must be a mapping")
        if payload.get("state_version") != TEMPORAL_TARGET_STATE_VERSION:
            raise ValueError("unsupported temporal augmented state version")
        base = DeepTabularPreprocessor.from_state_dict(dict(payload["base"]))
        encoder = PriorSeasonTargetEncoder.from_state_dict(dict(payload["encoder"]))
        means = payload.get("extra_means")
        scales = payload.get("extra_scales")
        if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
            raise ValueError("serialized temporal scaling state is missing")
        expected = set(TEMPORAL_TARGET_FEATURE_COLUMNS)
        if set(means) != expected or set(scales) != expected:
            raise ValueError("serialized temporal scaling keys differ from code")
        obj = cls(
            min_category_frequency=base.min_category_frequency,
            clip_value=base.clip_value,
        )
        obj.base = base
        obj.encoder = encoder
        obj.extra_means_ = {
            name: float(means[name]) for name in TEMPORAL_TARGET_FEATURE_COLUMNS
        }
        obj.extra_scales_ = {
            name: float(scales[name]) for name in TEMPORAL_TARGET_FEATURE_COLUMNS
        }
        if not all(np.isfinite(value) for value in obj.extra_means_.values()):
            raise ValueError("serialized temporal means must be finite")
        if not all(
            np.isfinite(value) and value > 0.0 for value in obj.extra_scales_.values()
        ):
            raise ValueError("serialized temporal scales must be positive and finite")
        obj.category_cardinalities_ = list(base.category_cardinalities_)
        obj.is_fitted_ = True
        return obj


__all__ = [
    "PriorSeasonTargetEncoder",
    "TARGET_ENCODING_SPECS",
    "TEMPORAL_TARGET_FEATURE_COLUMNS",
    "TemporalTargetAugmentedPreprocessor",
]
