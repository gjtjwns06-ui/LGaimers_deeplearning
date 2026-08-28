"""Fold-safe prior-season encodings of reconstructed auxiliary outcomes.

Only :meth:`TemporalAuxAugmentedPreprocessor.fit` is allowed to reconstruct
auxiliary labels.  The caller must pass an already isolated fold-training
frame.  All inference transforms are target-free, row-local lookups into
season-level sufficient statistics learned during that fit.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .auxiliary_targets import recover_fold_train_auxiliary_targets
from .model import TabularFeatureLayout
from .preprocessing import DeepTabularPreprocessor, validate_preprocessing_schema


TEMPORAL_AUX_STATE_VERSION = 1
AUXILIARY_OUTCOMES = ("reverse", "middle", "ball", "strike")
TEMPORAL_AUX_WINDOWS = (1, 2)


@dataclass(frozen=True)
class AuxiliaryEncodingSpec:
    name: str
    columns: tuple[str, ...]
    strength: float


AUXILIARY_ENCODING_SPECS = (
    AuxiliaryEncodingSpec("pitcher", ("pitcher_id",), 100.0),
    AuxiliaryEncodingSpec(
        "pitcher_game_type", ("pitcher_id", "game_type"), 100.0
    ),
    AuxiliaryEncodingSpec(
        "pitcher_batter_hand", ("pitcher_id", "batter_hand"), 100.0
    ),
    AuxiliaryEncodingSpec(
        "count_hands",
        ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
        500.0,
    ),
)


def _feature_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for spec in AUXILIARY_ENCODING_SPECS:
        for outcome in AUXILIARY_OUTCOMES:
            for window in TEMPORAL_AUX_WINDOWS:
                columns.append(f"auxte_{spec.name}_{outcome}_prev{window}_rate")
            columns.append(f"auxte_{spec.name}_{outcome}_prev1_minus_prev2")
        for window in TEMPORAL_AUX_WINDOWS:
            columns.extend(
                (
                    f"auxte_{spec.name}_prev{window}_reliability",
                    f"auxte_{spec.name}_prev{window}_log1p_count",
                )
            )
    return tuple(columns)


TEMPORAL_AUX_FEATURE_COLUMNS = _feature_columns()


def _tokens(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)


def _key_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: _tokens(frame[column]).fillna("<MISSING>") for column in columns},
        index=frame.index,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PriorSeasonAuxiliaryEncoder:
    """Season-level auxiliary sufficient statistics with strict time lookup."""

    feature_columns = TEMPORAL_AUX_FEATURE_COLUMNS

    def __init__(self) -> None:
        # Each value is (valid count, reverse, middle, ball, strike positives).
        self.group_statistics_: dict[
            tuple[str, int], dict[tuple[str, ...], tuple[int, int, int, int, int]]
        ] = {}
        self.game_type_statistics_: dict[
            tuple[int, str], tuple[int, int, int, int, int]
        ] = {}
        self.global_statistics_: dict[int, tuple[int, int, int, int, int]] = {}
        self.fit_seasons_: tuple[int, ...] = ()
        self.is_fitted_ = False
        self._combined_cache: dict[
            tuple[str, int, int],
            dict[tuple[str, ...], tuple[int, int, int, int, int]],
        ] = {}

    def fit(self, fold_train: pd.DataFrame) -> "PriorSeasonAuxiliaryEncoder":
        if not isinstance(fold_train, pd.DataFrame) or len(fold_train) == 0:
            raise ValueError("auxiliary encoder fit requires a non-empty fold-train frame")
        validate_preprocessing_schema(fold_train)
        recovered = recover_fold_train_auxiliary_targets(
            fold_train, partition="fold_train"
        )
        season = pd.to_numeric(fold_train["season"], errors="raise").to_numpy(
            dtype=np.int64
        )
        valid = recovered["auxiliary_valid_mask"].to_numpy(dtype=bool)
        if not valid.any():
            raise ValueError(
                "fold-train frame has no recoverable auxiliary labels; at least one "
                "within-fold n+1 pitcher transition is required"
            )
        outcomes = {
            name: recovered[f"aux_{name}"].to_numpy(dtype=np.float64)
            for name in AUXILIARY_OUTCOMES
        }

        self.group_statistics_ = {}
        for spec in AUXILIARY_ENCODING_SPECS:
            keys = _key_frame(fold_train, spec.columns)
            table = keys.loc[valid].copy()
            table.insert(0, "season", season[valid])
            for name in AUXILIARY_OUTCOMES:
                table[name] = outcomes[name][valid].astype(np.int64)
            grouped = table.groupby(
                ["season", *spec.columns],
                sort=True,
                dropna=False,
                observed=True,
            ).agg(
                count=("reverse", "size"),
                reverse=("reverse", "sum"),
                middle=("middle", "sum"),
                ball=("ball", "sum"),
                strike=("strike", "sum"),
            )
            by_season: dict[
                int, dict[tuple[str, ...], tuple[int, int, int, int, int]]
            ] = defaultdict(dict)
            for index, row in grouped.iterrows():
                year = int(index[0])
                key = tuple(str(value) for value in index[1:])
                by_season[year][key] = tuple(
                    int(row[column])
                    for column in ("count", *AUXILIARY_OUTCOMES)
                )
            for year, mapping in by_season.items():
                self.group_statistics_[(spec.name, year)] = mapping

        summary = pd.DataFrame(
            {
                "season": season[valid],
                "game_type": _tokens(fold_train.loc[valid, "game_type"])
                .str.upper()
                .fillna("<MISSING>")
                .to_numpy(),
                **{name: values[valid].astype(np.int64) for name, values in outcomes.items()},
            }
        )
        global_grouped = summary.groupby("season", sort=True).agg(
            count=("reverse", "size"),
            **{name: (name, "sum") for name in AUXILIARY_OUTCOMES},
        )
        self.global_statistics_ = {
            int(year): tuple(
                int(row[column]) for column in ("count", *AUXILIARY_OUTCOMES)
            )
            for year, row in global_grouped.iterrows()
        }
        type_grouped = summary.groupby(
            ["season", "game_type"], sort=True, observed=True
        ).agg(
            count=("reverse", "size"),
            **{name: (name, "sum") for name in AUXILIARY_OUTCOMES},
        )
        self.game_type_statistics_ = {
            (int(year), str(kind)): tuple(
                int(row[column]) for column in ("count", *AUXILIARY_OUTCOMES)
            )
            for (year, kind), row in type_grouped.iterrows()
        }
        self.fit_seasons_ = tuple(sorted(self.global_statistics_))
        self._combined_cache = {}
        self.is_fitted_ = True
        return self

    @staticmethod
    def _years(target_season: int, window: int) -> range:
        return range(target_season - window, target_season)

    def _prior(
        self, target_season: int, window: int, game_type: str, outcome_index: int
    ) -> float:
        count = positive = 0
        for year in self._years(target_season, window):
            values = self.game_type_statistics_.get((year, game_type))
            if values is not None:
                count += values[0]
                positive += values[outcome_index + 1]
        if count:
            return positive / count
        for year in self._years(target_season, window):
            values = self.global_statistics_.get(year)
            if values is not None:
                count += values[0]
                positive += values[outcome_index + 1]
        return positive / count if count else 0.5

    def _combined(
        self, spec: AuxiliaryEncodingSpec, target_season: int, window: int
    ) -> dict[tuple[str, ...], tuple[int, int, int, int, int]]:
        cache_key = (spec.name, target_season, window)
        cached = self._combined_cache.get(cache_key)
        if cached is not None:
            return cached
        totals: dict[tuple[str, ...], list[int]] = {}
        for year in self._years(target_season, window):
            for key, values in self.group_statistics_.get((spec.name, year), {}).items():
                current = totals.setdefault(key, [0, 0, 0, 0, 0])
                for position, value in enumerate(values):
                    current[position] += value
        result = {key: tuple(values) for key, values in totals.items()}
        self._combined_cache[cache_key] = result
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("PriorSeasonAuxiliaryEncoder must be fitted before transform")
        validate_preprocessing_schema(frame)
        season = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
        game_type = (
            _tokens(frame["game_type"])
            .str.upper()
            .fillna("<MISSING>")
            .to_numpy(dtype=object)
        )
        output = {
            name: np.zeros(len(frame), dtype=np.float64)
            for name in TEMPORAL_AUX_FEATURE_COLUMNS
        }
        for spec in AUXILIARY_ENCODING_SPECS:
            keys = _key_frame(frame, spec.columns)
            row_keys = pd.Series(
                list(map(tuple, keys.astype(str).to_numpy())), index=frame.index
            )
            rates = {
                (outcome, window): np.zeros(len(frame), dtype=np.float64)
                for outcome in AUXILIARY_OUTCOMES
                for window in TEMPORAL_AUX_WINDOWS
            }
            for target_season in np.unique(season):
                positions = np.flatnonzero(season == target_season)
                selected_keys = row_keys.iloc[positions]
                selected_types = game_type[positions]
                for window in TEMPORAL_AUX_WINDOWS:
                    lookup = self._combined(spec, int(target_season), window)
                    counts_lookup = {key: values[0] for key, values in lookup.items()}
                    counts = selected_keys.map(counts_lookup).fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
                    output[f"auxte_{spec.name}_prev{window}_reliability"][positions] = (
                        counts / (counts + spec.strength)
                    )
                    output[f"auxte_{spec.name}_prev{window}_log1p_count"][positions] = (
                        np.log1p(counts)
                    )
                    for outcome_index, outcome in enumerate(AUXILIARY_OUTCOMES):
                        positives_lookup = {
                            key: values[outcome_index + 1]
                            for key, values in lookup.items()
                        }
                        positives = (
                            selected_keys.map(positives_lookup)
                            .fillna(0.0)
                            .to_numpy(dtype=np.float64)
                        )
                        priors = np.fromiter(
                            (
                                self._prior(
                                    int(target_season),
                                    window,
                                    str(kind),
                                    outcome_index,
                                )
                                for kind in selected_types
                            ),
                            dtype=np.float64,
                            count=len(positions),
                        )
                        posterior = (positives + spec.strength * priors) / (
                            counts + spec.strength
                        )
                        rates[(outcome, window)][positions] = posterior
                        output[
                            f"auxte_{spec.name}_{outcome}_prev{window}_rate"
                        ][positions] = posterior
            for outcome in AUXILIARY_OUTCOMES:
                output[f"auxte_{spec.name}_{outcome}_prev1_minus_prev2"] = (
                    rates[(outcome, 1)] - rates[(outcome, 2)]
                )
        result = pd.DataFrame(output, index=frame.index).loc[
            :, TEMPORAL_AUX_FEATURE_COLUMNS
        ]
        if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
            raise RuntimeError("temporal auxiliary encoding produced non-finite values")
        return result

    def to_state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("cannot serialize an unfitted auxiliary encoder")
        groups: list[dict[str, Any]] = []
        for (name, year), mapping in sorted(self.group_statistics_.items()):
            for key, values in sorted(mapping.items()):
                groups.append(
                    {
                        "spec": name,
                        "season": year,
                        "key": list(key),
                        "values": list(values),
                    }
                )
        return {
            "state_version": TEMPORAL_AUX_STATE_VERSION,
            "feature_columns": list(TEMPORAL_AUX_FEATURE_COLUMNS),
            "specs": [
                {
                    "name": spec.name,
                    "columns": list(spec.columns),
                    "strength": spec.strength,
                }
                for spec in AUXILIARY_ENCODING_SPECS
            ],
            "groups": groups,
            "global": [
                {"season": year, "values": list(values)}
                for year, values in sorted(self.global_statistics_.items())
            ],
            "game_type": [
                {"season": year, "game_type": kind, "values": list(values)}
                for (year, kind), values in sorted(self.game_type_statistics_.items())
            ],
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PriorSeasonAuxiliaryEncoder":
        if not isinstance(payload, Mapping):
            raise TypeError("auxiliary encoder state must be a mapping")
        if payload.get("state_version") != TEMPORAL_AUX_STATE_VERSION:
            raise ValueError("unsupported temporal auxiliary encoder state version")
        if tuple(payload.get("feature_columns", ())) != TEMPORAL_AUX_FEATURE_COLUMNS:
            raise ValueError("serialized auxiliary feature order differs from code")
        expected_specs = [
            {
                "name": spec.name,
                "columns": list(spec.columns),
                "strength": spec.strength,
            }
            for spec in AUXILIARY_ENCODING_SPECS
        ]
        if payload.get("specs") != expected_specs:
            raise ValueError("serialized auxiliary encoding specs differ from code")

        obj = cls()

        def values_tuple(raw: Any) -> tuple[int, int, int, int, int]:
            if not isinstance(raw, list) or len(raw) != 5:
                raise ValueError("serialized auxiliary sufficient statistic is malformed")
            values = tuple(int(value) for value in raw)
            if any(value < 0 for value in values) or any(
                positive > values[0] for positive in values[1:]
            ):
                raise ValueError("serialized auxiliary sufficient statistic is invalid")
            return values  # type: ignore[return-value]

        groups: dict[
            tuple[str, int], dict[tuple[str, ...], tuple[int, int, int, int, int]]
        ] = defaultdict(dict)
        spec_names = {spec.name for spec in AUXILIARY_ENCODING_SPECS}
        for row in payload.get("groups", []):
            name = str(row["spec"])
            if name not in spec_names:
                raise ValueError("serialized auxiliary group has an unknown spec")
            year = int(row["season"])
            key = tuple(str(value) for value in row["key"])
            groups[(name, year)][key] = values_tuple(row["values"])
        obj.group_statistics_ = dict(groups)
        obj.global_statistics_ = {
            int(row["season"]): values_tuple(row["values"])
            for row in payload.get("global", [])
        }
        obj.game_type_statistics_ = {
            (int(row["season"]), str(row["game_type"])): values_tuple(row["values"])
            for row in payload.get("game_type", [])
        }
        if not obj.global_statistics_:
            raise ValueError("serialized auxiliary encoder has no global statistics")
        obj.fit_seasons_ = tuple(sorted(obj.global_statistics_))
        obj._combined_cache = {}
        obj.is_fitted_ = True
        return obj


class TemporalAuxAugmentedPreprocessor:
    """Canonical neural preprocessing plus 64 prior-season auxiliary features."""

    categorical_columns = DeepTabularPreprocessor.categorical_columns
    numerical_columns = (
        *DeepTabularPreprocessor.numerical_columns,
        *TEMPORAL_AUX_FEATURE_COLUMNS,
    )

    def __init__(self, *, min_category_frequency: int = 2, clip_value: float = 10.0):
        self.base = DeepTabularPreprocessor(
            min_category_frequency=min_category_frequency, clip_value=clip_value
        )
        self.encoder = PriorSeasonAuxiliaryEncoder()
        self.extra_means_: dict[str, float] = {}
        self.extra_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(self, fold_train: pd.DataFrame) -> "TemporalAuxAugmentedPreprocessor":
        if len(fold_train) == 0:
            raise ValueError("cannot fit an empty temporal-aux preprocessor")
        self.base.fit(fold_train)
        self.encoder.fit(fold_train)
        extra = self.encoder.transform(fold_train)
        self.extra_means_ = {}
        self.extra_scales_ = {}
        for column in TEMPORAL_AUX_FEATURE_COLUMNS:
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(mean):
                raise ValueError(f"auxiliary feature {column} has a non-finite mean")
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            self.extra_means_[column] = mean
            self.extra_scales_[column] = scale
        self.category_cardinalities_ = list(self.base.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        result = self.base.transform(frame)
        extra = self.encoder.transform(frame)
        array = np.empty((len(frame), len(TEMPORAL_AUX_FEATURE_COLUMNS)), dtype=np.float32)
        for position, column in enumerate(TEMPORAL_AUX_FEATURE_COLUMNS):
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            standardized = (values - self.extra_means_[column]) / self.extra_scales_[column]
            array[:, position] = np.clip(
                standardized, -self.base.clip_value, self.base.clip_value
            ).astype(np.float32)
        result["numerical"] = np.concatenate([result["numerical"], array], axis=1)
        return result

    def fit_transform(self, fold_train: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(fold_train).transform(fold_train)

    def feature_layout(self) -> TabularFeatureLayout:
        self._require_fitted()
        base_layout = TabularFeatureLayout.from_preprocessor(self.base)
        numerical_names = tuple(self.numerical_columns)
        return TabularFeatureLayout(
            categorical_names=base_layout.categorical_names,
            categorical_cardinalities=base_layout.categorical_cardinalities,
            numerical_names=numerical_names,
            pitcher_numerical_names=tuple(
                name
                for name in numerical_names
                if name in base_layout.pitcher_numerical_names
                or name.startswith("auxte_pitcher")
            ),
            batter_numerical_names=base_layout.batter_numerical_names,
            preprocessor_fingerprint=_fingerprint(self.to_state_dict()),
        )

    def to_state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_version": TEMPORAL_AUX_STATE_VERSION,
            "preprocessor_class": type(self).__name__,
            "feature_columns": list(TEMPORAL_AUX_FEATURE_COLUMNS),
            "base": self.base.to_state_dict(),
            "encoder": self.encoder.to_state_dict(),
            "extra_means": dict(self.extra_means_),
            "extra_scales": dict(self.extra_scales_),
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TemporalAuxAugmentedPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("temporal auxiliary preprocessor state must be a mapping")
        if payload.get("state_version") != TEMPORAL_AUX_STATE_VERSION:
            raise ValueError("unsupported temporal auxiliary preprocessor state version")
        if payload.get("preprocessor_class") != cls.__name__:
            raise ValueError("serialized temporal auxiliary preprocessor class differs")
        if tuple(payload.get("feature_columns", ())) != TEMPORAL_AUX_FEATURE_COLUMNS:
            raise ValueError("serialized temporal auxiliary feature order differs from code")
        base = DeepTabularPreprocessor.from_state_dict(dict(payload["base"]))
        encoder = PriorSeasonAuxiliaryEncoder.from_state_dict(payload["encoder"])
        obj = cls(
            min_category_frequency=base.min_category_frequency,
            clip_value=base.clip_value,
        )
        obj.base = base
        obj.encoder = encoder
        raw_means = payload.get("extra_means")
        raw_scales = payload.get("extra_scales")
        if not isinstance(raw_means, Mapping) or not isinstance(raw_scales, Mapping):
            raise ValueError("serialized temporal auxiliary scaling state is missing")
        if set(raw_means) != set(TEMPORAL_AUX_FEATURE_COLUMNS) or set(raw_scales) != set(
            TEMPORAL_AUX_FEATURE_COLUMNS
        ):
            raise ValueError("serialized temporal auxiliary scaling keys differ from code")
        obj.extra_means_ = {
            name: float(raw_means[name]) for name in TEMPORAL_AUX_FEATURE_COLUMNS
        }
        obj.extra_scales_ = {
            name: float(raw_scales[name]) for name in TEMPORAL_AUX_FEATURE_COLUMNS
        }
        if not all(np.isfinite(value) for value in obj.extra_means_.values()):
            raise ValueError("serialized temporal auxiliary means must be finite")
        if not all(
            np.isfinite(value) and value > 0.0 for value in obj.extra_scales_.values()
        ):
            raise ValueError("serialized temporal auxiliary scales must be finite and positive")
        obj.category_cardinalities_ = list(base.category_cardinalities_)
        obj.is_fitted_ = True
        return obj

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("TemporalAuxAugmentedPreprocessor must be fitted before use")


__all__ = [
    "AUXILIARY_ENCODING_SPECS",
    "AUXILIARY_OUTCOMES",
    "PriorSeasonAuxiliaryEncoder",
    "TEMPORAL_AUX_FEATURE_COLUMNS",
    "TEMPORAL_AUX_STATE_VERSION",
    "TemporalAuxAugmentedPreprocessor",
]
