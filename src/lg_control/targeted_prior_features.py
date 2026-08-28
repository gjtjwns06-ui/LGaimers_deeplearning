"""Compact, audit-selected, strictly prior-season features.

The lookup state is fitted from an explicitly identified fold-training
partition.  Every transformed row consults exactly ``season - 1``; no current
or future-season labels, transform-batch aggregates, or wider history windows
are used.  Auxiliary labels are reconstructed only inside the supplied
fold-training frame before their sufficient statistics are retained.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .auxiliary_targets import (
    FOLD_TRAIN_PARTITION,
    recover_fold_train_auxiliary_targets,
)
from .constants import TARGET_COL
from .model import TabularFeatureLayout
from .preprocessing import DeepTabularPreprocessor, validate_preprocessing_schema


TARGETED_PRIOR_STATE_VERSION = 1


@dataclass(frozen=True)
class TargetedLookupSpec:
    name: str
    label: str
    columns: tuple[str, ...]
    strength: float


TARGETED_LOOKUP_SPECS = (
    TargetedLookupSpec(
        "target_team_matchup_game_type",
        TARGET_COL,
        ("pitcher_team_id", "batter_team_id", "game_type"),
        100.0,
    ),
    TargetedLookupSpec(
        "target_batter_pitcher_hand",
        TARGET_COL,
        ("batter_id", "pitcher_hand"),
        100.0,
    ),
    TargetedLookupSpec(
        "target_game_hands_count",
        TARGET_COL,
        (
            "game_type",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
        ),
        100.0,
    ),
    TargetedLookupSpec(
        "aux_reverse_game_hands_count",
        "aux_reverse",
        (
            "game_type",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
        ),
        50.0,
    ),
    TargetedLookupSpec(
        "aux_reverse_pitcher_batter_hand",
        "aux_reverse",
        ("pitcher_id", "batter_hand"),
        50.0,
    ),
    TargetedLookupSpec(
        "aux_ball_pitcher_batter_hand",
        "aux_ball",
        ("pitcher_id", "batter_hand"),
        50.0,
    ),
    TargetedLookupSpec(
        "aux_middle_pitcher_game_batter_hand",
        "aux_middle",
        ("pitcher_id", "game_type", "batter_hand"),
        50.0,
    ),
)

TARGETED_LOOKUP_FEATURE_COLUMNS = tuple(
    feature
    for spec in TARGETED_LOOKUP_SPECS
    for feature in (
        f"{spec.name}_prev1_rate",
        f"{spec.name}_prev1_reliability",
        f"{spec.name}_prev1_log1p_count",
    )
)

TARGETED_ROW_LOCAL_FEATURE_COLUMNS = (
    "targeted_pitcher_middle_square",
    "targeted_pitcher_middle_logit",
    "targeted_batter_middle_square",
    "targeted_batter_middle_logit",
    "targeted_f_gate",
    "targeted_success_product",
    "targeted_success_product_one_minus_middle",
    "targeted_f_success_product",
    "targeted_f_success_product_one_minus_middle",
)

TARGETED_PRIOR_FEATURE_COLUMNS = (
    *TARGETED_LOOKUP_FEATURE_COLUMNS,
    *TARGETED_ROW_LOCAL_FEATURE_COLUMNS,
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


def _tokens(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    values = series.astype("string").str.strip().mask(lambda value: value.eq(""), pd.NA)
    values = values.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)
    return values.str.upper() if uppercase else values


def _key_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: _tokens(frame[column], uppercase=column == "game_type").fillna("<MISSING>")
            for column in columns
        },
        index=frame.index,
    )


def _row_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    keys = _key_frame(frame, columns)
    if len(columns) == 1:
        return keys.iloc[:, 0].map(lambda value: (str(value),))
    return pd.Series(
        list(map(tuple, keys.astype(str).to_numpy())),
        index=frame.index,
        dtype=object,
    )


def _finite_binary(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric)
    invalid = valid & ~np.isin(numeric, (0.0, 1.0))
    if invalid.any():
        raise ValueError(f"{name} must contain only binary values when present")
    return numeric


class TargetedPriorSeasonEncoder:
    """Store sufficient statistics and perform exact previous-season lookup."""

    feature_columns = TARGETED_LOOKUP_FEATURE_COLUMNS

    def __init__(self) -> None:
        self.group_statistics_: dict[
            tuple[str, int], dict[tuple[str, ...], tuple[int, int]]
        ] = {}
        self.game_type_statistics_: dict[tuple[str, int, str], tuple[int, int]] = {}
        self.global_statistics_: dict[tuple[str, int], tuple[int, int]] = {}
        self.fit_seasons_: tuple[int, ...] = ()
        self.is_fitted_ = False

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        partition: str,
    ) -> "TargetedPriorSeasonEncoder":
        if partition != FOLD_TRAIN_PARTITION:
            raise ValueError(
                "targeted prior fit requires partition='fold_train' before auxiliary reconstruction"
            )
        if len(frame) == 0 or TARGET_COL not in frame.columns:
            raise ValueError("targeted prior fit requires a non-empty labelled fold-train frame")
        validate_preprocessing_schema(frame)

        # Canonical row_id order makes reconstruction independent of caller
        # batching/shuffling while preserving the official pitch sequence.
        ordered = frame.assign(_targeted_row_order=_tokens(frame["row_id"])).sort_values(
            "_targeted_row_order", kind="stable"
        ).drop(columns="_targeted_row_order")
        auxiliary = recover_fold_train_auxiliary_targets(
            ordered, partition=FOLD_TRAIN_PARTITION
        )
        labels = ordered[[TARGET_COL]].copy()
        for column in ("aux_reverse", "aux_middle", "aux_ball"):
            labels[column] = auxiliary[column].to_numpy()

        season = pd.to_numeric(ordered["season"], errors="raise").to_numpy(dtype=np.int64)
        game_type = _tokens(ordered["game_type"], uppercase=True).fillna("<MISSING>")
        self.group_statistics_ = {}
        self.game_type_statistics_ = {}
        self.global_statistics_ = {}

        for spec in TARGETED_LOOKUP_SPECS:
            label_values = _finite_binary(labels[spec.label], name=spec.label)
            valid = np.isfinite(label_values)
            if valid.any():
                key_data = _key_frame(ordered, spec.columns)
                table = key_data.loc[valid].copy()
                table["_season"] = season[valid]
                table["_label"] = label_values[valid].astype(np.int64)
                grouped = table.groupby(
                    ["_season", *spec.columns],
                    sort=True,
                    dropna=False,
                    observed=True,
                )["_label"].agg(["size", "sum"])
                for index, row in grouped.iterrows():
                    year = int(index[0])
                    key = tuple(str(value) for value in index[1:])
                    self.group_statistics_.setdefault((spec.name, year), {})[key] = (
                        int(row["size"]),
                        int(row["sum"]),
                    )

                summary = pd.DataFrame(
                    {
                        "season": season[valid],
                        "game_type": game_type.loc[valid].to_numpy(dtype=object),
                        "label": label_values[valid].astype(np.int64),
                    }
                )
                global_grouped = summary.groupby("season")["label"].agg(["size", "sum"])
                for year, row in global_grouped.iterrows():
                    self.global_statistics_[(spec.label, int(year))] = (
                        int(row["size"]),
                        int(row["sum"]),
                    )
                type_grouped = summary.groupby(["season", "game_type"])["label"].agg(
                    ["size", "sum"]
                )
                for (year, kind), row in type_grouped.iterrows():
                    self.game_type_statistics_[(spec.label, int(year), str(kind))] = (
                        int(row["size"]),
                        int(row["sum"]),
                    )

        self.fit_seasons_ = tuple(sorted(set(int(value) for value in season)))
        self.is_fitted_ = True
        return self

    def _prior(self, label: str, year: int, game_type: str) -> float:
        count, positives = self.game_type_statistics_.get(
            (label, year, game_type), (0, 0)
        )
        if count:
            return positives / count
        count, positives = self.global_statistics_.get((label, year), (0, 0))
        return positives / count if count else 0.5

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("TargetedPriorSeasonEncoder must be fitted before transform")
        validate_preprocessing_schema(frame)
        seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
        game_types = _tokens(frame["game_type"], uppercase=True).fillna("<MISSING>").to_numpy(
            dtype=object
        )
        output = {
            column: np.zeros(len(frame), dtype=np.float64)
            for column in TARGETED_LOOKUP_FEATURE_COLUMNS
        }
        unique_seasons = np.unique(seasons)
        for spec in TARGETED_LOOKUP_SPECS:
            keys = _row_keys(frame, spec.columns)
            for target_season in unique_seasons:
                positions = np.flatnonzero(seasons == target_season)
                previous = int(target_season) - 1
                lookup = self.group_statistics_.get((spec.name, previous), {})
                count_map = {key: value[0] for key, value in lookup.items()}
                positive_map = {key: value[1] for key, value in lookup.items()}
                selected = keys.iloc[positions]
                counts = selected.map(count_map).fillna(0.0).to_numpy(dtype=np.float64)
                positives = selected.map(positive_map).fillna(0.0).to_numpy(dtype=np.float64)
                priors = np.fromiter(
                    (
                        self._prior(spec.label, previous, str(kind))
                        for kind in game_types[positions]
                    ),
                    dtype=np.float64,
                    count=len(positions),
                )
                output[f"{spec.name}_prev1_rate"][positions] = (
                    positives + spec.strength * priors
                ) / (counts + spec.strength)
                output[f"{spec.name}_prev1_reliability"][positions] = counts / (
                    counts + spec.strength
                )
                output[f"{spec.name}_prev1_log1p_count"][positions] = np.log1p(counts)
        result = pd.DataFrame(output, index=frame.index).loc[
            :, TARGETED_LOOKUP_FEATURE_COLUMNS
        ]
        if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
            raise RuntimeError("targeted prior lookup produced non-finite features")
        return result

    def to_state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("cannot serialize an unfitted targeted prior encoder")
        groups: list[dict[str, Any]] = []
        for (name, year), mapping in sorted(self.group_statistics_.items()):
            for key, (count, positives) in sorted(mapping.items()):
                groups.append(
                    {
                        "spec": name,
                        "season": year,
                        "key": list(key),
                        "count": count,
                        "successes": positives,
                    }
                )
        return {
            "state_version": TARGETED_PRIOR_STATE_VERSION,
            "feature_columns": list(TARGETED_LOOKUP_FEATURE_COLUMNS),
            "groups": groups,
            "global": [
                {
                    "label": label,
                    "season": year,
                    "count": count,
                    "successes": positives,
                }
                for (label, year), (count, positives) in sorted(
                    self.global_statistics_.items()
                )
            ],
            "game_type": [
                {
                    "label": label,
                    "season": year,
                    "game_type": kind,
                    "count": count,
                    "successes": positives,
                }
                for (label, year, kind), (count, positives) in sorted(
                    self.game_type_statistics_.items()
                )
            ],
            "fit_seasons": list(self.fit_seasons_),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "TargetedPriorSeasonEncoder":
        if not isinstance(payload, Mapping):
            raise TypeError("targeted prior encoder state must be a mapping")
        if payload.get("state_version") != TARGETED_PRIOR_STATE_VERSION:
            raise ValueError("unsupported targeted prior encoder state version")
        if tuple(payload.get("feature_columns", ())) != TARGETED_LOOKUP_FEATURE_COLUMNS:
            raise ValueError("serialized targeted prior feature order differs from code")
        specs = {spec.name: spec for spec in TARGETED_LOOKUP_SPECS}
        labels = {spec.label for spec in TARGETED_LOOKUP_SPECS}
        obj = cls()

        def statistic(record: Mapping[str, Any]) -> tuple[int, int]:
            count = int(record["count"])
            positives = int(record["successes"])
            if count < 0 or not 0 <= positives <= count:
                raise ValueError("invalid targeted prior statistic")
            return count, positives

        for record in payload.get("global", ()):  # type: ignore[union-attr]
            label = str(record["label"])
            if label not in labels:
                raise ValueError(f"unknown targeted prior label: {label}")
            key = (label, int(record["season"]))
            if key in obj.global_statistics_:
                raise ValueError("duplicate targeted global statistic")
            obj.global_statistics_[key] = statistic(record)
        for record in payload.get("game_type", ()):  # type: ignore[union-attr]
            label = str(record["label"])
            if label not in labels:
                raise ValueError(f"unknown targeted prior label: {label}")
            key = (label, int(record["season"]), str(record["game_type"]))
            if key in obj.game_type_statistics_:
                raise ValueError("duplicate targeted game-type statistic")
            obj.game_type_statistics_[key] = statistic(record)
        for record in payload.get("groups", ()):  # type: ignore[union-attr]
            name = str(record["spec"])
            spec = specs.get(name)
            if spec is None:
                raise ValueError(f"unknown targeted prior spec: {name}")
            raw_key = tuple(str(value) for value in record["key"])
            if len(raw_key) != len(spec.columns):
                raise ValueError(f"invalid targeted prior key width for {name}")
            group_key = (name, int(record["season"]))
            mapping = obj.group_statistics_.setdefault(group_key, {})
            if raw_key in mapping:
                raise ValueError("duplicate targeted grouped statistic")
            mapping[raw_key] = statistic(record)
        obj.fit_seasons_ = tuple(int(value) for value in payload.get("fit_seasons", ()))
        if not obj.fit_seasons_:
            raise ValueError("serialized targeted prior encoder has no fitted seasons")
        obj.is_fitted_ = True
        return obj


def targeted_row_local_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the nine audit-selected nonlinearities using this row only."""

    validate_preprocessing_schema(frame)

    def rate(column: str) -> np.ndarray:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        return np.clip(np.where(np.isfinite(values), values, 0.5), 0.0, 1.0)

    pitcher_middle = rate("asof_pitcher_middle_rate")
    batter_middle = rate("asof_batter_middle_rate")
    pitcher_success = rate("asof_pitcher_success_rate")
    batter_success = rate("asof_batter_success_rate")
    f_gate = _tokens(frame["game_type"], uppercase=True).eq("F").to_numpy(dtype=np.float64)
    success_product = pitcher_success * batter_success
    control_product = success_product * (1.0 - pitcher_middle)
    epsilon = 1e-4
    output = pd.DataFrame(
        {
            "targeted_pitcher_middle_square": pitcher_middle**2,
            "targeted_pitcher_middle_logit": np.log(
                np.clip(pitcher_middle, epsilon, 1.0 - epsilon)
                / np.clip(1.0 - pitcher_middle, epsilon, 1.0 - epsilon)
            ),
            "targeted_batter_middle_square": batter_middle**2,
            "targeted_batter_middle_logit": np.log(
                np.clip(batter_middle, epsilon, 1.0 - epsilon)
                / np.clip(1.0 - batter_middle, epsilon, 1.0 - epsilon)
            ),
            "targeted_f_gate": f_gate,
            "targeted_success_product": success_product,
            "targeted_success_product_one_minus_middle": control_product,
            "targeted_f_success_product": f_gate * success_product,
            "targeted_f_success_product_one_minus_middle": f_gate * control_product,
        },
        index=frame.index,
    ).loc[:, TARGETED_ROW_LOCAL_FEATURE_COLUMNS]
    if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("targeted row-local features must be finite")
    return output


class TargetedPriorAugmentedPreprocessor:
    """Canonical neural inputs plus 30 compact targeted prior features."""

    categorical_columns = DeepTabularPreprocessor.categorical_columns
    numerical_columns = (
        *DeepTabularPreprocessor.numerical_columns,
        *TARGETED_PRIOR_FEATURE_COLUMNS,
    )

    def __init__(self, *, min_category_frequency: int = 2, clip_value: float = 10.0) -> None:
        self.base = DeepTabularPreprocessor(
            min_category_frequency=min_category_frequency,
            clip_value=clip_value,
        )
        self.encoder = TargetedPriorSeasonEncoder()
        self.extra_means_: dict[str, float] = {}
        self.extra_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        partition: str,
    ) -> "TargetedPriorAugmentedPreprocessor":
        if partition != FOLD_TRAIN_PARTITION:
            raise ValueError("targeted preprocessor fit requires partition='fold_train'")
        self.base.fit(frame)
        self.encoder.fit(frame, partition=partition)
        extra = pd.concat(
            [self.encoder.transform(frame), targeted_row_local_features(frame)], axis=1
        ).loc[:, TARGETED_PRIOR_FEATURE_COLUMNS]
        self.extra_means_ = {}
        self.extra_scales_ = {}
        for column in TARGETED_PRIOR_FEATURE_COLUMNS:
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            self.extra_means_[column] = mean
            self.extra_scales_[column] = scale if np.isfinite(scale) and scale >= 1e-8 else 1.0
        self.category_cardinalities_ = list(self.base.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        result = self.base.transform(frame)
        extra = pd.concat(
            [self.encoder.transform(frame), targeted_row_local_features(frame)], axis=1
        ).loc[:, TARGETED_PRIOR_FEATURE_COLUMNS]
        array = np.empty((len(frame), len(TARGETED_PRIOR_FEATURE_COLUMNS)), dtype=np.float32)
        for position, column in enumerate(TARGETED_PRIOR_FEATURE_COLUMNS):
            standardized = (
                extra[column].to_numpy(dtype=np.float64) - self.extra_means_[column]
            ) / self.extra_scales_[column]
            array[:, position] = np.clip(
                standardized, -self.base.clip_value, self.base.clip_value
            ).astype(np.float32)
        result["numerical"] = np.concatenate([result["numerical"], array], axis=1)
        return result

    def fit_transform(
        self,
        frame: pd.DataFrame,
        *,
        partition: str,
    ) -> dict[str, np.ndarray]:
        return self.fit(frame, partition=partition).transform(frame)

    def feature_layout(self) -> TabularFeatureLayout:
        self._require_fitted()
        base_layout = TabularFeatureLayout.from_preprocessor(self.base)
        return TabularFeatureLayout(
            categorical_names=base_layout.categorical_names,
            categorical_cardinalities=base_layout.categorical_cardinalities,
            numerical_names=tuple(self.numerical_columns),
            pitcher_numerical_names=base_layout.pitcher_numerical_names,
            batter_numerical_names=base_layout.batter_numerical_names,
            preprocessor_fingerprint=_fingerprint(self.to_state_dict()),
        )

    def to_state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_version": TARGETED_PRIOR_STATE_VERSION,
            "feature_columns": list(TARGETED_PRIOR_FEATURE_COLUMNS),
            "base": self.base.to_state_dict(),
            "encoder": self.encoder.to_state_dict(),
            "extra_means": dict(self.extra_means_),
            "extra_scales": dict(self.extra_scales_),
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TargetedPriorAugmentedPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("targeted augmented state must be a mapping")
        if payload.get("state_version") != TARGETED_PRIOR_STATE_VERSION:
            raise ValueError("unsupported targeted augmented state version")
        if tuple(payload.get("feature_columns", ())) != TARGETED_PRIOR_FEATURE_COLUMNS:
            raise ValueError("serialized targeted augmented feature order differs from code")
        base = DeepTabularPreprocessor.from_state_dict(dict(payload["base"]))
        encoder = TargetedPriorSeasonEncoder.from_state_dict(dict(payload["encoder"]))
        means = payload.get("extra_means")
        scales = payload.get("extra_scales")
        if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
            raise ValueError("serialized targeted scaling state is missing")
        expected = set(TARGETED_PRIOR_FEATURE_COLUMNS)
        if set(means) != expected or set(scales) != expected:
            raise ValueError("serialized targeted scaling keys differ from code")
        obj = cls(
            min_category_frequency=base.min_category_frequency,
            clip_value=base.clip_value,
        )
        obj.base = base
        obj.encoder = encoder
        obj.extra_means_ = {
            name: float(means[name]) for name in TARGETED_PRIOR_FEATURE_COLUMNS
        }
        obj.extra_scales_ = {
            name: float(scales[name]) for name in TARGETED_PRIOR_FEATURE_COLUMNS
        }
        if not all(np.isfinite(value) for value in obj.extra_means_.values()):
            raise ValueError("serialized targeted means must be finite")
        if not all(
            np.isfinite(value) and value > 0.0 for value in obj.extra_scales_.values()
        ):
            raise ValueError("serialized targeted scales must be positive and finite")
        obj.category_cardinalities_ = list(base.category_cardinalities_)
        obj.is_fitted_ = True
        return obj

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("TargetedPriorAugmentedPreprocessor must be fitted before use")


__all__ = [
    "TARGETED_LOOKUP_FEATURE_COLUMNS",
    "TARGETED_LOOKUP_SPECS",
    "TARGETED_PRIOR_FEATURE_COLUMNS",
    "TARGETED_PRIOR_STATE_VERSION",
    "TARGETED_ROW_LOCAL_FEATURE_COLUMNS",
    "TargetedPriorAugmentedPreprocessor",
    "TargetedPriorSeasonEncoder",
    "targeted_row_local_features",
]
