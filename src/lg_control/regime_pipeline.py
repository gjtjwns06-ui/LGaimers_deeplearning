"""Combined canonical and regime-aware preprocessing for the V2 neural path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .model import TabularFeatureLayout
from .preprocessing import DeepTabularPreprocessor
from .regime_features import REGIME_FEATURE_COLUMNS, RegimeFeatureBuilderV2


REGIME_PIPELINE_STATE_VERSION = 1


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RegimeAugmentedPreprocessor:
    """Append standardized, row-local V2 features to canonical neural arrays.

    The base category vocabularies, base numeric scalers, regime sufficient
    statistics, and V2 numeric scalers are fitted exactly once on the supplied
    temporal training partition.  ``transform`` is read-only and preserves the
    base preprocessor's row IDs and optional target array.
    """

    categorical_columns = DeepTabularPreprocessor.categorical_columns
    numerical_columns = (
        *DeepTabularPreprocessor.numerical_columns,
        *REGIME_FEATURE_COLUMNS,
    )

    def __init__(
        self,
        *,
        min_category_frequency: int = 2,
        clip_value: float = 10.0,
        regime_builder: RegimeFeatureBuilderV2 | None = None,
    ) -> None:
        self.base = DeepTabularPreprocessor(
            min_category_frequency=min_category_frequency,
            clip_value=clip_value,
        )
        self.regime = regime_builder or RegimeFeatureBuilderV2()
        self.regime_means_: dict[str, float] = {}
        self.regime_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "RegimeAugmentedPreprocessor":
        if len(frame) == 0:
            raise ValueError("cannot fit an empty regime-augmented preprocessor")
        self.base.fit(frame)
        self.regime.fit(frame)
        regime_frame = self.regime.transform(frame)
        self.regime_means_ = {}
        self.regime_scales_ = {}
        for column in REGIME_FEATURE_COLUMNS:
            values = regime_frame[column].to_numpy(dtype=np.float64, copy=False)
            mean = float(values.mean())
            scale = float(values.std())
            if not np.isfinite(mean):
                raise ValueError(f"regime feature {column} has a non-finite mean")
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            self.regime_means_[column] = mean
            self.regime_scales_[column] = scale
        self.category_cardinalities_ = list(self.base.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        result = self.base.transform(frame)
        regime_frame = self.regime.transform(frame)
        regime_array = np.empty(
            (len(frame), len(REGIME_FEATURE_COLUMNS)), dtype=np.float32
        )
        for position, column in enumerate(REGIME_FEATURE_COLUMNS):
            values = regime_frame[column].to_numpy(dtype=np.float64, copy=False)
            standardized = (values - self.regime_means_[column]) / self.regime_scales_[
                column
            ]
            regime_array[:, position] = np.clip(
                standardized, -self.base.clip_value, self.base.clip_value
            ).astype(np.float32)
        result["numerical"] = np.concatenate(
            [result["numerical"], regime_array], axis=1
        )
        return result

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    def feature_layout(self) -> TabularFeatureLayout:
        self._require_fitted()
        base_layout = TabularFeatureLayout.from_preprocessor(self.base)
        numerical_names = tuple(self.numerical_columns)
        pitcher_names = tuple(
            name
            for name in numerical_names
            if name in base_layout.pitcher_numerical_names
            or name.startswith("pitcher_")
        )
        batter_names = tuple(
            name
            for name in numerical_names
            if name in base_layout.batter_numerical_names
            or name.startswith("batter_")
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
            "state_version": REGIME_PIPELINE_STATE_VERSION,
            "base": self.base.to_state_dict(),
            "regime": self.regime.to_state_dict(),
            "regime_feature_columns": list(REGIME_FEATURE_COLUMNS),
            "regime_means": dict(self.regime_means_),
            "regime_scales": dict(self.regime_scales_),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "RegimeAugmentedPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("regime pipeline state must be a mapping")
        if payload.get("state_version") != REGIME_PIPELINE_STATE_VERSION:
            raise ValueError("unsupported regime pipeline state version")
        if tuple(payload.get("regime_feature_columns", ())) != REGIME_FEATURE_COLUMNS:
            raise ValueError("serialized regime feature order differs from code")
        base = DeepTabularPreprocessor.from_state_dict(dict(payload["base"]))
        regime = RegimeFeatureBuilderV2.from_state_dict(dict(payload["regime"]))
        obj = cls(
            min_category_frequency=base.min_category_frequency,
            clip_value=base.clip_value,
            regime_builder=regime,
        )
        obj.base = base
        raw_means = payload.get("regime_means")
        raw_scales = payload.get("regime_scales")
        if not isinstance(raw_means, Mapping) or not isinstance(raw_scales, Mapping):
            raise ValueError("serialized regime scaling state is missing")
        if set(raw_means) != set(REGIME_FEATURE_COLUMNS) or set(raw_scales) != set(
            REGIME_FEATURE_COLUMNS
        ):
            raise ValueError("serialized regime scaling keys differ from code")
        obj.regime_means_ = {name: float(raw_means[name]) for name in REGIME_FEATURE_COLUMNS}
        obj.regime_scales_ = {
            name: float(raw_scales[name]) for name in REGIME_FEATURE_COLUMNS
        }
        if not all(np.isfinite(value) for value in obj.regime_means_.values()):
            raise ValueError("serialized regime means must be finite")
        if not all(
            np.isfinite(value) and value > 0.0
            for value in obj.regime_scales_.values()
        ):
            raise ValueError("serialized regime scales must be positive and finite")
        obj.category_cardinalities_ = list(base.category_cardinalities_)
        obj.is_fitted_ = True
        return obj

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("RegimeAugmentedPreprocessor must be fitted before use")


__all__ = ["REGIME_PIPELINE_STATE_VERSION", "RegimeAugmentedPreprocessor"]
