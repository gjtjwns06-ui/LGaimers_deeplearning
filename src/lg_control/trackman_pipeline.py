"""Neural preprocessing that appends strict prior-season TrackMan features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .model import TabularFeatureLayout
from .strict_trackman import ROLE_FEATURE_COLUMNS, StrictTrackmanFeatureBuilder
from .temporal_target_features import TemporalTargetAugmentedPreprocessor


TRACKMAN_PIPELINE_STATE_VERSION = 1


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StrictTrackmanAugmentedPreprocessor:
    """Compact temporal-target inputs plus strict pitcher TrackMan history."""

    trackman_columns = ROLE_FEATURE_COLUMNS["pitcher"]
    categorical_columns = TemporalTargetAugmentedPreprocessor.categorical_columns
    numerical_columns = (
        *TemporalTargetAugmentedPreprocessor.numerical_columns,
        *trackman_columns,
    )

    def __init__(
        self,
        trackman_builder: StrictTrackmanFeatureBuilder,
        *,
        min_category_frequency: int = 2,
        clip_value: float = 10.0,
    ) -> None:
        if not isinstance(trackman_builder, StrictTrackmanFeatureBuilder):
            raise TypeError("trackman_builder must be a StrictTrackmanFeatureBuilder")
        if not trackman_builder.is_fitted_:
            raise RuntimeError("trackman_builder must be fitted before neural preprocessing")
        self.base = TemporalTargetAugmentedPreprocessor(
            min_category_frequency=min_category_frequency,
            clip_value=clip_value,
        )
        self.trackman = trackman_builder
        self.trackman_means_: dict[str, float] = {}
        self.trackman_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "StrictTrackmanAugmentedPreprocessor":
        self.base.fit(frame)
        extra = self.trackman.transform(frame).loc[:, self.trackman_columns]
        for column in self.trackman_columns:
            values = extra[column].to_numpy(dtype=np.float64, copy=False)
            mean = float(values.mean())
            scale = float(values.std())
            if not np.isfinite(mean):
                raise ValueError(f"TrackMan feature {column} has a non-finite mean")
            self.trackman_means_[column] = mean
            self.trackman_scales_[column] = (
                scale if np.isfinite(scale) and scale >= 1e-8 else 1.0
            )
        self.category_cardinalities_ = list(self.base.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("StrictTrackmanAugmentedPreprocessor must be fitted")
        result = self.base.transform(frame)
        extra = self.trackman.transform(frame).loc[:, self.trackman_columns]
        array = np.empty((len(frame), len(self.trackman_columns)), dtype=np.float32)
        for position, column in enumerate(self.trackman_columns):
            values = (
                extra[column].to_numpy(dtype=np.float64, copy=False)
                - self.trackman_means_[column]
            ) / self.trackman_scales_[column]
            array[:, position] = np.clip(
                values, -self.base.base.clip_value, self.base.base.clip_value
            ).astype(np.float32)
        result["numerical"] = np.concatenate([result["numerical"], array], axis=1)
        return result

    def feature_layout(self) -> TabularFeatureLayout:
        if not self.is_fitted_:
            raise RuntimeError("StrictTrackmanAugmentedPreprocessor must be fitted")
        base_layout = self.base.feature_layout()
        numerical = tuple(self.numerical_columns)
        return TabularFeatureLayout(
            categorical_names=base_layout.categorical_names,
            categorical_cardinalities=base_layout.categorical_cardinalities,
            numerical_names=numerical,
            pitcher_numerical_names=(
                *base_layout.pitcher_numerical_names,
                *self.trackman_columns,
            ),
            batter_numerical_names=base_layout.batter_numerical_names,
            preprocessor_fingerprint=_fingerprint(self.to_state_dict()),
        )

    def to_state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("cannot serialize an unfitted TrackMan preprocessor")
        return {
            "state_version": TRACKMAN_PIPELINE_STATE_VERSION,
            "base": self.base.to_state_dict(),
            "trackman": self.trackman.to_state_dict(),
            "trackman_columns": list(self.trackman_columns),
            "trackman_means": dict(self.trackman_means_),
            "trackman_scales": dict(self.trackman_scales_),
        }


__all__ = ["TRACKMAN_PIPELINE_STATE_VERSION", "StrictTrackmanAugmentedPreprocessor"]
