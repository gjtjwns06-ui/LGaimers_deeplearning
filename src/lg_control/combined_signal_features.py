"""Combined compact rate bases and strictly prior-season target encodings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .model import TabularFeatureLayout
from .rate_basis_features import (
    RATE_BASIS_FEATURE_COLUMNS,
    RateBasisAugmentedPreprocessor,
)
from .temporal_target_features import (
    TEMPORAL_TARGET_FEATURE_COLUMNS,
    PriorSeasonTargetEncoder,
)


COMBINED_SIGNAL_STATE_VERSION = 1
COMBINED_SIGNAL_FEATURE_COLUMNS = (
    *RATE_BASIS_FEATURE_COLUMNS,
    *TEMPORAL_TARGET_FEATURE_COLUMNS,
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


class CombinedSignalAugmentedPreprocessor:
    """Append both target-independent rate bases and prior-season entity TE.

    The rate bases use only the current row's official pre-pitch columns.  The
    target encodings store sufficient statistics from the supplied fold-train
    rows and expose only seasons strictly earlier than each transformed row.
    """

    categorical_columns = RateBasisAugmentedPreprocessor.categorical_columns
    numerical_columns = (
        *RateBasisAugmentedPreprocessor.numerical_columns,
        *TEMPORAL_TARGET_FEATURE_COLUMNS,
    )

    def __init__(self, *, min_category_frequency: int = 2, clip_value: float = 10.0):
        self.rate = RateBasisAugmentedPreprocessor(
            min_category_frequency=min_category_frequency,
            clip_value=clip_value,
        )
        self.encoder = PriorSeasonTargetEncoder()
        self.temporal_means_: dict[str, float] = {}
        self.temporal_scales_: dict[str, float] = {}
        self.category_cardinalities_: list[int] = []
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "CombinedSignalAugmentedPreprocessor":
        if len(frame) == 0:
            raise ValueError("cannot fit an empty combined-signal preprocessor")
        self.rate.fit(frame)
        self.encoder.fit(frame)
        temporal = self.encoder.transform(frame)
        self.temporal_means_ = {}
        self.temporal_scales_ = {}
        for name in TEMPORAL_TARGET_FEATURE_COLUMNS:
            values = temporal[name].to_numpy(dtype=np.float64, copy=False)
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(mean):
                raise ValueError(f"combined temporal feature {name} has non-finite mean")
            if not np.isfinite(scale) or scale < 1.0e-8:
                scale = 1.0
            self.temporal_means_[name] = mean
            self.temporal_scales_[name] = scale
        self.category_cardinalities_ = list(self.rate.category_cardinalities_)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        result = self.rate.transform(frame)
        temporal = self.encoder.transform(frame)
        array = np.empty(
            (len(frame), len(TEMPORAL_TARGET_FEATURE_COLUMNS)), dtype=np.float32
        )
        for position, name in enumerate(TEMPORAL_TARGET_FEATURE_COLUMNS):
            values = temporal[name].to_numpy(dtype=np.float64, copy=False)
            standardized = (
                values - self.temporal_means_[name]
            ) / self.temporal_scales_[name]
            array[:, position] = np.clip(
                standardized,
                -self.rate.base.clip_value,
                self.rate.base.clip_value,
            ).astype(np.float32)
        result["numerical"] = np.concatenate([result["numerical"], array], axis=1)
        return result

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    def feature_layout(self) -> TabularFeatureLayout:
        self._require_fitted()
        rate_layout = self.rate.feature_layout()
        numerical = tuple(self.numerical_columns)
        pitcher = (
            *rate_layout.pitcher_numerical_names,
            *(name for name in TEMPORAL_TARGET_FEATURE_COLUMNS if name.startswith("te_pitcher")),
        )
        batter = (
            *rate_layout.batter_numerical_names,
            *(name for name in TEMPORAL_TARGET_FEATURE_COLUMNS if name.startswith("te_batter")),
        )
        return TabularFeatureLayout(
            categorical_names=rate_layout.categorical_names,
            categorical_cardinalities=rate_layout.categorical_cardinalities,
            numerical_names=numerical,
            pitcher_numerical_names=pitcher,
            batter_numerical_names=batter,
            preprocessor_fingerprint=_fingerprint(self.to_state_dict()),
        )

    def to_state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_version": COMBINED_SIGNAL_STATE_VERSION,
            "feature_columns": list(COMBINED_SIGNAL_FEATURE_COLUMNS),
            "rate": self.rate.to_state_dict(),
            "encoder": self.encoder.to_state_dict(),
            "temporal_means": dict(self.temporal_means_),
            "temporal_scales": dict(self.temporal_scales_),
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CombinedSignalAugmentedPreprocessor":
        if not isinstance(payload, Mapping):
            raise TypeError("combined-signal state must be a mapping")
        if payload.get("state_version") != COMBINED_SIGNAL_STATE_VERSION:
            raise ValueError("unsupported combined-signal state version")
        if tuple(payload.get("feature_columns", ())) != COMBINED_SIGNAL_FEATURE_COLUMNS:
            raise ValueError("serialized combined feature order differs from code")
        rate = RateBasisAugmentedPreprocessor.from_state_dict(dict(payload["rate"]))
        encoder = PriorSeasonTargetEncoder.from_state_dict(dict(payload["encoder"]))
        means = payload.get("temporal_means")
        scales = payload.get("temporal_scales")
        if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
            raise ValueError("serialized combined temporal scaling state is missing")
        expected = set(TEMPORAL_TARGET_FEATURE_COLUMNS)
        if set(means) != expected or set(scales) != expected:
            raise ValueError("serialized combined temporal scaling keys differ from code")
        obj = cls(
            min_category_frequency=rate.base.min_category_frequency,
            clip_value=rate.base.clip_value,
        )
        obj.rate = rate
        obj.encoder = encoder
        obj.temporal_means_ = {
            name: float(means[name]) for name in TEMPORAL_TARGET_FEATURE_COLUMNS
        }
        obj.temporal_scales_ = {
            name: float(scales[name]) for name in TEMPORAL_TARGET_FEATURE_COLUMNS
        }
        if not all(np.isfinite(value) for value in obj.temporal_means_.values()):
            raise ValueError("serialized combined temporal means must be finite")
        if not all(
            np.isfinite(value) and value > 0.0
            for value in obj.temporal_scales_.values()
        ):
            raise ValueError("serialized combined temporal scales must be positive and finite")
        obj.category_cardinalities_ = list(rate.category_cardinalities_)
        obj.is_fitted_ = True
        return obj

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("CombinedSignalAugmentedPreprocessor must be fitted")


__all__ = [
    "COMBINED_SIGNAL_FEATURE_COLUMNS",
    "COMBINED_SIGNAL_STATE_VERSION",
    "CombinedSignalAugmentedPreprocessor",
]
