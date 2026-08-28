"""Residual feature processing for the low-confidence v17 region."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import json

import numpy as np
import pandas as pd

from .hybrid_residual import (
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    load_model_npz,
    predict_bounded_residual,
)


RAW_NUMERIC_COLUMNS = (
    "inning",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
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
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)

BASE_CATEGORICAL_COLUMNS = (
    "game_type",
    "top_bottom",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "model_direction",
    "recent_history_missing",
)

COUNT_CATEGORICAL_COLUMNS = (
    "count_state",
    "game_count_state",
    "count_direction",
    "count_out_state",
)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _tokens(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<MISSING>").astype(str)


class UncertainResidualPreprocessor:
    """Train-only numeric scaling and inspectable one-hot state processing."""

    def __init__(self, *, include_count_context: bool) -> None:
        self.include_count_context = bool(include_count_context)
        self.numeric_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_scales_: dict[str, float] = {}
        self.category_values_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []
        self.fitted_ = False

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        if self.include_count_context:
            return BASE_CATEGORICAL_COLUMNS + COUNT_CATEGORICAL_COLUMNS
        return BASE_CATEGORICAL_COLUMNS

    @staticmethod
    def _engineered(
        frame: pd.DataFrame,
        prediction_v17: np.ndarray,
        prediction_dl: np.ndarray,
    ) -> pd.DataFrame:
        required = set(RAW_NUMERIC_COLUMNS) | {
            "game_type",
            "top_bottom",
            "outs_before",
            "base_state",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"uncertain residual input is missing: {sorted(missing)}")
        v17 = np.asarray(prediction_v17, dtype=np.float64)
        dl = np.asarray(prediction_dl, dtype=np.float64)
        if v17.shape != (len(frame),) or dl.shape != (len(frame),):
            raise ValueError("uncertain residual predictions must be row-aligned")
        output = frame.copy()
        output["meta_v17"] = v17
        output["meta_dl"] = dl
        output["meta_v17_logit"] = _logit(v17)
        output["meta_dl_logit"] = _logit(dl)
        output["meta_difference"] = dl - v17
        output["meta_abs_difference"] = np.abs(dl - v17)
        output["meta_v17_uncertainty"] = np.abs(v17 - 0.5)
        output["meta_dl_uncertainty"] = np.abs(dl - 0.5)
        output["model_direction"] = np.where(dl >= v17, "up", "down")
        recent = [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
        output["recent_history_missing_count"] = output[recent].isna().sum(axis=1)
        output["recent_history_missing"] = np.where(
            output["recent_history_missing_count"].gt(0), "missing", "available"
        )
        output["recent_success_delta_1_5"] = output[recent[0]] - output[recent[2]]
        output["recent_success_curvature"] = (
            output[recent[0]] - 2.0 * output[recent[1]] + output[recent[2]]
        )
        balls = pd.to_numeric(output["balls_before"], errors="raise").astype(int)
        strikes = pd.to_numeric(output["strikes_before"], errors="raise").astype(int)
        outs = pd.to_numeric(output["outs_before"], errors="raise").astype(int)
        count = balls.astype(str) + "-" + strikes.astype(str)
        output["count_state"] = count
        output["game_count_state"] = output["game_type"].astype(str) + ":" + count
        output["count_direction"] = count + ":" + output["model_direction"].astype(str)
        output["count_out_state"] = count + ":" + outs.astype(str)
        return output

    @staticmethod
    def _numeric_columns() -> list[str]:
        engineered = [
            "meta_v17",
            "meta_dl",
            "meta_v17_logit",
            "meta_dl_logit",
            "meta_difference",
            "meta_abs_difference",
            "meta_v17_uncertainty",
            "meta_dl_uncertainty",
            "recent_history_missing_count",
            "recent_success_delta_1_5",
            "recent_success_curvature",
        ]
        return engineered + list(RAW_NUMERIC_COLUMNS)

    def fit(
        self,
        frame: pd.DataFrame,
        prediction_v17: np.ndarray,
        prediction_dl: np.ndarray,
    ) -> "UncertainResidualPreprocessor":
        prepared = self._engineered(frame, prediction_v17, prediction_dl)
        self.numeric_columns_ = self._numeric_columns()
        for column in self.numeric_columns_:
            values = pd.to_numeric(prepared[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            mean = float(filled.mean())
            scale = float(filled.std())
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            self.numeric_medians_[column] = median
            self.numeric_means_[column] = mean
            self.numeric_scales_[column] = scale
        for column in self.categorical_columns:
            self.category_values_[column] = sorted(set(_tokens(prepared[column])))
        names: list[str] = []
        for column in self.numeric_columns_:
            names.extend([column, f"{column}__missing"])
        for column in self.categorical_columns:
            names.extend(
                f"{column}=={value}" for value in self.category_values_[column]
            )
            names.append(f"{column}==<UNKNOWN>")
        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        prediction_v17: np.ndarray,
        prediction_dl: np.ndarray,
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("UncertainResidualPreprocessor must be fitted")
        prepared = self._engineered(frame, prediction_v17, prediction_dl)
        arrays: list[np.ndarray] = []
        for column in self.numeric_columns_:
            values = pd.to_numeric(prepared[column], errors="coerce").to_numpy(float)
            missing = ~np.isfinite(values)
            filled = np.where(missing, self.numeric_medians_[column], values)
            arrays.append(
                ((filled - self.numeric_means_[column]) / self.numeric_scales_[column])
                .astype(np.float32)
                .reshape(-1, 1)
            )
            arrays.append(missing.astype(np.float32).reshape(-1, 1))
        for column in self.categorical_columns:
            tokens = _tokens(prepared[column]).to_numpy(dtype=object)
            known = np.zeros(len(prepared), dtype=bool)
            for value in self.category_values_[column]:
                indicator = tokens == value
                known |= indicator
                arrays.append(indicator.astype(np.float32).reshape(-1, 1))
            arrays.append((~known).astype(np.float32).reshape(-1, 1))
        result = np.concatenate(arrays, axis=1)
        if result.shape[1] != len(self.feature_names_):
            raise RuntimeError("uncertain residual feature width differs")
        if not np.isfinite(result).all():
            raise RuntimeError("uncertain residual features are non-finite")
        return result

    def to_state_dict(self) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("UncertainResidualPreprocessor must be fitted")
        return {
            "include_count_context": self.include_count_context,
            "numeric_columns": self.numeric_columns_,
            "numeric_medians": self.numeric_medians_,
            "numeric_means": self.numeric_means_,
            "numeric_scales": self.numeric_scales_,
            "category_values": self.category_values_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "UncertainResidualPreprocessor":
        instance = cls(include_count_context=bool(payload["include_count_context"]))
        instance.numeric_columns_ = [str(value) for value in payload["numeric_columns"]]
        instance.numeric_medians_ = {
            str(key): float(value) for key, value in payload["numeric_medians"].items()
        }
        instance.numeric_means_ = {
            str(key): float(value) for key, value in payload["numeric_means"].items()
        }
        instance.numeric_scales_ = {
            str(key): float(value) for key, value in payload["numeric_scales"].items()
        }
        instance.category_values_ = {
            str(key): [str(value) for value in values]
            for key, values in payload["category_values"].items()
        }
        instance.feature_names_ = [str(value) for value in payload["feature_names"]]
        instance.fitted_ = True
        return instance


@dataclass(frozen=True)
class LoadedUncertainResidual:
    preprocessor: UncertainResidualPreprocessor
    models: tuple[BoundedLogitResidualNet, ...]
    threshold: float
    manifest: dict[str, Any]


def predict_uncertain_region(
    artifact: LoadedUncertainResidual,
    frame: pd.DataFrame,
    prediction_v17: np.ndarray,
    prediction_dl: np.ndarray,
    *,
    batch_size: int = 32768,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v17 = np.asarray(prediction_v17, dtype=np.float64)
    dl = np.asarray(prediction_dl, dtype=np.float64)
    mask = np.abs(v17 - 0.5) <= artifact.threshold
    output = v17.copy()
    correction = np.zeros(len(frame), dtype=np.float64)
    if mask.any():
        subset = frame.loc[mask].reset_index(drop=True)
        features = artifact.preprocessor.transform(subset, v17[mask], dl[mask])
        probabilities: list[np.ndarray] = []
        corrections: list[np.ndarray] = []
        for model in artifact.models:
            probability, delta = predict_bounded_residual(
                model,
                features,
                v17[mask],
                batch_size=batch_size,
                device=device,
            )
            probabilities.append(probability)
            corrections.append(delta)
        output[mask] = np.mean(probabilities, axis=0)
        correction[mask] = np.mean(corrections, axis=0)
    return output, correction, mask


def load_uncertain_residual(directory: Path, *, device: str = "cpu") -> LoadedUncertainResidual:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "v17_uncertain_residual_v1":
        raise ValueError("unexpected uncertain residual artifact kind")
    preprocessor = UncertainResidualPreprocessor.from_state_dict(
        json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
    )
    config = HybridResidualModelConfig(**manifest["model_config"])
    models: list[BoundedLogitResidualNet] = []
    for filename in manifest["model_files"]:
        model = BoundedLogitResidualNet(config)
        load_model_npz(model, directory / filename)
        model.to(device).eval()
        models.append(model)
    if not models:
        raise ValueError("uncertain residual artifact has no models")
    return LoadedUncertainResidual(
        preprocessor=preprocessor,
        models=tuple(models),
        threshold=float(manifest["threshold"]),
        manifest=manifest,
    )


__all__ = [
    "LoadedUncertainResidual",
    "UncertainResidualPreprocessor",
    "load_uncertain_residual",
    "predict_uncertain_region",
]
