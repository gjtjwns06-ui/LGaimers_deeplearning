"""F-only bounded residual driven by post-2022 reverse-state DL heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np
import pandas as pd

from .hybrid_residual import (
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    load_model_npz,
    predict_bounded_residual,
)


HEAD_COLUMNS = (
    "outcome_control_success",
    "outcome_reverse_only",
    "outcome_middle_only",
    "outcome_reverse_and_middle",
    "outcome_other_failure",
    "aux_ball",
    "aux_strike",
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

CATEGORICAL_COLUMNS = (
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "model_direction",
)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _tokens(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<MISSING>").astype(str)


def _head_array(heads: Mapping[str, np.ndarray], name: str, rows: int) -> np.ndarray:
    if name not in heads:
        raise ValueError(f"recent DL heads are missing {name}")
    values = np.asarray(heads[name], dtype=np.float64).reshape(-1)
    if len(values) != rows or not np.isfinite(values).all():
        raise ValueError(f"recent DL head {name} must be finite and row-aligned")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"recent DL head {name} is outside [0, 1]")
    return values


class FRegimeResidualPreprocessor:
    """Frozen scaling for a compact cause-aware residual network."""

    def __init__(self, *, include_reverse_heads: bool) -> None:
        self.include_reverse_heads = bool(include_reverse_heads)
        self.numeric_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_scales_: dict[str, float] = {}
        self.category_values_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []
        self.fitted_ = False

    def _engineered(
        self,
        frame: pd.DataFrame,
        base_probability: np.ndarray,
        heads: Mapping[str, np.ndarray],
    ) -> pd.DataFrame:
        required = set(RAW_NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS) - {"model_direction"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"F regime residual input is missing: {sorted(missing)}")
        base = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        if len(base) != len(frame) or not np.isfinite(base).all():
            raise ValueError("F regime base prediction must be finite and row-aligned")
        head = {name: _head_array(heads, name, len(frame)) for name in HEAD_COLUMNS}
        outcome = np.column_stack([head[name] for name in HEAD_COLUMNS[:5]])
        if not np.allclose(outcome.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError("recent DL exclusive outcome heads must sum to one")

        output = frame.copy()
        success = head["outcome_control_success"]
        reverse = head["outcome_reverse_only"] + head["outcome_reverse_and_middle"]
        middle = head["outcome_middle_only"] + head["outcome_reverse_and_middle"]
        output["meta_base"] = base
        output["meta_base_logit"] = _logit(base)
        output["meta_dl_success"] = success
        output["meta_dl_success_logit"] = _logit(success)
        output["meta_dl_minus_base"] = success - base
        output["meta_abs_dl_minus_base"] = np.abs(success - base)
        output["meta_base_uncertainty"] = np.abs(base - 0.5)
        output["model_direction"] = np.where(success >= base, "up", "down")
        if self.include_reverse_heads:
            for name, values in head.items():
                output[f"dl_{name}"] = values
            output["dl_reverse_total"] = reverse
            output["dl_middle_total"] = middle
            output["dl_reverse_minus_asof"] = (
                reverse - pd.to_numeric(output["asof_pitcher_reverse_rate"], errors="coerce")
            )
            output["dl_middle_minus_asof"] = (
                middle - pd.to_numeric(output["asof_pitcher_middle_rate"], errors="coerce")
            )
            output["dl_ball_minus_asof"] = (
                head["aux_ball"] - pd.to_numeric(output["asof_pitcher_ball_rate"], errors="coerce")
            )
            output["dl_strike_minus_asof"] = (
                head["aux_strike"] - pd.to_numeric(output["asof_pitcher_strike_rate"], errors="coerce")
            )
            output["dl_outcome_entropy"] = -np.sum(
                outcome * np.log(np.clip(outcome, 1e-7, 1.0)), axis=1
            )
            failure = np.clip(1.0 - success, 1e-6, 1.0)
            output["dl_reverse_share_of_failure"] = reverse / failure
            output["dl_middle_share_of_failure"] = middle / failure
        return output

    def _numeric_columns(self) -> list[str]:
        meta = [
            "meta_base",
            "meta_base_logit",
            "meta_dl_success",
            "meta_dl_success_logit",
            "meta_dl_minus_base",
            "meta_abs_dl_minus_base",
            "meta_base_uncertainty",
        ]
        if self.include_reverse_heads:
            meta.extend(
                [
                    *(f"dl_{name}" for name in HEAD_COLUMNS),
                    "dl_reverse_total",
                    "dl_middle_total",
                    "dl_reverse_minus_asof",
                    "dl_middle_minus_asof",
                    "dl_ball_minus_asof",
                    "dl_strike_minus_asof",
                    "dl_outcome_entropy",
                    "dl_reverse_share_of_failure",
                    "dl_middle_share_of_failure",
                ]
            )
        return meta + list(RAW_NUMERIC_COLUMNS)

    def fit(
        self,
        frame: pd.DataFrame,
        base_probability: np.ndarray,
        heads: Mapping[str, np.ndarray],
    ) -> "FRegimeResidualPreprocessor":
        prepared = self._engineered(frame, base_probability, heads)
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
        for column in CATEGORICAL_COLUMNS:
            self.category_values_[column] = sorted(set(_tokens(prepared[column])))
        names: list[str] = []
        for column in self.numeric_columns_:
            names.extend([column, f"{column}__missing"])
        for column in CATEGORICAL_COLUMNS:
            names.extend(f"{column}=={value}" for value in self.category_values_[column])
            names.append(f"{column}==<UNKNOWN>")
        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        base_probability: np.ndarray,
        heads: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("FRegimeResidualPreprocessor must be fitted")
        prepared = self._engineered(frame, base_probability, heads)
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
        for column in CATEGORICAL_COLUMNS:
            tokens = _tokens(prepared[column]).to_numpy(dtype=object)
            known = np.zeros(len(prepared), dtype=bool)
            for value in self.category_values_[column]:
                indicator = tokens == value
                known |= indicator
                arrays.append(indicator.astype(np.float32).reshape(-1, 1))
            arrays.append((~known).astype(np.float32).reshape(-1, 1))
        result = np.concatenate(arrays, axis=1)
        if result.shape[1] != len(self.feature_names_) or not np.isfinite(result).all():
            raise RuntimeError("F regime residual feature matrix is invalid")
        return result

    def to_state_dict(self) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("FRegimeResidualPreprocessor must be fitted")
        return {
            "include_reverse_heads": self.include_reverse_heads,
            "numeric_columns": self.numeric_columns_,
            "numeric_medians": self.numeric_medians_,
            "numeric_means": self.numeric_means_,
            "numeric_scales": self.numeric_scales_,
            "category_values": self.category_values_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "FRegimeResidualPreprocessor":
        instance = cls(include_reverse_heads=bool(payload["include_reverse_heads"]))
        instance.numeric_columns_ = [str(value) for value in payload["numeric_columns"]]
        instance.numeric_medians_ = {str(k): float(v) for k, v in payload["numeric_medians"].items()}
        instance.numeric_means_ = {str(k): float(v) for k, v in payload["numeric_means"].items()}
        instance.numeric_scales_ = {str(k): float(v) for k, v in payload["numeric_scales"].items()}
        instance.category_values_ = {
            str(k): [str(value) for value in values]
            for k, values in payload["category_values"].items()
        }
        instance.feature_names_ = [str(value) for value in payload["feature_names"]]
        instance.fitted_ = True
        return instance


@dataclass(frozen=True)
class LoadedFRegimeResidual:
    preprocessor: FRegimeResidualPreprocessor
    models: tuple[BoundedLogitResidualNet, ...]
    strength: float
    correction_center: float
    manifest: dict[str, Any]


def load_f_regime_residual(directory: Path, *, device: str = "cpu") -> LoadedFRegimeResidual:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "f_regime_reverse_residual_v1":
        raise ValueError("unexpected F regime residual artifact kind")
    preprocessor = FRegimeResidualPreprocessor.from_state_dict(
        json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
    )
    config = HybridResidualModelConfig(**manifest["model_config"])
    models: list[BoundedLogitResidualNet] = []
    for filename in manifest["model_files"]:
        model = BoundedLogitResidualNet(config)
        load_model_npz(model, directory / filename)
        models.append(model.to(device).eval())
    if not models:
        raise ValueError("F regime residual artifact has no models")
    strength = float(manifest["strength"])
    if not 0.0 < strength <= 1.0:
        raise ValueError("F regime residual strength is outside (0, 1]")
    correction_center = float(manifest.get("correction_center", 0.0))
    if not np.isfinite(correction_center):
        raise ValueError("F regime residual correction center is non-finite")
    return LoadedFRegimeResidual(
        preprocessor, tuple(models), strength, correction_center, manifest
    )


def predict_f_regime_residual(
    artifact: LoadedFRegimeResidual,
    frame: pd.DataFrame,
    base_probability: np.ndarray,
    heads: Mapping[str, np.ndarray],
    *,
    batch_size: int = 32768,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_probability, dtype=np.float64).reshape(-1)
    if len(base) != len(frame) or not np.isfinite(base).all():
        raise ValueError("F regime base probability must be finite and row-aligned")
    mask = frame["game_type"].astype(str).str.upper().eq("F").to_numpy()
    output = base.copy()
    correction = np.zeros(len(frame), dtype=np.float64)
    if mask.any():
        subset = frame.loc[mask].reset_index(drop=True)
        subset_heads = {name: _head_array(heads, name, len(frame))[mask] for name in HEAD_COLUMNS}
        features = artifact.preprocessor.transform(subset, base[mask], subset_heads)
        probabilities: list[np.ndarray] = []
        corrections: list[np.ndarray] = []
        for model in artifact.models:
            probability, delta = predict_bounded_residual(
                model, features, base[mask], batch_size=batch_size, device=device
            )
            probabilities.append(probability)
            corrections.append(delta)
        del candidate
        mean_correction = np.mean(corrections, axis=0)
        applied_correction = artifact.strength * (
            mean_correction - artifact.correction_center
        )
        output[mask] = 1.0 / (
            1.0 + np.exp(-(_logit(base[mask]) + applied_correction))
        )
        correction[mask] = applied_correction
    return np.clip(output, 1e-6, 1.0 - 1e-6), correction, mask


__all__ = [
    "FRegimeResidualPreprocessor",
    "HEAD_COLUMNS",
    "LoadedFRegimeResidual",
    "load_f_regime_residual",
    "predict_f_regime_residual",
]
