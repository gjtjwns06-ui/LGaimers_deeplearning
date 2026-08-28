"""Bounded neural logit correction for a frozen row-independent base model."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import metric_dict


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
    "game_type",
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
)

META_REQUIRED_COLUMNS = (
    "prediction_v7",
    "prediction_rate_exclusive",
    "prediction_temporal_te",
    "prediction_equal",
    "v7_seed_std",
    "season_success_delta_k20",
)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _tokens(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<MISSING>").astype(str)


class HybridResidualPreprocessor:
    """Train-only scaling and low-cardinality one-hot encoding for the meta MLP."""

    def __init__(self) -> None:
        self.numeric_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_scales_: dict[str, float] = {}
        self.category_values_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []
        self.fitted_ = False

    @staticmethod
    def _engineered(frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(META_REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"hybrid meta frame is missing: {sorted(missing)}")
        output = frame.copy()
        probabilities = {
            name: pd.to_numeric(output[column], errors="raise").to_numpy(dtype=np.float64)
            for name, column in {
                "v7": "prediction_v7",
                "rate": "prediction_rate_exclusive",
                "temporal": "prediction_temporal_te",
                "equal": "prediction_equal",
            }.items()
        }
        for name, values in probabilities.items():
            output[f"meta_logit_{name}"] = _logit(values)
            output[f"meta_uncertainty_{name}"] = np.abs(values - 0.5)
        output["meta_prob_equal_minus_v7"] = probabilities["equal"] - probabilities["v7"]
        output["meta_abs_prob_equal_minus_v7"] = np.abs(
            output["meta_prob_equal_minus_v7"]
        )
        output["meta_logit_equal_minus_v7"] = (
            output["meta_logit_equal"] - output["meta_logit_v7"]
        )
        output["meta_abs_logit_equal_minus_v7"] = np.abs(
            output["meta_logit_equal_minus_v7"]
        )
        output["meta_prob_rate_minus_temporal"] = (
            probabilities["rate"] - probabilities["temporal"]
        )
        output["meta_abs_prob_rate_minus_temporal"] = np.abs(
            output["meta_prob_rate_minus_temporal"]
        )
        month = pd.to_numeric(output["game_month"], errors="raise").to_numpy(
            dtype=np.float64
        )
        output["meta_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        output["meta_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        return output

    @staticmethod
    def _meta_numeric_columns(frame: pd.DataFrame) -> list[str]:
        selected = [
            column
            for column in frame.columns
            if column.startswith("prediction_")
            or column.startswith("rate_exclusive_head_")
            or column.startswith("temporal_te_head_")
            or column.startswith("meta_")
        ]
        selected.extend(["v7_seed_std", "season_success_delta_k20"])
        return sorted(set(selected))

    def fit(self, frame: pd.DataFrame) -> "HybridResidualPreprocessor":
        prepared = self._engineered(frame)
        required_raw = set(RAW_NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS)
        missing = required_raw - set(prepared.columns)
        if missing:
            raise ValueError(f"raw residual features are missing: {sorted(missing)}")
        self.numeric_columns_ = self._meta_numeric_columns(prepared) + list(
            RAW_NUMERIC_COLUMNS
        )
        for column in self.numeric_columns_:
            values = pd.to_numeric(prepared[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            self.numeric_medians_[column] = median
            self.numeric_means_[column] = mean
            self.numeric_scales_[column] = scale
        for column in CATEGORICAL_COLUMNS:
            values = sorted(set(_tokens(prepared[column])))
            self.category_values_[column] = values
        feature_names: list[str] = []
        for column in self.numeric_columns_:
            feature_names.extend([column, f"{column}__missing"])
        for column in CATEGORICAL_COLUMNS:
            feature_names.extend(
                f"{column}=={value}" for value in self.category_values_[column]
            )
            feature_names.append(f"{column}==<UNKNOWN>")
        self.feature_names_ = feature_names
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("HybridResidualPreprocessor must be fitted")
        prepared = self._engineered(frame)
        columns: list[np.ndarray] = []
        for column in self.numeric_columns_:
            values = pd.to_numeric(prepared[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            missing = ~np.isfinite(values)
            filled = np.where(missing, self.numeric_medians_[column], values)
            columns.append(
                ((filled - self.numeric_means_[column]) / self.numeric_scales_[column])
                .astype(np.float32)
                .reshape(-1, 1)
            )
            columns.append(missing.astype(np.float32).reshape(-1, 1))
        for column in CATEGORICAL_COLUMNS:
            tokens = _tokens(prepared[column]).to_numpy(dtype=object)
            known = np.zeros(len(prepared), dtype=bool)
            for value in self.category_values_[column]:
                indicator = tokens == value
                known |= indicator
                columns.append(indicator.astype(np.float32).reshape(-1, 1))
            columns.append((~known).astype(np.float32).reshape(-1, 1))
        result = np.concatenate(columns, axis=1)
        if result.shape[1] != len(self.feature_names_):
            raise RuntimeError("residual feature width differs from fitted state")
        if not np.isfinite(result).all():
            raise RuntimeError("residual features contain non-finite values")
        return result

    def to_state_dict(self) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("HybridResidualPreprocessor must be fitted")
        return {
            "numeric_columns": self.numeric_columns_,
            "numeric_medians": self.numeric_medians_,
            "numeric_means": self.numeric_means_,
            "numeric_scales": self.numeric_scales_,
            "category_values": self.category_values_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "HybridResidualPreprocessor":
        instance = cls()
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
class HybridResidualModelConfig:
    input_dim: int
    correction_cap: float = 0.05
    hidden_dim: int = 32
    hidden_layers: int = 2
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1 or self.hidden_layers < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 < self.correction_cap <= 0.5:
            raise ValueError("correction_cap must be in (0, 0.5]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedLogitResidualNet(nn.Module):
    def __init__(self, config: HybridResidualModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.input_dim
        for _ in range(config.hidden_layers):
            layers.extend(
                [
                    nn.Linear(width, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            width = config.hidden_dim
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.body.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, features: torch.Tensor, base_logit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correction = self.config.correction_cap * torch.tanh(
            self.output(self.body(features)).squeeze(1)
        )
        probability = torch.sigmoid(base_logit + correction)
        return probability, correction


@dataclass(frozen=True)
class HybridResidualTrainConfig:
    epochs: int = 30
    scheduler_epochs: int = 30
    batch_size: int = 8192
    eval_batch_size: int = 32768
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    correction_l2: float = 0.02
    mean_correction_penalty: float = 0.05
    patience: int = 5
    min_delta: float = 1e-7
    seed: int = 42
    gradient_clip_norm: float = 2.0
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.scheduler_epochs < self.epochs:
            raise ValueError("scheduler_epochs must be at least epochs")
        if self.batch_size < 1 or self.eval_batch_size < 1 or self.patience < 1:
            raise ValueError("batch sizes and patience must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridResidualTrainResult:
    best_epoch: int
    best_metrics: dict[str, float]
    history: list[dict[str, float]]
    probabilities: np.ndarray
    corrections: np.ndarray


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _arrays(
    features: np.ndarray,
    base_probability: np.ndarray,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite two-dimensional array")
    base = np.asarray(base_probability, dtype=np.float64).reshape(-1)
    if len(base) != len(x) or not np.isfinite(base).all():
        raise ValueError("base_probability must be finite and row-aligned")
    base_logit = _logit(base).astype(np.float32)
    output: list[np.ndarray] = [x, base_logit]
    if target is not None:
        y = np.asarray(target, dtype=np.float32).reshape(-1)
        if len(y) != len(x) or not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("target must be binary and row-aligned")
        output.append(y)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(w) != len(x) or not np.isfinite(w).all() or np.any(w <= 0.0):
            raise ValueError("weights must be finite, positive and row-aligned")
        output.append(w)
    return tuple(output)


def predict_bounded_residual(
    model: BoundedLogitResidualNet,
    features: np.ndarray,
    base_probability: np.ndarray,
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    x, z = _arrays(features, base_probability)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(z))
    target_device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(target_device)
    model.eval()
    probabilities: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    with torch.inference_mode():
        for x_batch, z_batch in loader:
            probability, correction = model(
                x_batch.to(target_device), z_batch.to(target_device)
            )
            probabilities.append(probability.float().cpu().numpy())
            corrections.append(correction.float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(corrections)


def _train_epoch(
    model: BoundedLogitResidualNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: HybridResidualTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "brier": 0.0, "correction_l2": 0.0, "weight": 0.0}
    for x_batch, z_batch, y_batch, weight_batch in loader:
        x_batch = x_batch.to(device)
        z_batch = z_batch.to(device)
        y_batch = y_batch.to(device)
        weight_batch = weight_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        probability, correction = model(x_batch, z_batch)
        denominator = torch.clamp(weight_batch.sum(), min=1e-8)
        brier = torch.sum(torch.square(probability - y_batch) * weight_batch) / denominator
        correction_l2 = torch.sum(torch.square(correction) * weight_batch) / denominator
        mean_correction = torch.sum(correction * weight_batch) / denominator
        loss = (
            brier
            + config.correction_l2 * correction_l2
            + config.mean_correction_penalty * mean_correction.square()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        batch_weight = float(weight_batch.sum().detach().cpu())
        totals["loss"] += float(loss.detach().cpu()) * batch_weight
        totals["brier"] += float(brier.detach().cpu()) * batch_weight
        totals["correction_l2"] += float(correction_l2.detach().cpu()) * batch_weight
        totals["weight"] += batch_weight
    return {
        name: totals[name] / totals["weight"]
        for name in ("loss", "brier", "correction_l2")
    }


def _loader(
    features: np.ndarray,
    base_probability: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    if weights is None:
        weights = np.ones(len(target), dtype=np.float32)
    x, z, y, w = _arrays(features, base_probability, target, weights)
    dataset = TensorDataset(*(torch.from_numpy(values) for values in (x, z, y, w)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_bounded_residual(
    model: BoundedLogitResidualNet,
    train_features: np.ndarray,
    train_base_probability: np.ndarray,
    train_target: np.ndarray,
    val_features: np.ndarray,
    val_base_probability: np.ndarray,
    val_target: np.ndarray,
    *,
    config: HybridResidualTrainConfig,
    device: str | torch.device = "cpu",
    sample_weights: np.ndarray | None = None,
) -> HybridResidualTrainResult:
    _set_seed(config.seed)
    target_device = torch.device(device)
    model.to(target_device)
    loader = _loader(
        train_features,
        train_base_probability,
        train_target,
        sample_weights,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_epochs,
        eta_min=config.learning_rate * 0.05,
    )
    best_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_probabilities = np.empty(0, dtype=np.float32)
    best_corrections = np.empty(0, dtype=np.float32)
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(model, loader, optimizer, config, target_device)
        scheduler.step()
        probability, correction = predict_bounded_residual(
            model,
            val_features,
            val_base_probability,
            batch_size=config.eval_batch_size,
            device=target_device,
        )
        metrics = metric_dict(val_target, probability)
        record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_brier": train_metrics["brier"],
            "train_correction_l2": train_metrics["correction_l2"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "mean_correction": float(np.mean(correction)),
            "cap_hit_rate": float(
                np.mean(np.abs(correction) >= model.config.correction_cap * 0.95)
            ),
            **metrics,
        }
        history.append(record)
        if config.verbose:
            print(
                f"epoch={epoch} train_brier={record['train_brier']:.8f} "
                f"val_brier={record['brier']:.8f} "
                f"mean_delta={record['mean_correction']:.5f}",
                flush=True,
            )
        if metrics["brier"] < best_brier - config.min_delta:
            best_brier = metrics["brier"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_probabilities = probability.copy()
            best_corrections = correction.copy()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("bounded residual training produced no checkpoint")
    model.load_state_dict(best_state)
    return HybridResidualTrainResult(
        best_epoch=best_epoch,
        best_metrics=metric_dict(val_target, best_probabilities),
        history=history,
        probabilities=best_probabilities,
        corrections=best_corrections,
    )


def train_bounded_residual_fixed_epochs(
    model: BoundedLogitResidualNet,
    features: np.ndarray,
    base_probability: np.ndarray,
    target: np.ndarray,
    *,
    config: HybridResidualTrainConfig,
    device: str | torch.device = "cpu",
    sample_weights: np.ndarray | None = None,
) -> list[dict[str, float]]:
    _set_seed(config.seed)
    target_device = torch.device(device)
    model.to(target_device)
    loader = _loader(
        features,
        base_probability,
        target,
        sample_weights,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_epochs,
        eta_min=config.learning_rate * 0.05,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(model, loader, optimizer, config, target_device)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **train_metrics,
            }
        )
        if config.verbose:
            print(
                f"fixed_epoch={epoch} train_brier={train_metrics['brier']:.8f}",
                flush=True,
            )
    model.eval()
    return history


def save_model_npz(model: nn.Module, path: str | Any) -> None:
    state = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    np.savez_compressed(path, **state)


def load_model_npz(model: nn.Module, path: str | Any) -> None:
    with np.load(path, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    model.load_state_dict(state, strict=True)


__all__ = [
    "BoundedLogitResidualNet",
    "CATEGORICAL_COLUMNS",
    "HybridResidualModelConfig",
    "HybridResidualPreprocessor",
    "HybridResidualTrainConfig",
    "HybridResidualTrainResult",
    "META_REQUIRED_COLUMNS",
    "RAW_NUMERIC_COLUMNS",
    "load_model_npz",
    "predict_bounded_residual",
    "save_model_npz",
    "train_bounded_residual",
    "train_bounded_residual_fixed_epochs",
]
