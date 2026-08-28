"""Exhaustive outcome-state model and fold-safe trainer.

The main prediction is the first component of a five-class simplex.  Reverse
and middle may co-occur, while success is exclusive with either failure flag.
Rows for which the next-occurrence reconstruction is unavailable retain their
observed binary success label and use a binary fallback loss; they never
receive an invented exhaustive label.
"""

from __future__ import annotations

import copy
import math
import operator
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .auxiliary_targets import recover_fold_train_auxiliary_targets
from .metrics import metric_dict
from .model import TabularFeatureLayout
from .regime_model import GatedCrossModelConfig, GatedCrossRegimeModel
from .training import TrainConfig, TrainResult


OUTCOME_NAMES = (
    "control_success",
    "reverse_only",
    "middle_only",
    "reverse_and_middle",
    "other_failure",
)
BALL_STRIKE_NAMES = ("ball", "strike")
FOLD_TRAIN_PARTITION = "fold_train"


class ExclusiveOutcomeLabelError(ValueError):
    """Raised when observed labels cannot form the declared outcome simplex."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: Any, name: str, *, upper_open: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    upper_valid = result < 1.0 if upper_open else result <= 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_valid:
        boundary = "[0, 1)" if upper_open else "[0, 1]"
        raise ValueError(f"{name} must be finite and in {boundary}")
    return result


@dataclass(frozen=True)
class ExclusiveOutcomeHeadConfig:
    """Capacity and optional auxiliary-head controls."""

    hidden_dim: int = 64
    dropout: float = 0.10
    use_ball_strike_auxiliary: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dim", _positive_int(self.hidden_dim, "hidden_dim"))
        object.__setattr__(
            self, "dropout", _probability(self.dropout, "dropout", upper_open=True)
        )
        if not isinstance(self.use_ball_strike_auxiliary, bool):
            raise TypeError("use_ball_strike_auxiliary must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "outcomes": list(OUTCOME_NAMES),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "use_ball_strike_auxiliary": self.use_ball_strike_auxiliary,
            "auxiliary_targets": list(BALL_STRIKE_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExclusiveOutcomeHeadConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("exclusive head config must be a mapping")
        expected = {
            "outcomes",
            "hidden_dim",
            "dropout",
            "use_ball_strike_auxiliary",
            "auxiliary_targets",
        }
        if set(payload) != expected:
            raise ValueError("exclusive head config keys differ from the contract")
        if payload["outcomes"] != list(OUTCOME_NAMES):
            raise ValueError("exclusive outcome order differs from the fixed contract")
        if payload["auxiliary_targets"] != list(BALL_STRIKE_NAMES):
            raise ValueError("ball/strike target order differs from the fixed contract")
        return cls(
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
            use_ball_strike_auxiliary=payload["use_ball_strike_auxiliary"],
        )


@dataclass(frozen=True)
class ExclusiveOutcomeLossConfig:
    """Weights beyond :class:`TrainConfig`'s success-Brier weight."""

    ball_strike_weight: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ball_strike_weight",
            _probability(self.ball_strike_weight, "ball_strike_weight"),
        )


@dataclass(frozen=True)
class ExclusiveOutcomeTargets:
    """Training labels with explicit observation masks."""

    binary_success: np.ndarray
    class_indices: np.ndarray
    outcome_mask: np.ndarray
    ball_strike_targets: np.ndarray
    ball_strike_mask: np.ndarray

    def __len__(self) -> int:
        return len(self.binary_success)


def _as_binary(values: Any, name: str, *, mask: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    checked = np.ones(len(array), dtype=bool) if mask is None else mask
    invalid = checked & (~np.isfinite(array) | ~np.isin(array, (0.0, 1.0)))
    if invalid.any():
        examples = np.flatnonzero(invalid)[:5].tolist()
        raise ExclusiveOutcomeLabelError(
            f"observed {name} labels must be finite binary values; positions={examples}"
        )
    return array


def compose_exclusive_outcome_targets(
    control_success: Any,
    reverse: Any,
    middle: Any,
    ball: Any,
    strike: Any,
    observed_mask: Any,
) -> ExclusiveOutcomeTargets:
    """Validate and compose labels without filling unobserved outcomes."""

    success = _as_binary(control_success, "control_success")
    observed = np.asarray(observed_mask, dtype=bool).reshape(-1)
    if len(observed) != len(success):
        raise ExclusiveOutcomeLabelError("observed_mask length differs from labels")
    arrays = {
        "reverse": _as_binary(reverse, "reverse", mask=observed),
        "middle": _as_binary(middle, "middle", mask=observed),
        "ball": _as_binary(ball, "ball", mask=observed),
        "strike": _as_binary(strike, "strike", mask=observed),
    }
    if any(len(value) != len(success) for value in arrays.values()):
        raise ExclusiveOutcomeLabelError("all label arrays must have the same length")

    failure_overlap = arrays["reverse"] + arrays["middle"]
    overlap = observed & (success == 1.0) & (failure_overlap > 0.0)
    if overlap.any():
        examples = np.flatnonzero(overlap)[:5].tolist()
        raise ExclusiveOutcomeLabelError(
            "control_success must be exclusive with reverse and middle; "
            f"positions={examples}"
        )

    classes = np.full(len(success), -1, dtype=np.int64)
    classes[observed & (success == 1.0)] = 0
    classes[
        observed & (success == 0.0) & (arrays["reverse"] == 1.0)
        & (arrays["middle"] == 0.0)
    ] = 1
    classes[
        observed & (success == 0.0) & (arrays["reverse"] == 0.0)
        & (arrays["middle"] == 1.0)
    ] = 2
    classes[
        observed & (success == 0.0) & (arrays["reverse"] == 1.0)
        & (arrays["middle"] == 1.0)
    ] = 3
    classes[
        observed & (success == 0.0) & (arrays["reverse"] == 0.0)
        & (arrays["middle"] == 0.0)
    ] = 4
    if np.any(observed & (classes < 0)):
        raise ExclusiveOutcomeLabelError("an observed row did not map to exactly one class")

    ball_strike = np.stack([arrays["ball"], arrays["strike"]], axis=1)
    ball_strike = np.where(observed[:, None], ball_strike, 0.0).astype(np.float32)
    ball_strike_mask = np.repeat(observed[:, None], 2, axis=1)
    return ExclusiveOutcomeTargets(
        binary_success=success.astype(np.float32),
        class_indices=classes,
        outcome_mask=observed,
        ball_strike_targets=ball_strike,
        ball_strike_mask=ball_strike_mask,
    )


def recover_fold_train_exclusive_targets(
    frame: pd.DataFrame,
    *,
    partition: str,
) -> ExclusiveOutcomeTargets:
    """Reconstruct outcome labels strictly inside an already split fold-train."""

    if partition != FOLD_TRAIN_PARTITION:
        raise ExclusiveOutcomeLabelError(
            "exclusive labels may be reconstructed only for partition='fold_train'"
        )
    recovered = recover_fold_train_auxiliary_targets(frame, partition=partition)
    return compose_exclusive_outcome_targets(
        frame["control_success"].to_numpy(),
        recovered["aux_reverse"].to_numpy(),
        recovered["aux_middle"].to_numpy(),
        recovered["aux_ball"].to_numpy(),
        recovered["aux_strike"].to_numpy(),
        recovered["auxiliary_valid_mask"].to_numpy(),
    )


class ExclusiveOutcomeGatedCrossModel(GatedCrossRegimeModel):
    """Gated-cross backbone with an exhaustive five-class main head."""

    architecture_name = "dcnv2_gated_experts_exclusive_outcome_v2"

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: GatedCrossModelConfig | None = None,
        head_config: ExclusiveOutcomeHeadConfig | None = None,
    ) -> None:
        if head_config is None:
            head_config = ExclusiveOutcomeHeadConfig()
        if not isinstance(head_config, ExclusiveOutcomeHeadConfig):
            raise TypeError("head_config must be an ExclusiveOutcomeHeadConfig")
        self.head_config = head_config
        super().__init__(feature_layout, config)
        width = self.config.interaction_dim
        self.outcome_norm = nn.LayerNorm(width)
        self.outcome_hidden = nn.Linear(width, head_config.hidden_dim)
        self.outcome_activation = nn.GELU()
        self.outcome_dropout = nn.Dropout(head_config.dropout)
        self.outcome_head = nn.Linear(head_config.hidden_dim, len(OUTCOME_NAMES))
        self.outcome_wide_head = nn.Linear(
            len(feature_layout.numerical_names), len(OUTCOME_NAMES), bias=False
        )
        if head_config.use_ball_strike_auxiliary:
            self.ball_strike_head: nn.Linear | None = nn.Linear(head_config.hidden_dim, 2)
        else:
            self.ball_strike_head = None
        self._reset_exclusive_parameters()

    def _reset_exclusive_parameters(self) -> None:
        nn.init.ones_(self.outcome_norm.weight)
        nn.init.zeros_(self.outcome_norm.bias)
        nn.init.xavier_uniform_(self.outcome_hidden.weight)
        nn.init.zeros_(self.outcome_hidden.bias)
        nn.init.normal_(self.outcome_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.outcome_head.bias)
        nn.init.normal_(self.outcome_wide_head.weight, mean=0.0, std=0.01)
        if self.ball_strike_head is not None:
            nn.init.normal_(self.ball_strike_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ball_strike_head.bias)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "outcome_head"):
            self._reset_exclusive_parameters()

    def forward_with_aux(self, categorical: Tensor, numerical: Tensor) -> dict[str, Tensor]:
        shared = self._shared_states(categorical, numerical)
        base = self._main_outputs(shared, numerical)
        hidden = self.outcome_hidden(self.outcome_norm(shared["fused_state"]))
        hidden = self.outcome_dropout(self.outcome_activation(hidden))
        outcome_logits = self.outcome_head(hidden) + self.outcome_wide_head(numerical)
        # Reuse the complete scalar backbone as the success-class evidence.
        outcome_logits = outcome_logits.clone()
        outcome_logits[:, 0] = outcome_logits[:, 0] + base["logits"]
        probabilities = torch.softmax(outcome_logits, dim=1)
        success_probability = probabilities[:, 0]
        success_log_odds = outcome_logits[:, 0] - torch.logsumexp(
            outcome_logits[:, 1:], dim=1
        )
        outputs = dict(base)
        outputs.update(
            {
                "logits": success_log_odds,
                "outcome_logits": outcome_logits,
                "outcome_probabilities": probabilities,
                "success_probability": success_probability,
            }
        )
        if self.ball_strike_head is not None:
            outputs["ball_strike_logits"] = self.ball_strike_head(hidden)
        return outputs

    def forward(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        # sigmoid(log-odds) is exactly softmax(outcome_logits)[:, 0].
        return self.forward_with_aux(categorical, numerical)["logits"]

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
            "exclusive_head": self.head_config.to_dict(),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> "ExclusiveOutcomeGatedCrossModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            GatedCrossModelConfig.from_dict(payload["model"]),
            ExclusiveOutcomeHeadConfig.from_dict(payload["exclusive_head"]),
        )


def build_exclusive_outcome_model(
    preprocessor: Any,
    *,
    config: GatedCrossModelConfig | None = None,
    head_config: ExclusiveOutcomeHeadConfig | None = None,
    initialization_seed: int = 42,
) -> ExclusiveOutcomeGatedCrossModel:
    if isinstance(initialization_seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    try:
        seed = operator.index(initialization_seed)
    except TypeError as error:
        raise TypeError("initialization_seed must be an integer") from error
    if seed < 0:
        raise ValueError("initialization_seed must be non-negative")
    layout = TabularFeatureLayout.from_preprocessor(preprocessor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ExclusiveOutcomeGatedCrossModel(layout, config, head_config)
    model.initialization_seed = seed
    return model


def predict_exclusive_success_arrays(
    model: ExclusiveOutcomeGatedCrossModel,
    categorical: np.ndarray,
    numerical: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> np.ndarray:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        torch.from_numpy(np.asarray(numerical, dtype=np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for cat_batch, num_batch in loader:
            outputs = model.forward_with_aux(cat_batch.to(device), num_batch.to(device))
            pieces.append(outputs["success_probability"].float().cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def _validate_training_targets(targets: ExclusiveOutcomeTargets, rows: int) -> None:
    if not isinstance(targets, ExclusiveOutcomeTargets):
        raise TypeError("targets must be ExclusiveOutcomeTargets")
    if len(targets) != rows:
        raise ValueError("target rows differ from feature rows")
    if targets.class_indices.shape != (rows,):
        raise ValueError("class_indices must have shape [rows]")
    if targets.outcome_mask.shape != (rows,):
        raise ValueError("outcome_mask must have shape [rows]")
    if targets.ball_strike_targets.shape != (rows, 2):
        raise ValueError("ball_strike_targets must have shape [rows, 2]")
    if targets.ball_strike_mask.shape != (rows, 2):
        raise ValueError("ball_strike_mask must have shape [rows, 2]")
    observed = np.asarray(targets.outcome_mask, dtype=bool)
    if np.any((targets.class_indices[observed] < 0) | (targets.class_indices[observed] > 4)):
        raise ValueError("observed class indices must be in [0, 4]")
    if np.any(targets.class_indices[~observed] != -1):
        raise ValueError("unobserved class indices must be -1")
    _as_binary(targets.binary_success, "binary_success")
    aux_mask = np.asarray(targets.ball_strike_mask, dtype=bool)
    invalid_aux = aux_mask & (
        ~np.isfinite(targets.ball_strike_targets)
        | ~np.isin(targets.ball_strike_targets, (0.0, 1.0))
    )
    if invalid_aux.any():
        raise ValueError("observed ball/strike targets must be finite binary values")


def train_exclusive_outcome_model(
    model: ExclusiveOutcomeGatedCrossModel,
    train_categorical: np.ndarray,
    train_numerical: np.ndarray,
    train_targets: ExclusiveOutcomeTargets,
    val_categorical: np.ndarray,
    val_numerical: np.ndarray,
    val_targets: np.ndarray,
    *,
    config: TrainConfig,
    loss_config: ExclusiveOutcomeLossConfig | None = None,
    device: str | torch.device | None = None,
    sample_weights: np.ndarray | None = None,
    checkpoint_mode: str = "best",
) -> TrainResult:
    """Fit on weighted CE/fallback BCE/Brier and early-stop on success Brier."""

    if not isinstance(model, ExclusiveOutcomeGatedCrossModel):
        raise TypeError("model must be an ExclusiveOutcomeGatedCrossModel")
    if config.epochs < 1 or config.batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if checkpoint_mode not in {"best", "last"}:
        raise ValueError("checkpoint_mode must be 'best' or 'last'")
    rows = len(train_categorical)
    _validate_training_targets(train_targets, rows)
    categorical = np.asarray(train_categorical, dtype=np.int64)
    numerical = np.asarray(train_numerical, dtype=np.float32)
    if len(numerical) != rows:
        raise ValueError("categorical and numerical rows differ")
    if sample_weights is None:
        sample_weights = np.ones(rows, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if len(weights) != rows or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample_weights must be finite, positive and row-aligned")
    loss_config = loss_config or ExclusiveOutcomeLossConfig()

    dataset = TensorDataset(
        torch.from_numpy(categorical),
        torch.from_numpy(numerical),
        torch.from_numpy(train_targets.binary_success.astype(np.float32)),
        torch.from_numpy(train_targets.class_indices.astype(np.int64)),
        torch.from_numpy(train_targets.outcome_mask.astype(bool)),
        torch.from_numpy(train_targets.ball_strike_targets.astype(np.float32)),
        torch.from_numpy(train_targets.ball_strike_mask.astype(bool)),
        torch.from_numpy(weights),
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=target_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.scheduler_epochs or config.epochs),
        eta_min=config.learning_rate * 0.05,
    )
    amp_enabled = bool(config.use_amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    best_probabilities = np.empty(0, dtype=np.float32)
    history: list[dict[str, float]] = []
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "brier": 0.0, "auxiliary": 0.0}
        total_weight = 0.0
        for batch in loader:
            cat, num, success, classes, outcome_mask, aux, aux_mask, weight = (
                value.to(target_device) for value in batch
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model.forward_with_aux(cat, num)
                probabilities = outputs["success_probability"]
                # CE is used exactly where the exhaustive label is observed;
                # otherwise the same row contributes binary BCE only.
                safe_classes = torch.clamp(classes, min=0)
                ce = F.cross_entropy(outputs["outcome_logits"], safe_classes, reduction="none")
                fallback = F.binary_cross_entropy(
                    probabilities.clamp(1e-7, 1.0 - 1e-7), success, reduction="none"
                )
                classification_row = torch.where(outcome_mask, ce, fallback)
                denominator = torch.clamp(weight.sum(), min=1e-8)
                classification_loss = torch.sum(classification_row * weight) / denominator
                brier_loss = torch.sum(torch.square(probabilities - success) * weight) / denominator

                auxiliary_loss = probabilities.new_zeros(())
                if "ball_strike_logits" in outputs:
                    aux_element = F.binary_cross_entropy_with_logits(
                        outputs["ball_strike_logits"], aux, reduction="none"
                    )
                    masked_weight = aux_mask.to(weight.dtype) * weight.unsqueeze(1)
                    auxiliary_loss = torch.sum(aux_element * masked_weight) / torch.clamp(
                        masked_weight.sum(), min=1.0
                    )
                loss = (
                    classification_loss
                    + config.brier_weight * brier_loss
                    + loss_config.ball_strike_weight * auxiliary_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight.sum().detach().cpu())
            total_weight += batch_weight
            totals["loss"] += float(loss.detach().cpu()) * batch_weight
            totals["classification"] += float(classification_loss.detach().cpu()) * batch_weight
            totals["brier"] += float(brier_loss.detach().cpu()) * batch_weight
            totals["auxiliary"] += float(auxiliary_loss.detach().cpu()) * batch_weight
        scheduler.step()

        val_probabilities = predict_exclusive_success_arrays(
            model,
            val_categorical,
            val_numerical,
            batch_size=config.eval_batch_size,
            device=target_device,
            num_workers=config.num_workers,
        )
        metrics = metric_dict(np.asarray(val_targets, dtype=np.float32), val_probabilities)
        record = {
            "epoch": float(epoch),
            "train_loss": totals["loss"] / total_weight,
            "train_classification_loss": totals["classification"] / total_weight,
            "train_success_brier": totals["brier"] / total_weight,
            "train_ball_strike_loss": totals["auxiliary"] / total_weight,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(record)
        if config.verbose:
            print(
                f"epoch={epoch} train_loss={record['train_loss']:.8f} "
                f"val_brier={record['brier']:.8f} "
                f"score={record['competition_score']:.2f}",
                flush=True,
            )
        if checkpoint_mode == "last":
            best_brier = metrics["brier"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_probabilities = val_probabilities.copy()
            continue
        if metrics["brier"] < best_brier - config.min_delta:
            best_brier = metrics["brier"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_probabilities = val_probabilities.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("exclusive-outcome training produced no checkpoint")
    model.load_state_dict(best_state)
    return TrainResult(
        best_epoch=best_epoch,
        best_metrics=metric_dict(np.asarray(val_targets, dtype=np.float32), best_probabilities),
        history=history,
        probabilities=best_probabilities,
    )


def train_exclusive_outcome_fixed_epochs(
    model: ExclusiveOutcomeGatedCrossModel,
    categorical: np.ndarray,
    numerical: np.ndarray,
    targets: ExclusiveOutcomeTargets,
    *,
    config: TrainConfig,
    loss_config: ExclusiveOutcomeLossConfig | None = None,
    device: str | torch.device | None = None,
    sample_weights: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Fit a final exclusive-outcome model without consulting a holdout.

    The caller must preselect ``config.epochs`` and the scheduler horizon on a
    development partition.  This function deliberately has no validation
    arguments, checkpoint selection, calibration, or early stopping path.
    """

    if not isinstance(model, ExclusiveOutcomeGatedCrossModel):
        raise TypeError("model must be an ExclusiveOutcomeGatedCrossModel")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    rows = len(categorical)
    _validate_training_targets(targets, rows)
    categorical = np.asarray(categorical, dtype=np.int64)
    numerical = np.asarray(numerical, dtype=np.float32)
    if len(numerical) != rows:
        raise ValueError("categorical and numerical rows differ")
    if sample_weights is None:
        sample_weights = np.ones(rows, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if len(weights) != rows or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample_weights must be finite, positive and row-aligned")
    loss_config = loss_config or ExclusiveOutcomeLossConfig()

    dataset = TensorDataset(
        torch.from_numpy(categorical),
        torch.from_numpy(numerical),
        torch.from_numpy(targets.binary_success.astype(np.float32)),
        torch.from_numpy(targets.class_indices.astype(np.int64)),
        torch.from_numpy(targets.outcome_mask.astype(bool)),
        torch.from_numpy(targets.ball_strike_targets.astype(np.float32)),
        torch.from_numpy(targets.ball_strike_mask.astype(bool)),
        torch.from_numpy(weights),
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=target_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.scheduler_epochs or config.epochs),
        eta_min=config.learning_rate * 0.05,
    )
    amp_enabled = bool(config.use_amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "brier": 0.0, "auxiliary": 0.0}
        total_weight = 0.0
        for batch in loader:
            cat, num, success, classes, outcome_mask, aux, aux_mask, weight = (
                value.to(target_device) for value in batch
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model.forward_with_aux(cat, num)
                probabilities = outputs["success_probability"]
                safe_classes = torch.clamp(classes, min=0)
                ce = F.cross_entropy(outputs["outcome_logits"], safe_classes, reduction="none")
                fallback = F.binary_cross_entropy(
                    probabilities.clamp(1e-7, 1.0 - 1e-7), success, reduction="none"
                )
                classification_row = torch.where(outcome_mask, ce, fallback)
                denominator = torch.clamp(weight.sum(), min=1e-8)
                classification_loss = torch.sum(classification_row * weight) / denominator
                brier_loss = torch.sum(torch.square(probabilities - success) * weight) / denominator
                auxiliary_loss = probabilities.new_zeros(())
                if "ball_strike_logits" in outputs:
                    aux_element = F.binary_cross_entropy_with_logits(
                        outputs["ball_strike_logits"], aux, reduction="none"
                    )
                    masked_weight = aux_mask.to(weight.dtype) * weight.unsqueeze(1)
                    auxiliary_loss = torch.sum(aux_element * masked_weight) / torch.clamp(
                        masked_weight.sum(), min=1.0
                    )
                loss = (
                    classification_loss
                    + config.brier_weight * brier_loss
                    + loss_config.ball_strike_weight * auxiliary_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight.sum().detach().cpu())
            total_weight += batch_weight
            totals["loss"] += float(loss.detach().cpu()) * batch_weight
            totals["classification"] += float(classification_loss.detach().cpu()) * batch_weight
            totals["brier"] += float(brier_loss.detach().cpu()) * batch_weight
            totals["auxiliary"] += float(auxiliary_loss.detach().cpu()) * batch_weight
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "train_loss": totals["loss"] / total_weight,
            "train_classification_loss": totals["classification"] / total_weight,
            "train_success_brier": totals["brier"] / total_weight,
            "train_ball_strike_loss": totals["auxiliary"] / total_weight,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if config.verbose:
            print(
                f"final_epoch={epoch} train_loss={record['train_loss']:.8f} "
                f"train_brier={record['train_success_brier']:.8f}",
                flush=True,
            )
    model.eval()
    return history


__all__ = (
    "BALL_STRIKE_NAMES",
    "OUTCOME_NAMES",
    "ExclusiveOutcomeGatedCrossModel",
    "ExclusiveOutcomeHeadConfig",
    "ExclusiveOutcomeLabelError",
    "ExclusiveOutcomeLossConfig",
    "ExclusiveOutcomeTargets",
    "build_exclusive_outcome_model",
    "compose_exclusive_outcome_targets",
    "predict_exclusive_success_arrays",
    "recover_fold_train_exclusive_targets",
    "train_exclusive_outcome_model",
    "train_exclusive_outcome_fixed_epochs",
)
