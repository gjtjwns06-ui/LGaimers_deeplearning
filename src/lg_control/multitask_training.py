"""Masked multi-task optimisation for the regime-aware neural candidate."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import metric_dict
from .regime_model import AUXILIARY_TARGET_NAMES, MultiTaskGatedCrossRegimeModel
from .training import TrainConfig, TrainResult, predict_arrays


@dataclass
class MultiTaskLossConfig:
    auxiliary_weight: float = 0.15
    consistency_weight: float = 0.02

    def __post_init__(self) -> None:
        for name in ("auxiliary_weight", "consistency_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)


def _validate_arrays(
    categorical: np.ndarray,
    numerical: np.ndarray,
    target: np.ndarray,
    auxiliary_target: np.ndarray,
    auxiliary_mask: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    categorical = np.asarray(categorical, dtype=np.int64)
    numerical = np.asarray(numerical, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    auxiliary_target = np.asarray(auxiliary_target, dtype=np.float32)
    auxiliary_mask = np.asarray(auxiliary_mask, dtype=np.float32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    rows = len(target)
    if categorical.ndim != 2 or numerical.ndim != 2:
        raise ValueError("categorical and numerical inputs must be two-dimensional")
    if auxiliary_target.shape != (rows, len(AUXILIARY_TARGET_NAMES)):
        raise ValueError("auxiliary_target must have shape [rows, 4]")
    if auxiliary_mask.ndim == 1:
        auxiliary_mask = np.repeat(auxiliary_mask.reshape(-1, 1), 4, axis=1)
    if auxiliary_mask.shape != auxiliary_target.shape:
        raise ValueError("auxiliary_mask must have shape [rows] or [rows, 4]")
    if any(len(values) != rows for values in (categorical, numerical, sample_weights)):
        raise ValueError("all multi-task training arrays must have equal row counts")
    if not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("main target must be binary")
    observed_auxiliary = auxiliary_mask > 0.0
    if not np.isfinite(auxiliary_target[observed_auxiliary]).all() or not np.isin(
        auxiliary_target[observed_auxiliary], [0.0, 1.0]
    ).all():
        raise ValueError("observed auxiliary targets must be finite binary values")
    auxiliary_target = np.where(observed_auxiliary, auxiliary_target, 0.0).astype(
        np.float32
    )
    auxiliary_mask = observed_auxiliary.astype(np.float32)
    if not np.isfinite(numerical).all():
        raise ValueError("numerical inputs must be finite")
    if not np.isfinite(sample_weights).all() or not (sample_weights > 0.0).all():
        raise ValueError("sample weights must be finite and positive")
    return (
        categorical,
        numerical,
        target,
        auxiliary_target,
        auxiliary_mask,
        sample_weights,
    )


def train_multitask_model(
    model: MultiTaskGatedCrossRegimeModel,
    train_categorical: np.ndarray,
    train_numerical: np.ndarray,
    train_targets: np.ndarray,
    train_auxiliary_targets: np.ndarray,
    train_auxiliary_mask: np.ndarray,
    val_categorical: np.ndarray,
    val_numerical: np.ndarray,
    val_targets: np.ndarray,
    *,
    config: TrainConfig,
    loss_config: MultiTaskLossConfig | None = None,
    device: str | torch.device | None = None,
    sample_weights: np.ndarray | None = None,
    checkpoint_mode: str = "best",
) -> TrainResult:
    """Fit with train-only auxiliary labels and early-stop on main Brier only."""

    if not isinstance(model, MultiTaskGatedCrossRegimeModel):
        raise TypeError("model must be a MultiTaskGatedCrossRegimeModel")
    if config.epochs < 1 or config.batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if checkpoint_mode not in {"best", "last"}:
        raise ValueError("checkpoint_mode must be 'best' or 'last'")
    loss_config = loss_config or MultiTaskLossConfig()
    if sample_weights is None:
        sample_weights = np.ones(len(train_targets), dtype=np.float32)
    arrays = _validate_arrays(
        train_categorical,
        train_numerical,
        train_targets,
        train_auxiliary_targets,
        train_auxiliary_mask,
        sample_weights,
    )
    dataset = TensorDataset(*(torch.from_numpy(values) for values in arrays))

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
    best_state: dict[str, torch.Tensor] | None = None
    best_probabilities = np.empty(0, dtype=np.float32)
    history: list[dict[str, float]] = []
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "main": 0.0, "auxiliary": 0.0, "consistency": 0.0}
        weight_sum = 0.0
        for (
            cat_batch,
            num_batch,
            target_batch,
            auxiliary_batch,
            auxiliary_mask_batch,
            weight_batch,
        ) in loader:
            cat_batch = cat_batch.to(target_device)
            num_batch = num_batch.to(target_device)
            target_batch = target_batch.to(target_device)
            auxiliary_batch = auxiliary_batch.to(target_device)
            auxiliary_mask_batch = auxiliary_mask_batch.to(target_device)
            weight_batch = weight_batch.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model.forward_with_aux(cat_batch, num_batch)
                logits = outputs["logits"]
                auxiliary_logits = outputs["auxiliary_logits"]
                probabilities = torch.sigmoid(logits)
                main_row = F.binary_cross_entropy_with_logits(
                    logits, target_batch, reduction="none"
                ) + config.brier_weight * torch.square(probabilities - target_batch)
                main_loss = torch.sum(main_row * weight_batch) / torch.clamp(
                    weight_batch.sum(), min=1e-8
                )

                auxiliary_element = F.binary_cross_entropy_with_logits(
                    auxiliary_logits, auxiliary_batch, reduction="none"
                )
                masked_weights = auxiliary_mask_batch * weight_batch.unsqueeze(1)
                auxiliary_loss = torch.sum(auxiliary_element * masked_weights) / torch.clamp(
                    masked_weights.sum(), min=1.0
                )
                auxiliary_probabilities = torch.sigmoid(auxiliary_logits)
                reverse_conflict = torch.relu(
                    probabilities + auxiliary_probabilities[:, 0] - 1.0
                )
                middle_conflict = torch.relu(
                    probabilities + auxiliary_probabilities[:, 1] - 1.0
                )
                consistency_row = reverse_conflict.square() + middle_conflict.square()
                consistency_loss = torch.sum(consistency_row * weight_batch) / torch.clamp(
                    weight_batch.sum(), min=1e-8
                )
                loss = (
                    main_loss
                    + loss_config.auxiliary_weight * auxiliary_loss
                    + loss_config.consistency_weight * consistency_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight_batch.sum().detach().cpu())
            weight_sum += batch_weight
            totals["loss"] += float(loss.detach().cpu()) * batch_weight
            totals["main"] += float(main_loss.detach().cpu()) * batch_weight
            totals["auxiliary"] += float(auxiliary_loss.detach().cpu()) * batch_weight
            totals["consistency"] += float(consistency_loss.detach().cpu()) * batch_weight
        scheduler.step()

        val_probabilities = predict_arrays(
            model,
            val_categorical,
            val_numerical,
            batch_size=config.eval_batch_size,
            device=target_device,
            num_workers=config.num_workers,
        )
        metrics = metric_dict(val_targets, val_probabilities)
        record = {
            "epoch": float(epoch),
            "train_loss": totals["loss"] / weight_sum,
            "train_main_loss": totals["main"] / weight_sum,
            "train_auxiliary_loss": totals["auxiliary"] / weight_sum,
            "train_consistency_loss": totals["consistency"] / weight_sum,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(record)
        if config.verbose:
            print(
                f"epoch={epoch} train_loss={record['train_loss']:.8f} "
                f"aux={record['train_auxiliary_loss']:.8f} "
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
        raise RuntimeError("multi-task training produced no checkpoint")
    model.load_state_dict(best_state)
    return TrainResult(
        best_epoch=best_epoch,
        best_metrics=metric_dict(val_targets, best_probabilities),
        history=history,
        probabilities=best_probabilities,
    )


__all__ = ["MultiTaskLossConfig", "train_multitask_model"]
