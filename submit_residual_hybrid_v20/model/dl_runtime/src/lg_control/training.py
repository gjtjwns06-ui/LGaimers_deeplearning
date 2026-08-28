"""Deterministic training loop for the tabular neural network."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import metric_dict


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4096
    eval_batch_size: int = 16384
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    brier_weight: float = 0.5
    patience: int = 4
    min_delta: float = 1e-6
    seed: int = 42
    num_workers: int = 0
    use_amp: bool = True
    gradient_clip_norm: float = 5.0
    verbose: bool = True
    scheduler_epochs: int | None = None

    def __post_init__(self) -> None:
        if self.scheduler_epochs is not None and self.scheduler_epochs < self.epochs:
            raise ValueError("scheduler_epochs must be at least epochs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    best_epoch: int
    best_metrics: dict[str, float]
    history: list[dict[str, float]]
    probabilities: np.ndarray


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_dataset(
    categorical: np.ndarray,
    numerical: np.ndarray,
    targets: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> TensorDataset:
    tensors: list[torch.Tensor] = [
        torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        torch.from_numpy(np.asarray(numerical, dtype=np.float32)),
    ]
    if targets is not None:
        tensors.append(torch.from_numpy(np.asarray(targets, dtype=np.float32).reshape(-1)))
    if weights is not None:
        tensors.append(torch.from_numpy(np.asarray(weights, dtype=np.float32).reshape(-1)))
    expected = len(categorical)
    if any(len(tensor) != expected for tensor in tensors):
        raise ValueError("all training arrays must have the same number of rows")
    return TensorDataset(*tensors)


def predict_arrays(
    model: nn.Module,
    categorical: np.ndarray,
    numerical: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> np.ndarray:
    loader = DataLoader(
        _tensor_dataset(categorical, numerical),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for cat_batch, num_batch in loader:
            cat_batch = cat_batch.to(device, non_blocking=device.type == "cuda")
            num_batch = num_batch.to(device, non_blocking=device.type == "cuda")
            pieces.append(torch.sigmoid(model(cat_batch, num_batch)).float().cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def train_model(
    model: nn.Module,
    train_categorical: np.ndarray,
    train_numerical: np.ndarray,
    train_targets: np.ndarray,
    val_categorical: np.ndarray,
    val_numerical: np.ndarray,
    val_targets: np.ndarray,
    *,
    config: TrainConfig,
    device: str | torch.device | None = None,
    sample_weights: np.ndarray | None = None,
    checkpoint_mode: str = "best",
) -> TrainResult:
    if config.epochs < 1 or config.batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if checkpoint_mode not in {"best", "last"}:
        raise ValueError("checkpoint_mode must be 'best' or 'last'")
    set_deterministic_seed(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)

    if sample_weights is None:
        sample_weights = np.ones(len(train_targets), dtype=np.float32)
    dataset = _tensor_dataset(
        train_categorical,
        train_numerical,
        train_targets,
        sample_weights,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=target_device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
        loss_sum = 0.0
        weight_sum = 0.0
        for cat_batch, num_batch, target_batch, weight_batch in loader:
            cat_batch = cat_batch.to(target_device, non_blocking=target_device.type == "cuda")
            num_batch = num_batch.to(target_device, non_blocking=target_device.type == "cuda")
            target_batch = target_batch.to(target_device, non_blocking=target_device.type == "cuda")
            weight_batch = weight_batch.to(target_device, non_blocking=target_device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(cat_batch, num_batch)
                bce = F.binary_cross_entropy_with_logits(logits, target_batch, reduction="none")
                probabilities = torch.sigmoid(logits)
                brier = torch.square(probabilities - target_batch)
                row_loss = bce + config.brier_weight * brier
                loss = torch.sum(row_loss * weight_batch) / torch.clamp(weight_batch.sum(), min=1e-8)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight_batch.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
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
            "train_loss": loss_sum / max(weight_sum, 1e-8),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(record)
        if config.verbose:
            print(
                f"epoch={epoch} train_loss={record['train_loss']:.8f} "
                f"val_brier={record['brier']:.8f} "
                f"val_skill={record['brier_skill']:.6f} "
                f"lr={record['learning_rate']:.3g}",
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
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    best_metrics = metric_dict(val_targets, best_probabilities)
    return TrainResult(
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        history=history,
        probabilities=best_probabilities,
    )


def train_fixed_epochs(
    model: nn.Module,
    categorical: np.ndarray,
    numerical: np.ndarray,
    targets: np.ndarray,
    *,
    config: TrainConfig,
    device: str | torch.device | None = None,
    sample_weights: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Fit a final model for a preselected epoch count without a holdout."""

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_deterministic_seed(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    if sample_weights is None:
        sample_weights = np.ones(len(targets), dtype=np.float32)
    dataset = _tensor_dataset(categorical, numerical, targets, sample_weights)
    generator = torch.Generator().manual_seed(config.seed)
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
        loss_sum = 0.0
        weight_sum = 0.0
        for cat_batch, num_batch, target_batch, weight_batch in loader:
            cat_batch = cat_batch.to(target_device, non_blocking=target_device.type == "cuda")
            num_batch = num_batch.to(target_device, non_blocking=target_device.type == "cuda")
            target_batch = target_batch.to(target_device, non_blocking=target_device.type == "cuda")
            weight_batch = weight_batch.to(target_device, non_blocking=target_device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(cat_batch, num_batch)
                bce = F.binary_cross_entropy_with_logits(logits, target_batch, reduction="none")
                brier = torch.square(torch.sigmoid(logits) - target_batch)
                row_loss = bce + config.brier_weight * brier
                loss = torch.sum(row_loss * weight_batch) / torch.clamp(
                    weight_batch.sum(), min=1e-8
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight_batch.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(weight_sum, 1e-8),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if config.verbose:
            print(
                f"final_epoch={epoch} train_loss={record['train_loss']:.8f} "
                f"lr={record['learning_rate']:.3g}",
                flush=True,
            )
    model.eval()
    return history
