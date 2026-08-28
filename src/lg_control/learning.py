"""Leakage-safe learning stage for the context-gated tabular network.

This module intentionally stops before outer-fold evaluation.  A development
fold first selects an epoch on its train-contained inner stop season.  The
selection model is then discarded and a fresh preprocessor/model/optimizer are
fit on the complete outer-training partition for exactly that many epochs.

No public refit API accepts outer-validation arrays, labels, predictions, or
metrics.  Evaluation belongs to the next pipeline stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import (
    ContextGatedModelConfig,
    ContextGatedResidualModel,
    build_context_gated_model,
    validate_transformed_arrays,
)
from .preprocessing import DeepTabularPreprocessor
from .validation import (
    SeasonFold,
    ValidationDesignError,
    assert_selection_scope,
    fit_preprocessor_on_fold_train,
    fit_preprocessor_on_inner_train,
)


LEARNING_RECORD_VERSION = 1
DEVELOPMENT_VALIDATION_SEASONS = (2022, 2023)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not np.isfinite(numeric) or numeric != np.floor(numeric) or numeric < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _non_negative_seed(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [0, 2**32 - 1]")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer in [0, 2**32 - 1]") from error
    if (
        not np.isfinite(numeric)
        or numeric != np.floor(numeric)
        or numeric < 0
        or numeric > np.iinfo(np.uint32).max
    ):
        raise ValueError(f"{name} must be an integer in [0, 2**32 - 1]")
    return int(numeric)


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _non_negative_finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


@dataclass(frozen=True)
class OptimizationProcedure:
    """Frozen optimization procedure, not a hyperparameter search space."""

    max_selection_epochs: int = 8
    scheduler_horizon_epochs: int = 8
    batch_size: int = 4096
    eval_batch_size: int = 16384
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    brier_weight: float = 0.5
    patience: int = 3
    min_delta: float = 1e-6
    gradient_clip_norm: float = 5.0
    num_workers: int = 0
    use_amp: bool = False
    cpu_threads: int = 8

    def __post_init__(self) -> None:
        for name in (
            "max_selection_epochs",
            "scheduler_horizon_epochs",
            "batch_size",
            "eval_batch_size",
            "patience",
            "cpu_threads",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.scheduler_horizon_epochs < self.max_selection_epochs:
            raise ValueError(
                "scheduler_horizon_epochs must be >= max_selection_epochs"
            )
        if isinstance(self.num_workers, (bool, np.bool_)):
            raise ValueError("num_workers must be a non-negative integer")
        try:
            num_workers = int(self.num_workers)
        except (TypeError, ValueError) as error:
            raise ValueError("num_workers must be a non-negative integer") from error
        if num_workers != self.num_workers or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        object.__setattr__(self, "num_workers", num_workers)
        object.__setattr__(
            self, "learning_rate", _positive_finite(self.learning_rate, "learning_rate")
        )
        object.__setattr__(
            self, "weight_decay", _non_negative_finite(self.weight_decay, "weight_decay")
        )
        object.__setattr__(
            self, "brier_weight", _non_negative_finite(self.brier_weight, "brier_weight")
        )
        object.__setattr__(
            self, "min_delta", _non_negative_finite(self.min_delta, "min_delta")
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _positive_finite(self.gradient_clip_norm, "gradient_clip_norm"),
        )
        if not isinstance(self.use_amp, (bool, np.bool_)):
            raise ValueError("use_amp must be boolean")
        object.__setattr__(self, "use_amp", bool(self.use_amp))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizationProcedure":
        if not isinstance(payload, Mapping):
            raise TypeError("optimization procedure must be a mapping")
        return cls(**dict(payload))


@dataclass(frozen=True)
class SeedSpec:
    """Independent seeds for initialization, stochastic layers, and shuffling."""

    initialization_seed: int
    training_seed: int
    loader_seed: int

    def __post_init__(self) -> None:
        for name in ("initialization_seed", "training_seed", "loader_seed"):
            object.__setattr__(self, name, _non_negative_seed(getattr(self, name), name))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("seed specification must be a mapping")
        return cls(**dict(payload))


@dataclass(frozen=True)
class EpochSelectionRecord:
    """Metadata-only result of inner-stop epoch selection.

    It deliberately carries neither model weights nor row-level probabilities.
    """

    outer_fold_name: str
    outer_validation_season: int
    inner_partition_name: str
    inner_train_seasons: tuple[int, ...]
    inner_stop_season: int
    fit_rows: int
    stop_rows: int
    fit_row_id_sha256: str
    stop_row_id_sha256: str
    selected_epoch: int
    stopped_epoch: int
    best_inner_brier: float
    scheduler_horizon_epochs: int
    criterion: str
    architecture_procedure_sha256: str
    inner_preprocessor_fingerprint: str
    optimization: dict[str, Any]
    seeds: dict[str, int]
    history: tuple[dict[str, Any], ...]
    record_version: int = LEARNING_RECORD_VERSION

    @property
    def fold_name(self) -> str:
        return self.outer_fold_name

    def __post_init__(self) -> None:
        if self.record_version != LEARNING_RECORD_VERSION:
            raise ValueError(
                f"unsupported epoch-selection record version: {self.record_version}"
            )
        if not self.outer_fold_name or not self.inner_partition_name:
            raise ValueError("selection fold and inner partition names must be non-empty")
        if self.criterion != "brier":
            raise ValueError("the fixed epoch-selection criterion must be brier")
        if self.selected_epoch < 1 or self.selected_epoch > self.stopped_epoch:
            raise ValueError("selected_epoch must be within completed selection epochs")
        if self.stopped_epoch > int(self.optimization["max_selection_epochs"]):
            raise ValueError("stopped_epoch exceeds the selection procedure")
        if self.scheduler_horizon_epochs != int(
            self.optimization["scheduler_horizon_epochs"]
        ):
            raise ValueError("selection scheduler horizon differs from procedure")
        if not np.isfinite(self.best_inner_brier) or self.best_inner_brier < 0.0:
            raise ValueError("best_inner_brier must be non-negative and finite")
        if self.fit_rows < 1 or self.stop_rows < 1:
            raise ValueError("inner fit and stop partitions must be non-empty")
        if len(self.history) != self.stopped_epoch:
            raise ValueError("selection history length differs from stopped_epoch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_version": self.record_version,
            "outer_fold_name": self.outer_fold_name,
            "outer_validation_season": self.outer_validation_season,
            "inner_partition_name": self.inner_partition_name,
            "inner_train_seasons": list(self.inner_train_seasons),
            "inner_stop_season": self.inner_stop_season,
            "fit_rows": self.fit_rows,
            "stop_rows": self.stop_rows,
            "fit_row_id_sha256": self.fit_row_id_sha256,
            "stop_row_id_sha256": self.stop_row_id_sha256,
            "selected_epoch": self.selected_epoch,
            "stopped_epoch": self.stopped_epoch,
            "best_inner_brier": self.best_inner_brier,
            "scheduler_horizon_epochs": self.scheduler_horizon_epochs,
            "criterion": self.criterion,
            "architecture_procedure_sha256": self.architecture_procedure_sha256,
            "inner_preprocessor_fingerprint": self.inner_preprocessor_fingerprint,
            "optimization": self.optimization,
            "seeds": self.seeds,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochSelectionRecord":
        if not isinstance(payload, Mapping):
            raise TypeError("epoch-selection record must be a mapping")
        values = dict(payload)
        values["inner_train_seasons"] = tuple(values["inner_train_seasons"])
        values["history"] = tuple(dict(item) for item in values["history"])
        values["optimization"] = dict(values["optimization"])
        values["seeds"] = dict(values["seeds"])
        return cls(**values)


@dataclass(frozen=True)
class FittedOuterModel:
    """Fresh outer-training fit, ready to persist but not yet evaluate."""

    preprocessor: DeepTabularPreprocessor
    model: ContextGatedResidualModel
    training_record: dict[str, Any]


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _architecture_procedure_sha256(config: ContextGatedModelConfig) -> str:
    return _canonical_sha256(
        {
            "architecture": ContextGatedResidualModel.architecture_name,
            "model": config.to_dict(),
        }
    )


def row_id_sha256(values: Sequence[object] | np.ndarray) -> str:
    """Hash ordered row IDs with length framing to prevent concatenation collisions."""

    digest = hashlib.sha256()
    identifiers = np.asarray(values, dtype=object).reshape(-1)
    digest.update(len(identifiers).to_bytes(8, "big", signed=False))
    for value in identifiers:
        if pd.isna(value):
            raise ValueError("row IDs must not contain missing values")
        encoded = f"{type(value).__name__}:{value}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _assert_development_fold(fold: SeasonFold) -> None:
    if not isinstance(fold, SeasonFold):
        raise TypeError("fold must be a SeasonFold")
    if (
        fold.role != "rolling"
        or fold.validation_season not in DEVELOPMENT_VALIDATION_SEASONS
    ):
        raise ValidationDesignError(
            "learning stage accepts only rolling development folds 2022 and 2023; "
            "the 2024 anchor and diagnostic folds remain untouched"
        )
    if fold.inner_stop_split is None:
        raise ValidationDesignError(f"{fold.name}: an inner stop split is required")


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        result = device
    elif str(device).strip().lower() == "auto":
        result = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("learning supports only CPU or CUDA devices")
    return result


def _set_deterministic_runtime(seed: int, *, cpu_threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _validate_training_arrays(
    model: ContextGatedResidualModel,
    transformed: Mapping[str, Any],
    *,
    expected_row_ids: Sequence[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_transformed_arrays(
        model.feature_layout,
        transformed,
        expected_row_ids=expected_row_ids,
    )
    if "target" not in transformed:
        raise ValueError("labelled learning arrays must contain target")
    categorical = np.asarray(transformed["categorical"])
    numerical = np.asarray(transformed["numerical"])
    target = np.asarray(transformed["target"])
    if target.dtype != np.float32:
        raise TypeError(f"target array must be float32, got {target.dtype}")
    if target.ndim != 1 or len(target) != len(categorical):
        raise ValueError("target must be one-dimensional and match transformed rows")
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("target must contain only finite binary 0/1 values")
    return categorical, numerical, target


def _dataset(
    categorical: np.ndarray,
    numerical: np.ndarray,
    target: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(categorical),
        torch.from_numpy(numerical),
        torch.from_numpy(target),
    )


def _loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    loader_seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(loader_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _optimizer_and_scheduler(
    model: ContextGatedResidualModel,
    procedure: OptimizationProcedure,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=procedure.learning_rate,
        weight_decay=procedure.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=procedure.scheduler_horizon_epochs,
        eta_min=procedure.learning_rate * 0.05,
    )
    return optimizer, scheduler


def _train_epoch(
    model: ContextGatedResidualModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    procedure: OptimizationProcedure,
    amp_enabled: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    row_count = 0
    for categorical, numerical, target in loader:
        categorical = categorical.to(device, non_blocking=device.type == "cuda")
        numerical = numerical.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(categorical, numerical)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            brier = torch.square(torch.sigmoid(logits) - target)
            loss = (bce + procedure.brier_weight * brier).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), procedure.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        rows = int(len(target))
        loss_sum += float(loss.detach().cpu()) * rows
        row_count += rows
    if row_count == 0:
        raise RuntimeError("training loader produced no rows")
    return loss_sum / row_count


def _online_brier(
    model: ContextGatedResidualModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    """Compute only the authorized inner-stop aggregate; keep no predictions."""

    model.eval()
    squared_error_sum = 0.0
    row_count = 0
    with torch.inference_mode():
        for categorical, numerical, target in loader:
            categorical = categorical.to(device, non_blocking=device.type == "cuda")
            numerical = numerical.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            probabilities = torch.sigmoid(model(categorical, numerical)).float()
            errors = probabilities.double() - target.double()
            squared_error_sum += float(torch.square(errors).sum().cpu())
            row_count += int(len(target))
    if row_count == 0:
        raise RuntimeError("inner stop loader produced no rows")
    return squared_error_sum / row_count


def _environment_record(device: torch.device, *, amp_enabled: bool) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "amp_enabled": amp_enabled,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "python_executable": sys.executable,
    }


def select_epoch_on_inner_stop(
    frame: pd.DataFrame,
    fold: SeasonFold,
    *,
    model_config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    seeds: SeedSpec,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> EpochSelectionRecord:
    """Select an epoch using only inner train and inner stop rows."""

    _assert_development_fold(fold)
    if not isinstance(model_config, ContextGatedModelConfig):
        raise TypeError("model_config must be ContextGatedModelConfig")
    if not isinstance(optimization, OptimizationProcedure):
        raise TypeError("optimization must be OptimizationProcedure")
    if not isinstance(seeds, SeedSpec):
        raise TypeError("seeds must be SeedSpec")
    target_device = _resolve_device(device)
    inner = fold.inner_stop_split
    assert inner is not None
    assert_selection_scope(
        fold,
        np.concatenate([inner.train_positions, inner.validation_positions]),
    )

    fitted = fit_preprocessor_on_inner_train(
        lambda: DeepTabularPreprocessor(min_category_frequency=1),
        frame,
        fold,
    )
    preprocessor = fitted.preprocessor
    if not isinstance(preprocessor, DeepTabularPreprocessor):
        raise TypeError("inner preprocessing returned the wrong implementation")
    train_arrays = fitted.transform_train(frame)
    stop_arrays = fitted.transform_validation(frame)
    model = build_context_gated_model(
        preprocessor,
        config=model_config,
        initialization_seed=seeds.initialization_seed,
    )
    train_cat, train_num, train_target = _validate_training_arrays(
        model,
        train_arrays,
        expected_row_ids=inner.train_row_ids,
    )
    stop_cat, stop_num, stop_target = _validate_training_arrays(
        model,
        stop_arrays,
        expected_row_ids=inner.validation_row_ids,
    )

    _set_deterministic_runtime(seeds.training_seed, cpu_threads=optimization.cpu_threads)
    model.to(target_device)
    train_loader = _loader(
        _dataset(train_cat, train_num, train_target),
        batch_size=optimization.batch_size,
        shuffle=True,
        loader_seed=seeds.loader_seed,
        num_workers=optimization.num_workers,
        device=target_device,
    )
    stop_loader = _loader(
        _dataset(stop_cat, stop_num, stop_target),
        batch_size=optimization.eval_batch_size,
        shuffle=False,
        loader_seed=seeds.loader_seed,
        num_workers=optimization.num_workers,
        device=target_device,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, optimization)
    amp_enabled = bool(optimization.use_amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_brier = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, optimization.max_selection_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=target_device,
            procedure=optimization,
            amp_enabled=amp_enabled,
        )
        inner_brier = _online_brier(model, stop_loader, device=target_device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "inner_stop_brier": inner_brier,
            }
        )
        if verbose:
            print(
                f"[{fold.name}][inner] epoch={epoch} "
                f"train_loss={train_loss:.8f} inner_brier={inner_brier:.8f} "
                f"lr={learning_rate:.3g}",
                flush=True,
            )
        if inner_brier < best_brier - optimization.min_delta:
            best_brier = inner_brier
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= optimization.patience:
            break

    if best_epoch < 1 or not np.isfinite(best_brier):
        raise RuntimeError("inner selection did not produce a finite Brier score")
    # The inner model is intentionally not returned or persisted.  Only the
    # selected procedure crosses into the fresh outer refit phase.
    model.to("cpu")
    return EpochSelectionRecord(
        outer_fold_name=fold.name,
        outer_validation_season=fold.validation_season,
        inner_partition_name=inner.name,
        inner_train_seasons=inner.train_seasons,
        inner_stop_season=inner.validation_season,
        fit_rows=inner.train_row_count,
        stop_rows=inner.validation_row_count,
        fit_row_id_sha256=row_id_sha256(inner.train_row_ids),
        stop_row_id_sha256=row_id_sha256(inner.validation_row_ids),
        selected_epoch=best_epoch,
        stopped_epoch=len(history),
        best_inner_brier=best_brier,
        scheduler_horizon_epochs=optimization.scheduler_horizon_epochs,
        criterion="brier",
        architecture_procedure_sha256=_architecture_procedure_sha256(model_config),
        inner_preprocessor_fingerprint=model.feature_layout.preprocessor_fingerprint,
        optimization=optimization.to_dict(),
        seeds=seeds.to_dict(),
        history=tuple(history),
    )


def _assert_selection_record_matches(
    record: EpochSelectionRecord,
    fold: SeasonFold,
    model_config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
) -> None:
    if not isinstance(record, EpochSelectionRecord):
        raise TypeError("selection_record must be EpochSelectionRecord")
    inner = fold.inner_stop_split
    assert inner is not None
    if record.outer_fold_name != fold.name:
        raise ValueError("selection record belongs to a different outer fold")
    if record.outer_validation_season != fold.validation_season:
        raise ValueError("selection record outer season differs from fold")
    if record.inner_partition_name != inner.name:
        raise ValueError("selection record inner partition differs from fold")
    if record.fit_row_id_sha256 != row_id_sha256(inner.train_row_ids):
        raise ValueError("selection fit-row identity differs from fold")
    if record.stop_row_id_sha256 != row_id_sha256(inner.validation_row_ids):
        raise ValueError("selection stop-row identity differs from fold")
    if record.optimization != optimization.to_dict():
        raise ValueError("outer refit optimization differs from epoch-selection procedure")
    expected_architecture = _architecture_procedure_sha256(model_config)
    if record.architecture_procedure_sha256 != expected_architecture:
        raise ValueError("outer refit architecture differs from epoch selection")


def refit_outer_fixed_epochs(
    frame: pd.DataFrame,
    fold: SeasonFold,
    selection_record: EpochSelectionRecord,
    *,
    model_config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    seeds: SeedSpec,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> FittedOuterModel:
    """Freshly fit on outer training rows for the preselected epoch count.

    This signature deliberately has no validation argument.  The outer
    validation frame is neither materialized nor transformed in this stage.
    """

    _assert_development_fold(fold)
    if not isinstance(model_config, ContextGatedModelConfig):
        raise TypeError("model_config must be ContextGatedModelConfig")
    if not isinstance(optimization, OptimizationProcedure):
        raise TypeError("optimization must be OptimizationProcedure")
    if not isinstance(seeds, SeedSpec):
        raise TypeError("seeds must be SeedSpec")
    _assert_selection_record_matches(
        selection_record,
        fold,
        model_config,
        optimization,
    )
    target_device = _resolve_device(device)

    fitted = fit_preprocessor_on_fold_train(
        lambda: DeepTabularPreprocessor(min_category_frequency=1),
        frame,
        fold,
    )
    preprocessor = fitted.preprocessor
    if not isinstance(preprocessor, DeepTabularPreprocessor):
        raise TypeError("outer preprocessing returned the wrong implementation")
    train_arrays = fitted.transform_train(frame)
    model = build_context_gated_model(
        preprocessor,
        config=model_config,
        initialization_seed=seeds.initialization_seed,
    )
    train_cat, train_num, train_target = _validate_training_arrays(
        model,
        train_arrays,
        expected_row_ids=fold.train_row_ids,
    )

    _set_deterministic_runtime(seeds.training_seed, cpu_threads=optimization.cpu_threads)
    model.to(target_device)
    train_loader = _loader(
        _dataset(train_cat, train_num, train_target),
        batch_size=optimization.batch_size,
        shuffle=True,
        loader_seed=seeds.loader_seed,
        num_workers=optimization.num_workers,
        device=target_device,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, optimization)
    amp_enabled = bool(optimization.use_amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    for epoch in range(1, selection_record.selected_epoch + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=target_device,
            procedure=optimization,
            amp_enabled=amp_enabled,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
            }
        )
        if verbose:
            print(
                f"[{fold.name}][outer-train] epoch={epoch} "
                f"train_loss={train_loss:.8f} lr={learning_rate:.3g}",
                flush=True,
            )
        scheduler.step()

    model.to("cpu").eval()
    preprocessor_fingerprint = model.feature_layout.preprocessor_fingerprint
    training_record: dict[str, Any] = {
        "record_version": LEARNING_RECORD_VERSION,
        "phase": "development_outer_train_fixed_epoch_refit",
        "outer_fold_name": fold.name,
        "outer_validation_season": fold.validation_season,
        "outer_train_seasons": list(fold.train_seasons),
        "outer_train_rows": fold.train_row_count,
        "outer_train_row_id_sha256": row_id_sha256(fold.train_row_ids),
        "source_row_id_sha256": row_id_sha256(fold._source.row_ids),
        "selected_epoch": selection_record.selected_epoch,
        "scheduler_horizon_epochs": optimization.scheduler_horizon_epochs,
        "selection_record": selection_record.to_dict(),
        "architecture_procedure_sha256": _architecture_procedure_sha256(model_config),
        "model_config": model.get_config(),
        "parameter_report": model.parameter_report(),
        "preprocessor_fingerprint": preprocessor_fingerprint,
        "optimization": optimization.to_dict(),
        "seeds": seeds.to_dict(),
        "history": history,
        "environment": _environment_record(target_device, amp_enabled=amp_enabled),
        "outer_validation_access": {
            "transform_calls": 0,
            "predict_calls": 0,
            "metric_calls": 0,
        },
        "outer_validation_rows_accessed": False,
        "contains_outer_validation_predictions": False,
        "contains_outer_validation_metrics": False,
        "anchor_accessed": False,
        "diagnostic_accessed": False,
        "evaluation_status": "not_evaluated",
        "promotion_allowed": False,
    }
    return FittedOuterModel(
        preprocessor=preprocessor,
        model=model,
        training_record=training_record,
    )


__all__ = [
    "LEARNING_RECORD_VERSION",
    "DEVELOPMENT_VALIDATION_SEASONS",
    "OptimizationProcedure",
    "SeedSpec",
    "EpochSelectionRecord",
    "FittedOuterModel",
    "row_id_sha256",
    "select_epoch_on_inner_stop",
    "refit_outer_fixed_epochs",
]
