"""Strict outer-validation evaluation for completed development checkpoints.

This module evaluates only the already-trained 2022 and 2023 rolling-origin
models.  It never trains, selects, tunes, calibrates, ensembles, or predicts or
scores the 2024 anchor.  Validation targets are removed before preprocessing
and are reintroduced only after row-aligned probabilities have been produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .constants import ID_COL, RECENT_RATE_COLUMNS, TARGET_COL
from .learning import DEVELOPMENT_VALIDATION_SEASONS, row_id_sha256
from .learning_artifacts import load_learning_checkpoint
from .metrics import brier_score, expected_calibration_error, log_loss
from .model import assert_preprocessor_compatible, validate_transformed_arrays
from .validation import SeasonFold, ValidationDesignError


EVALUATION_RECORD_VERSION = 1
REQUIRED_SLICE_FAMILIES = (
    "overall",
    "game_type",
    "entity_status",
    "recent_status",
    "pitcher_history",
    "batter_history",
)


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


@dataclass(frozen=True)
class EvaluationConfig:
    eval_batch_size: int = 16384
    num_workers: int = 0
    cpu_threads: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "eval_batch_size",
            _positive_int(self.eval_batch_size, "eval_batch_size"),
        )
        object.__setattr__(
            self, "cpu_threads", _positive_int(self.cpu_threads, "cpu_threads")
        )
        if isinstance(self.num_workers, (bool, np.bool_)):
            raise ValueError("num_workers must be a non-negative integer")
        try:
            workers = int(self.num_workers)
        except (TypeError, ValueError) as error:
            raise ValueError("num_workers must be a non-negative integer") from error
        if workers != self.num_workers or workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if workers != 0:
            raise ValueError(
                "num_workers is fixed at 0 for deterministic, memory-safe evaluation"
            )
        object.__setattr__(self, "num_workers", workers)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FoldEvaluation:
    fold_name: str
    validation_season: int
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    slice_metrics: pd.DataFrame
    evaluation_record: dict[str, Any]


@dataclass(frozen=True)
class DevelopmentEvaluation:
    fold_evaluations: tuple[FoldEvaluation, ...]
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    slice_metrics: pd.DataFrame
    aggregate_metrics: dict[str, Any]


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        metadata = json.dumps(
            {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big", signed=False))
        digest.update(metadata)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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
        raise ValueError("evaluation supports only CPU or CUDA devices")
    return result


def _assert_development_fold(fold: SeasonFold) -> None:
    if not isinstance(fold, SeasonFold):
        raise TypeError("fold must be a SeasonFold")
    if (
        fold.role != "rolling"
        or fold.validation_season not in DEVELOPMENT_VALIDATION_SEASONS
    ):
        raise ValidationDesignError(
            "evaluation accepts only the trained development folds 2022 and 2023; "
            "the 2024 anchor and diagnostic folds require a separate explicit stage"
        )


def _validate_target_probability(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("target and probabilities must be non-empty and aligned")
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("evaluation target must contain finite binary 0/1 values")
    if not np.isfinite(p).all():
        raise ValueError("predicted probabilities contain NaN or infinity")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("predicted probabilities must lie in [0, 1]")
    return y, p


def probability_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Return strict probability metrics and the fold-local constant baseline."""

    y, p = _validate_target_probability(target, probabilities)
    target_rate = float(y.mean())
    baseline_brier = float(target_rate * (1.0 - target_rate))
    brier = brier_score(y, p)
    if baseline_brier <= 0.0:
        skill: float | None = None
        score: float | None = None
    else:
        skill = float(1.0 - brier / baseline_brier)
        score = float(max(0.0, 100_000.0 * skill))
    return {
        "rows": int(len(y)),
        "target_rate": target_rate,
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std(ddof=0)),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
        "baseline_brier": baseline_brier,
        "brier": brier,
        "brier_skill": skill,
        "competition_score_analogue": score,
        "log_loss": log_loss(y, p),
        "ece_15": expected_calibration_error(y, p, n_bins=15),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "rows": 0,
        "target_rate": None,
        "prediction_mean": None,
        "prediction_std": None,
        "prediction_min": None,
        "prediction_max": None,
        "baseline_brier": None,
        "brier": None,
        "brier_skill": None,
        "competition_score_analogue": None,
        "log_loss": None,
        "ece_15": None,
    }


def _predict(
    model: torch.nn.Module,
    categorical: np.ndarray,
    numerical: np.ndarray,
    *,
    config: EvaluationConfig,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(
        torch.from_numpy(categorical),
        torch.from_numpy(numerical),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device).eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for categorical_batch, numerical_batch in loader:
            categorical_batch = categorical_batch.to(
                device, non_blocking=device.type == "cuda"
            )
            numerical_batch = numerical_batch.to(
                device, non_blocking=device.type == "cuda"
            )
            logits = model(categorical_batch, numerical_batch)
            pieces.append(torch.sigmoid(logits).float().cpu().numpy())
    if not pieces:
        raise RuntimeError("validation prediction loader produced no rows")
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _partition_masks(
    validation: pd.DataFrame,
    categorical: np.ndarray,
    *,
    categorical_names: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    names = tuple(categorical_names)
    pitcher_position = names.index("pitcher_id")
    batter_position = names.index("batter_id")
    # min_category_frequency=1 makes retained vocabulary membership exactly
    # equivalent to presence in outer train.  Using encoded indices also makes
    # the slice match the identities the loaded checkpoint can actually use.
    pitcher_seen = categorical[:, pitcher_position] >= 2
    batter_seen = categorical[:, batter_position] >= 2
    recent_missing = validation.loc[:, list(RECENT_RATE_COLUMNS)].isna().any(axis=1).to_numpy()
    masks: dict[str, dict[str, np.ndarray]] = {
        "overall": {"all": np.ones(len(validation), dtype=bool)},
        "game_type": {
            "F": validation["game_type"].eq("F").to_numpy(),
            "R": validation["game_type"].eq("R").to_numpy(),
        },
        "entity_status": {
            "both_seen": pitcher_seen & batter_seen,
            "pitcher_only": pitcher_seen & ~batter_seen,
            "batter_only": ~pitcher_seen & batter_seen,
            "neither_seen": ~pitcher_seen & ~batter_seen,
        },
        "recent_status": {
            "recent_missing": recent_missing,
            "recent_observed": ~recent_missing,
        },
    }
    for entity in ("pitcher", "batter"):
        counts = pd.to_numeric(
            validation[f"asof_{entity}_n"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        masks[f"{entity}_history"] = {
            "cold_start_0": counts == 0.0,
            "low_history_1_49": (counts > 0.0) & (counts < 50.0),
            "history_50_plus": counts >= 50.0,
            "history_missing": ~np.isfinite(counts),
        }

    for family, members in masks.items():
        coverage = np.zeros(len(validation), dtype=np.int8)
        for value, raw_mask in members.items():
            mask = np.asarray(raw_mask)
            if mask.dtype != np.bool_ or mask.shape != (len(validation),):
                raise ValueError(f"invalid mask {family}/{value}")
            coverage += mask.astype(np.int8)
        if family != "overall" and not np.equal(coverage, 1).all():
            raise ValueError(f"slice family {family} is not exhaustive and disjoint")
    return masks


def _slice_metrics(
    *,
    fold: SeasonFold,
    target: np.ndarray,
    probabilities: np.ndarray,
    masks: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in REQUIRED_SLICE_FAMILIES:
        if family not in masks:
            raise ValueError(f"required slice family is missing: {family}")
        for value, mask in masks[family].items():
            selected = np.asarray(mask, dtype=bool)
            metrics = (
                probability_metrics(target[selected], probabilities[selected])
                if selected.any()
                else _empty_metrics()
            )
            rows.append(
                {
                    "fold": fold.name,
                    "validation_season": fold.validation_season,
                    "slice": family,
                    "value": value,
                    "share": float(selected.mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _verify_checkpoint_identity(
    *,
    frame: pd.DataFrame,
    fold: SeasonFold,
    checkpoint_dir: Path,
    training_record: Mapping[str, Any],
) -> None:
    if training_record.get("outer_fold_name") != fold.name:
        raise ValueError("checkpoint belongs to a different outer fold")
    if training_record.get("outer_validation_season") != fold.validation_season:
        raise ValueError("checkpoint validation season differs from fold")
    if training_record.get("outer_train_seasons") != list(fold.train_seasons):
        raise ValueError("checkpoint outer-train seasons differ from fold")
    if training_record.get("outer_train_rows") != fold.train_row_count:
        raise ValueError("checkpoint outer-train row count differs from fold")
    if training_record.get("outer_train_row_id_sha256") != row_id_sha256(
        fold.train_row_ids
    ):
        raise ValueError("checkpoint outer-train row identity differs from fold")
    if training_record.get("source_row_id_sha256") != row_id_sha256(
        frame[ID_COL].to_numpy(dtype=object, copy=False)
    ):
        raise ValueError("checkpoint source row identity differs from input frame")
    if training_record.get("evaluation_status") != "not_evaluated":
        raise ValueError("checkpoint was not preserved in its pre-evaluation state")
    selection = training_record.get("selection_record")
    if not isinstance(selection, Mapping):
        raise ValueError("checkpoint selection record is missing")
    if selection.get("outer_fold_name") != fold.name:
        raise ValueError("checkpoint selection record belongs to another fold")
    if selection.get("selected_epoch") != training_record.get("selected_epoch"):
        raise ValueError("checkpoint refit epoch differs from its selection record")
    history = training_record.get("history")
    if not isinstance(history, list) or len(history) != training_record.get(
        "selected_epoch"
    ):
        raise ValueError("checkpoint refit history differs from selected epoch")
    if training_record.get("scheduler_horizon_epochs") != selection.get(
        "scheduler_horizon_epochs"
    ):
        raise ValueError("checkpoint scheduler horizon differs from selection")
    if not (checkpoint_dir / "manifest.json").is_file():
        raise FileNotFoundError("checkpoint manifest is missing")


def evaluate_outer_validation(
    frame: pd.DataFrame,
    fold: SeasonFold,
    checkpoint_dir: str | Path,
    *,
    config: EvaluationConfig | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> FoldEvaluation:
    """Predict and score exactly one development outer-validation partition."""

    _assert_development_fold(fold)
    if config is None:
        config = EvaluationConfig()
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be EvaluationConfig")
    target_device = _resolve_device(device)
    torch.set_num_threads(config.cpu_threads)
    checkpoint_path = Path(checkpoint_dir).resolve()
    loaded = load_learning_checkpoint(checkpoint_path, device="cpu")
    _verify_checkpoint_identity(
        frame=frame,
        fold=fold,
        checkpoint_dir=checkpoint_path,
        training_record=loaded.training_record,
    )
    assert_preprocessor_compatible(loaded.model.feature_layout, loaded.preprocessor)
    state_before = _model_state_sha256(loaded.model)
    preprocessor_fingerprint = loaded.model.feature_layout.preprocessor_fingerprint

    validation = fold.validation_frame(frame)
    validation_seasons = pd.to_numeric(
        validation["season"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.equal(validation_seasons, float(fold.validation_season)).all():
        raise ValueError("outer validation rows contain an unexpected season")
    target = pd.to_numeric(validation[TARGET_COL], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    # Make the target boundary explicit: the preprocessor and model receive no
    # validation label column.  Labels are used only after predictions exist.
    feature_frame = validation.drop(columns=[TARGET_COL])
    transformed = loaded.preprocessor.transform(feature_frame)
    if "target" in transformed:
        raise AssertionError("validation target unexpectedly entered preprocessing output")
    array_audit = validate_transformed_arrays(
        loaded.model.feature_layout,
        transformed,
        expected_row_ids=fold.validation_row_ids,
    )
    probabilities = _predict(
        loaded.model,
        transformed["categorical"],
        transformed["numerical"],
        config=config,
        device=target_device,
    )
    _, checked_probabilities = _validate_target_probability(target, probabilities)
    if len(checked_probabilities) != fold.validation_row_count:
        raise ValueError("prediction count differs from outer validation rows")
    if not validation[ID_COL].is_unique:
        raise ValueError("outer validation row IDs must be unique")

    state_after = _model_state_sha256(loaded.model)
    if state_after != state_before:
        raise RuntimeError("evaluation mutated checkpoint model state")
    assert_preprocessor_compatible(loaded.model.feature_layout, loaded.preprocessor)
    if loaded.model.feature_layout.preprocessor_fingerprint != preprocessor_fingerprint:
        raise RuntimeError("evaluation mutated preprocessing state")

    metrics = probability_metrics(target, probabilities)
    masks = _partition_masks(
        validation,
        transformed["categorical"],
        categorical_names=loaded.model.feature_layout.categorical_names,
    )
    slice_metrics = _slice_metrics(
        fold=fold,
        target=target,
        probabilities=probabilities,
        masks=masks,
    )
    predictions = pd.DataFrame(
        {
            ID_COL: validation[ID_COL].to_numpy(copy=True),
            "season": np.full(len(validation), fold.validation_season, dtype=np.int16),
            "y_true": target.astype(np.int8),
            "prediction": probabilities,
        }
    )
    if not np.array_equal(
        predictions[ID_COL].to_numpy(dtype=object), fold.validation_row_ids
    ):
        raise RuntimeError("prediction row order differs from validation partition")
    record: dict[str, Any] = {
        "record_version": EVALUATION_RECORD_VERSION,
        "phase": "development_outer_validation_evaluation",
        "fold": fold.name,
        "validation_season": fold.validation_season,
        "validation_rows": fold.validation_row_count,
        "validation_row_id_sha256": row_id_sha256(fold.validation_row_ids),
        "checkpoint_directory": str(checkpoint_path),
        "checkpoint_manifest_sha256": _sha256_file(
            checkpoint_path / "manifest.json"
        ),
        "checkpoint_model_state_sha256": loaded.manifest["files_sha256"][
            "model_state.npz"
        ],
        "loaded_model_state_sha256_before": state_before,
        "loaded_model_state_sha256_after": state_after,
        "preprocessor_fingerprint": preprocessor_fingerprint,
        "evaluation_config": config.to_dict(),
        "device": str(target_device),
        "target_removed_before_preprocessing": True,
        "target_consumed_by_model": False,
        "prediction_order_preserved": True,
        "model_state_unchanged": True,
        "preprocessor_state_unchanged": True,
        "array_audit": array_audit,
        "metrics": metrics,
        "required_slice_families": list(REQUIRED_SLICE_FAMILIES),
        "selection_or_tuning_performed": False,
        "calibration_or_ensemble_performed": False,
        "anchor_2024_evaluated": False,
        "inference_or_submission_performed": False,
    }
    if verbose:
        skill_text = (
            "null" if metrics["brier_skill"] is None else f"{metrics['brier_skill']:.6f}"
        )
        score_text = (
            "null"
            if metrics["competition_score_analogue"] is None
            else f"{metrics['competition_score_analogue']:.2f}"
        )
        print(
            f"[{fold.name}] rows={len(predictions)} "
            f"brier={metrics['brier']:.8f} "
            f"skill={skill_text} score_analogue={score_text}",
            flush=True,
        )
    loaded.model.to("cpu").eval()
    return FoldEvaluation(
        fold_name=fold.name,
        validation_season=fold.validation_season,
        predictions=predictions,
        metrics=metrics,
        slice_metrics=slice_metrics,
        evaluation_record=record,
    )


def _mean_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(np.mean(numeric))


def aggregate_development_evaluations(
    evaluations: Sequence[FoldEvaluation],
) -> DevelopmentEvaluation:
    """Create explicitly secondary macro and pooled development summaries."""

    ordered = tuple(sorted(evaluations, key=lambda item: item.validation_season))
    if tuple(item.validation_season for item in ordered) != DEVELOPMENT_VALIDATION_SEASONS:
        raise ValueError("aggregate requires exactly the 2022 and 2023 development folds")
    if len({item.fold_name for item in ordered}) != len(ordered):
        raise ValueError("aggregate contains duplicate folds")
    oof = pd.concat([item.predictions for item in ordered], ignore_index=True)
    if not oof[ID_COL].is_unique:
        raise ValueError("development OOF row IDs overlap across folds")
    pooled = probability_metrics(
        oof["y_true"].to_numpy(dtype=np.float64),
        oof["prediction"].to_numpy(dtype=np.float64),
    )
    metric_names = (
        "brier",
        "brier_skill",
        "competition_score_analogue",
        "log_loss",
        "ece_15",
    )
    macro = {
        name: _mean_or_none(item.metrics[name] for item in ordered)
        for name in metric_names
    }
    weighted_brier = float(
        sum(item.metrics["rows"] * item.metrics["brier"] for item in ordered)
        / sum(item.metrics["rows"] for item in ordered)
    )
    if not np.isclose(weighted_brier, pooled["brier"], rtol=0.0, atol=1e-15):
        raise RuntimeError("pooled Brier does not reconcile with row-weighted folds")
    fold_metric_rows = [
        {
            "fold": item.fold_name,
            "validation_season": item.validation_season,
            **item.metrics,
        }
        for item in ordered
    ]
    fold_metrics = pd.DataFrame(fold_metric_rows)
    slice_metrics = pd.concat(
        [item.slice_metrics for item in ordered], ignore_index=True
    )
    worst = max(ordered, key=lambda item: float(item.metrics["brier"]))
    aggregate = {
        "scope": "development outer validation only (2022 and 2023)",
        "primary_reporting": "fold metrics; aggregates are secondary",
        "folds": [item.fold_name for item in ordered],
        "rows": int(len(oof)),
        "macro_fold_mean": macro,
        "row_pooled": pooled,
        "row_weighted_fold_brier": weighted_brier,
        "worst_fold_by_brier": {
            "fold": worst.fold_name,
            "brier": worst.metrics["brier"],
        },
        "official_test_score": None,
        "anchor_2024_included": False,
    }
    return DevelopmentEvaluation(
        fold_evaluations=ordered,
        oof_predictions=oof,
        fold_metrics=fold_metrics,
        slice_metrics=slice_metrics,
        aggregate_metrics=aggregate,
    )


__all__ = [
    "EVALUATION_RECORD_VERSION",
    "REQUIRED_SLICE_FAMILIES",
    "EvaluationConfig",
    "FoldEvaluation",
    "DevelopmentEvaluation",
    "probability_metrics",
    "evaluate_outer_validation",
    "aggregate_development_evaluations",
]
