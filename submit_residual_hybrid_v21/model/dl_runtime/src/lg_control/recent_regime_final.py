"""Final train/load/inference path for the post-2022 regime ensemble."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .constants import ID_COL, TARGET_COL
from .exclusive_outcome import (
    OUTCOME_NAMES,
    ExclusiveOutcomeGatedCrossModel,
    ExclusiveOutcomeHeadConfig,
    ExclusiveOutcomeLossConfig,
    predict_exclusive_output_arrays,
    predict_exclusive_success_arrays,
    recover_fold_train_exclusive_targets,
    train_exclusive_outcome_fixed_epochs,
)
from .rate_basis_features import RateBasisAugmentedPreprocessor
from .recent_regime import (
    RecentRegimePolicy,
    recent_regime_audit,
    recent_regime_weights,
    select_recent_regime_training_rows,
)
from .regime_model import GatedCrossModelConfig
from .schema import validate_input_frame, validate_predictions
from .training import TrainConfig


ARTIFACT_VERSION = 1
REGIME_START_YEAR = 2023
FINAL_CUTOFF_YEAR = 2025
SELECTED_BASE_EPOCHS = 2
SELECTED_POLISH_EPOCHS = 4
SCHEDULER_EPOCHS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_policy() -> RecentRegimePolicy:
    return RecentRegimePolicy(
        regime_start_year=REGIME_START_YEAR,
        history_policy="stable_r_plus_recent",
        historical_r_decay=0.60,
        recent_decay=1.0,
    )


def selected_model_config() -> GatedCrossModelConfig:
    return GatedCrossModelConfig(
        embedding_dim_cap=16,
        interaction_dim=64,
        cross_layers=3,
        num_experts=4,
        low_rank=16,
        deep_hidden_dim=96,
        deep_layers=2,
        dropout=0.10,
    )


def _build_model(
    preprocessor: RateBasisAugmentedPreprocessor, *, seed: int
) -> ExclusiveOutcomeGatedCrossModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ExclusiveOutcomeGatedCrossModel(
            preprocessor.feature_layout(),
            selected_model_config(),
            ExclusiveOutcomeHeadConfig(
                hidden_dim=64, dropout=0.10, use_ball_strike_auxiliary=True
            ),
        )


def _initialize_exclusive_priors(
    model: ExclusiveOutcomeGatedCrossModel,
    targets: Any,
    weights: np.ndarray,
) -> dict[str, list[float] | int]:
    observed = np.asarray(targets.outcome_mask, dtype=bool)
    class_weight = np.asarray(weights, dtype=np.float64)[observed]
    class_indices = np.asarray(targets.class_indices, dtype=np.int64)[observed]
    class_counts = np.bincount(
        class_indices, weights=class_weight, minlength=len(OUTCOME_NAMES)
    ).astype(np.float64)
    class_rates = np.clip(class_counts / class_counts.sum(), 1e-6, 1.0)
    auxiliary_rates = np.array(
        [
            np.average(
                targets.ball_strike_targets[observed, position],
                weights=class_weight,
            )
            for position in range(2)
        ],
        dtype=np.float64,
    )
    auxiliary_rates = np.clip(auxiliary_rates, 1e-5, 1.0 - 1e-5)
    with torch.no_grad():
        model.output_bias.zero_()
        model.outcome_head.bias.copy_(
            torch.from_numpy(np.log(class_rates).astype(np.float32))
        )
        if model.ball_strike_head is not None:
            model.ball_strike_head.bias.copy_(
                torch.from_numpy(
                    np.log(auxiliary_rates / (1.0 - auxiliary_rates)).astype(np.float32)
                )
            )
    return {
        "valid_rows": int(observed.sum()),
        "weighted_class_rates": class_rates.tolist(),
        "weighted_ball_strike_rates": auxiliary_rates.tolist(),
    }


def _save_npz(model: torch.nn.Module, path: Path) -> None:
    np.savez_compressed(
        path,
        **{
            name: tensor.detach().cpu().numpy()
            for name, tensor in model.state_dict().items()
        },
    )


@dataclass(frozen=True)
class LoadedRecentRegimeArtifact:
    directory: str
    preprocessor: RateBasisAugmentedPreprocessor
    models: tuple[ExclusiveOutcomeGatedCrossModel, ...]
    manifest: dict[str, Any]
    training_record: dict[str, Any]


def train_final_recent_regime(
    train_input: str | Path,
    output_directory: str | Path,
    *,
    seeds: Sequence[int] = (42, 2026, 3407),
    device: str = "cpu",
    cpu_threads: int = 8,
    verbose: bool = True,
) -> LoadedRecentRegimeArtifact:
    """Fresh two-stage fit with no validation or checkpoint-selection input."""

    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds or any(seed < 0 for seed in unique_seeds):
        raise ValueError("seeds must contain unique non-negative integers")
    source = Path(train_input)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    tracked_names = [
        "preprocessor.json",
        "model_config.json",
        "training_record.json",
        "manifest.json",
    ]
    if any((destination / name).exists() for name in tracked_names):
        raise FileExistsError("final recent-regime artifact already exists")

    torch.set_num_threads(cpu_threads)
    data = pd.read_csv(source, low_memory=False)
    validate_input_frame(data, require_target=True, allow_target=True)
    policy = selected_policy()
    fit = select_recent_regime_training_rows(
        data, cutoff_year=FINAL_CUTOFF_YEAR, policy=policy
    )
    weights = recent_regime_weights(
        fit, cutoff_year=FINAL_CUTOFF_YEAR, policy=policy
    )
    audit = recent_regime_audit(fit, weights, policy=policy)
    if audit["obsolete_pre_regime_f_rows"] != 0:
        raise RuntimeError("final policy exposed obsolete F rows")
    recent = fit.loc[fit["season"] >= REGIME_START_YEAR].copy()
    if sorted(int(value) for value in recent["season"].unique()) != [2023, 2024]:
        raise RuntimeError("final recent polish must contain exactly 2023 and 2024")

    preprocessor = RateBasisAugmentedPreprocessor(min_category_frequency=2).fit(fit)
    fit_arrays = preprocessor.transform(fit)
    recent_arrays = preprocessor.transform(recent)
    fit_targets = recover_fold_train_exclusive_targets(fit, partition="fold_train")
    recent_targets = recover_fold_train_exclusive_targets(recent, partition="fold_train")

    base_config_template = dict(
        epochs=SELECTED_BASE_EPOCHS,
        scheduler_epochs=SCHEDULER_EPOCHS,
        batch_size=8192,
        eval_batch_size=32768,
        learning_rate=8e-4,
        weight_decay=2e-5,
        brier_weight=1.0,
        patience=SELECTED_BASE_EPOCHS,
        num_workers=0,
        use_amp=True,
        verbose=verbose,
    )
    polish_config_template = dict(
        epochs=SELECTED_POLISH_EPOCHS,
        scheduler_epochs=SCHEDULER_EPOCHS,
        batch_size=8192,
        eval_batch_size=32768,
        learning_rate=5e-5,
        weight_decay=2e-5,
        brier_weight=4.0,
        patience=SELECTED_POLISH_EPOCHS,
        num_workers=0,
        use_amp=True,
        verbose=verbose,
    )
    records: list[dict[str, Any]] = []
    model_files: list[str] = []
    models: list[ExclusiveOutcomeGatedCrossModel] = []
    for seed in unique_seeds:
        model = _build_model(preprocessor, seed=seed)
        prior = _initialize_exclusive_priors(model, fit_targets, weights)
        base_config = TrainConfig(seed=seed, **base_config_template)
        base_history = train_exclusive_outcome_fixed_epochs(
            model,
            fit_arrays["categorical"],
            fit_arrays["numerical"],
            fit_targets,
            config=base_config,
            loss_config=ExclusiveOutcomeLossConfig(ball_strike_weight=0.10),
            device=device,
            sample_weights=weights,
        )
        polish_config = TrainConfig(seed=seed, **polish_config_template)
        polish_history = train_exclusive_outcome_fixed_epochs(
            model,
            recent_arrays["categorical"],
            recent_arrays["numerical"],
            recent_targets,
            config=polish_config,
            loss_config=ExclusiveOutcomeLossConfig(ball_strike_weight=0.10),
            device=device,
            sample_weights=np.ones(len(recent), dtype=np.float32),
        )
        filename = f"model_seed_{seed}.npz"
        _save_npz(model, destination / filename)
        model_files.append(filename)
        models.append(model.cpu().eval())
        records.append(
            {
                "seed": seed,
                "prior": prior,
                "base_config": base_config.to_dict(),
                "base_history": base_history,
                "polish_config": polish_config.to_dict(),
                "polish_history": polish_history,
            }
        )

    (destination / "preprocessor.json").write_text(
        json.dumps(preprocessor.to_state_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_config = models[0].get_config()
    if any(model.get_config() != model_config for model in models[1:]):
        raise RuntimeError("final ensemble model configurations differ")
    (destination / "model_config.json").write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    training_record = {
        "artifact_version": ARTIFACT_VERSION,
        "stage": "final_recent_regime_fixed_epoch",
        "source_input": str(source),
        "source_sha256": _sha256(source),
        "policy_audit": audit,
        "fit_rows": int(len(fit)),
        "recent_polish_rows": int(len(recent)),
        "recent_polish_seasons": [2023, 2024],
        "fit_exclusive_valid_rows": int(fit_targets.outcome_mask.sum()),
        "polish_exclusive_valid_rows": int(recent_targets.outcome_mask.sum()),
        "selection_contract": {
            "inner_fit": "2023 months < 7",
            "inner_stop": "2023 months >= 7",
            "outer_confirmation": "2023 -> 2024 (previously observed anchor)",
            "base_epochs": SELECTED_BASE_EPOCHS,
            "polish_epochs": SELECTED_POLISH_EPOCHS,
            "scheduler_epochs": SCHEDULER_EPOCHS,
            "calibration": None,
        },
        "seeds": list(unique_seeds),
        "seed_records": records,
    }
    (destination / "training_record.json").write_text(
        json.dumps(training_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hash_names = ["preprocessor.json", "model_config.json", "training_record.json", *model_files]
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "architecture": "recent_regime_rate_basis_exclusive_ensemble_v1",
        "model_files": model_files,
        "files_sha256": {name: _sha256(destination / name) for name in hash_names},
        "row_independent_inference": True,
        "obsolete_pre_2023_f_training_rows": 0,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return LoadedRecentRegimeArtifact(
        directory=str(destination),
        preprocessor=preprocessor,
        models=tuple(models),
        manifest=manifest,
        training_record=training_record,
    )


def load_recent_regime_artifact(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_hashes: bool = True,
) -> LoadedRecentRegimeArtifact:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported recent-regime artifact version")
    if manifest.get("obsolete_pre_2023_f_training_rows") != 0:
        raise ValueError("artifact does not prove obsolete-F exclusion")
    if verify_hashes:
        for name, expected in manifest["files_sha256"].items():
            if _sha256(source / name) != expected:
                raise ValueError(f"artifact hash mismatch for {name}")
    preprocessor = RateBasisAugmentedPreprocessor.from_state_dict(
        json.loads((source / "preprocessor.json").read_text(encoding="utf-8"))
    )
    model_config = json.loads((source / "model_config.json").read_text(encoding="utf-8"))
    target_device = torch.device(device)
    models: list[ExclusiveOutcomeGatedCrossModel] = []
    for filename in manifest["model_files"]:
        model = ExclusiveOutcomeGatedCrossModel.from_config(model_config)
        with np.load(source / filename, allow_pickle=False) as archive:
            state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
        model.load_state_dict(state, strict=True)
        models.append(model.to(target_device).eval())
    training_record = json.loads(
        (source / "training_record.json").read_text(encoding="utf-8")
    )
    return LoadedRecentRegimeArtifact(
        directory=str(source),
        preprocessor=preprocessor,
        models=tuple(models),
        manifest=manifest,
        training_record=training_record,
    )


def predict_recent_regime(
    artifact: LoadedRecentRegimeArtifact,
    frame: pd.DataFrame,
    *,
    batch_size: int = 32768,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    validate_input_frame(frame, require_target=False, allow_target=False)
    arrays = artifact.preprocessor.transform(frame)
    predictions = [
        predict_exclusive_success_arrays(
            model,
            arrays["categorical"],
            arrays["numerical"],
            batch_size=batch_size,
            device=torch.device(device),
            num_workers=0,
        )
        for model in artifact.models
    ]
    probability = np.mean(np.stack(predictions, axis=0), axis=0)
    return validate_predictions(frame[ID_COL], probability)


def predict_recent_regime_heads(
    artifact: LoadedRecentRegimeArtifact,
    frame: pd.DataFrame,
    *,
    batch_size: int = 32768,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Predict the averaged five-state and ball/strike DL heads.

    These heads are row-local model outputs.  They expose the post-2022
    reverse mechanism without reconstructing, serializing, or looking up any
    privileged label at inference time.
    """

    validate_input_frame(frame, require_target=False, allow_target=False)
    arrays = artifact.preprocessor.transform(frame)
    per_model = [
        predict_exclusive_output_arrays(
            model,
            arrays["categorical"],
            arrays["numerical"],
            batch_size=batch_size,
            device=torch.device(device),
            num_workers=0,
        )
        for model in artifact.models
    ]
    if not per_model:
        raise ValueError("recent-regime artifact has no models")
    names = tuple(per_model[0])
    if any(tuple(output) != names for output in per_model[1:]):
        raise RuntimeError("recent-regime head contracts differ across seeds")
    averaged = {
        name: np.mean(
            np.stack([output[name] for output in per_model], axis=0), axis=0
        ).astype(np.float64)
        for name in names
    }
    success = validate_predictions(frame[ID_COL], averaged["outcome_control_success"])
    averaged["outcome_control_success"] = success
    outcome = np.column_stack(
        [averaged[f"outcome_{name}"] for name in OUTCOME_NAMES]
    )
    if not np.allclose(outcome.sum(axis=1), 1.0, atol=2e-6):
        raise RuntimeError("recent-regime outcome heads do not sum to one")
    return averaged


__all__ = (
    "LoadedRecentRegimeArtifact",
    "load_recent_regime_artifact",
    "predict_recent_regime",
    "predict_recent_regime_heads",
    "selected_model_config",
    "selected_policy",
    "train_final_recent_regime",
)
