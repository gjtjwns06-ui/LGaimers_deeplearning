"""Final single-model training and inference after a confirmed 2024 anchor.

This module is intentionally separate from the legacy ensemble pipeline.  It
accepts only the hash-sealed C2 neutral campaign, trains one fresh model for a
fixed epoch count on 2019--2024, and persists a strict NPZ artifact.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .anchor_confirmation import (
    ANCHOR_SEEDS,
    PRODUCTION_CAMPAIGN_LOCK,
    _metrics_from_prediction_table,
    _read_prediction_table,
    evaluate_confirmation_gates,
)
from .constants import ID_COL, TARGET_COL, TRAIN_COLUMNS
from .evaluation import EvaluationConfig, _model_state_sha256, _predict
from .learning import (
    OptimizationProcedure,
    _architecture_procedure_sha256,
    _dataset,
    _loader,
    _optimizer_and_scheduler,
    _set_deterministic_runtime,
    _train_epoch,
    _validate_training_arrays,
    row_id_sha256,
)
from .model import (
    ContextGatedModelConfig,
    ContextGatedResidualModel,
    assert_preprocessor_compatible,
    build_context_gated_model,
    validate_transformed_arrays,
)
from .preprocessing import DeepTabularPreprocessor
from .regime_tuning import (
    _canonical_sha256,
    _payload_with_hash,
    _read_json,
    _serialize_json,
    _sha256_file,
    _verify_embedded_hash,
)
from .schema import validate_input_frame, validate_predictions


SCHEMA_VERSION = 1
FINAL_CHECKPOINT_KIND = "final_context_c2_full_train_fixed_epoch_v1"
FINAL_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
EXPECTED_SELECTED_CANDIDATE_ID = "c2_neutral"
EXPECTED_SELECTED_EPOCH = 1
EXPECTED_ANCHOR_READY_SHA256 = (
    "e883d77c61ca1de4053d9670caf4dd4a8d70f7eecad73f609bee9f432066722b"
)
EXPECTED_ANCHOR_CONFIRMATION_SHA256 = (
    "1de24687a3702071971983ce6a331df08c47b29a083322f39b64ef89410d539a"
)
EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256 = (
    "25e26ceee5485a0a4d74ac12c8e10d0fd9078fadcc211a2c4ba9cb755d1c6283"
)
EXPECTED_ANCHOR_METRICS_SHA256 = (
    "bf970d98dccb3a620bf25e5439621845bb3a47705143fecac214495aa1607261"
)
EXPECTED_ANCHOR_METRICS_FILE_SHA256 = (
    "56a47480930a2193085c8d70eb25014cbe0ea6853a4b5efb16467cba9b6d83f7"
)
EXPECTED_ANCHOR_READY_FILE_SHA256 = (
    "a37581641cb0d1dcd274082a0dfdce2ac6cbe17ca235abb0ca5e1542e6830f29"
)
EXPECTED_FULL_SOURCE_SHA256 = (
    "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
)
EXPECTED_FULL_ROWS = 1_475_092
EXPECTED_FULL_ROW_ID_SHA256 = (
    "cfcbfdfab747fbecaa19857772a7a78b4a2a705568f830c287be2ded26266dfb"
)
EXPECTED_FULL_TARGET_RATE = 0.5237659752747625
EXPECTED_SEASON_COUNTS = {
    "2019": 237_413,
    "2020": 244_087,
    "2021": 247_088,
    "2022": 247_472,
    "2023": 245_525,
    "2024": 253_507,
}
EXPECTED_PREDICTIONS_COMMIT_SHA256 = (
    "8ee57e75699b24d825396665a6a4a66b38eb0fde1148ba115c93cb68191cdcfe"
)
EXPECTED_ANCHOR_PREDICTION_FILE_SHA256 = (
    "743ba2e07a625f40813cf5504b5fd15341ce31759b83d1970ecc8a8a4755ce92"
)
EXPECTED_PROMOTION_LOCK_SHA256 = (
    "920b45cf38f8a20eeefda42b83c1e3036411fff5b69f9638daf8eff7fd74ee6c"
)
FINAL_CODE_RELATIVE_FILES = (
    "scripts/run_anchor_confirmation.py",
    "scripts/train_final_context.py",
    "src/lg_control/__init__.py",
    "src/lg_control/anchor_confirmation.py",
    "src/lg_control/constants.py",
    "src/lg_control/evaluation.py",
    "src/lg_control/final_context_training.py",
    "src/lg_control/learning.py",
    "src/lg_control/learning_artifacts.py",
    "src/lg_control/metrics.py",
    "src/lg_control/model.py",
    "src/lg_control/preprocessing.py",
    "src/lg_control/regime_tuning.py",
    "src/lg_control/schema.py",
    "src/lg_control/validation.py",
)
FINAL_TRACKED_FILES = frozenset(
    {
        "anchor_provenance.json",
        "model_config.json",
        "model_state.npz",
        "preprocessor.json",
        "training_record.json",
    }
)


@dataclass(frozen=True)
class VerifiedFinalAnchor:
    anchor_directory: str
    anchor_ready_sha256: str
    anchor_ready_file_sha256: str
    confirmation_sha256: str
    confirmation_file_sha256: str
    metrics_sha256: str
    metrics_file_sha256: str
    predictions_commit_sha256: str
    prediction_file_sha256: str
    promotion_lock_sha256: str
    selected_candidate_id: str
    selected_epoch: int
    selected_model_config: dict[str, Any]
    optimization: dict[str, Any]
    seeds: dict[str, int]
    expected_full_source_sha256: str
    anchor_rows: int
    gates: dict[str, Any]

    def model_config(self) -> ContextGatedModelConfig:
        return ContextGatedModelConfig.from_dict(self.selected_model_config)

    def optimization_procedure(self) -> OptimizationProcedure:
        return OptimizationProcedure.from_dict(self.optimization)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedFinalContextArtifact:
    directory: str
    preprocessor: DeepTabularPreprocessor
    model: ContextGatedResidualModel
    training_record: dict[str, Any]
    anchor_provenance: dict[str, Any]
    manifest: dict[str, Any]


def _preprocessor_state_sha256(preprocessor: DeepTabularPreprocessor) -> str:
    return _canonical_sha256(preprocessor.to_state_dict())


def _expected_candidate_spec() -> dict[str, Any]:
    spec = PRODUCTION_CAMPAIGN_LOCK.selected_candidate_spec()
    if spec.get("candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID:
        raise RuntimeError("production campaign lock no longer selects C2 neutral")
    return spec


def _expected_optimization() -> OptimizationProcedure:
    return OptimizationProcedure(
        max_selection_epochs=8,
        scheduler_horizon_epochs=8,
        batch_size=4096,
        eval_batch_size=16384,
        learning_rate=0.001,
        weight_decay=0.00001,
        brier_weight=0.5,
        patience=3,
        min_delta=0.000001,
        gradient_clip_norm=5.0,
        num_workers=0,
        use_amp=False,
        cpu_threads=8,
    )


def _expected_anchor_gates() -> dict[str, Any]:
    checks = {
        "game_type_F_absolute_bias_at_most_0_05": {
            "operator": "<=",
            "passed": True,
            "threshold": 0.05,
            "value": 0.04779571822606421,
        },
        "game_type_F_brier_within_selected_2023_plus_0_005": {
            "operator": "<=",
            "passed": True,
            "threshold": 0.2656530925321217,
            "value": 0.2514938463709604,
        },
        "game_type_R_brier_within_selected_2023_plus_0_0005": {
            "operator": "<=",
            "passed": True,
            "threshold": 0.2491110413527776,
            "value": 0.24837452655155065,
        },
        "overall_brier_strictly_below_anchor_constant": {
            "operator": "<",
            "passed": True,
            "threshold": 0.24980692675678798,
            "value": 0.24874378967951363,
        },
        "overall_brier_within_selected_2023_plus_0_0025": {
            "operator": "<=",
            "passed": True,
            "threshold": 0.2523708402595799,
            "value": 0.24874378967951363,
        },
    }
    return {"passed": True, "checks": checks}


def _safe_project_file(relative: str) -> Path:
    candidate = Path(relative)
    project = Path(__file__).resolve().parents[2]
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe sealed source path: {relative!r}")
    resolved = (project / candidate).resolve()
    if project not in resolved.parents:
        raise ValueError(f"sealed source path escapes project: {relative!r}")
    return resolved


def _verify_sealed_dependency_sources(
    anchor_root: Path,
    promotion: Mapping[str, Any],
) -> None:
    anchor_inventory = promotion.get("anchor_code_files_sha256")
    if not isinstance(anchor_inventory, Mapping) or not anchor_inventory:
        raise ValueError("anchor code inventory is missing")
    for relative, expected in anchor_inventory.items():
        path = _safe_project_file(str(relative))
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"sealed anchor source differs: {relative}")

    regime_root = anchor_root.parent / "regime_tuning"
    plan_path = regime_root / "plan.json"
    selection_path = regime_root / "selection.json"
    if (
        not plan_path.is_file()
        or not selection_path.is_file()
        or _sha256_file(plan_path) != PRODUCTION_CAMPAIGN_LOCK.plan_file_sha256
        or _sha256_file(selection_path)
        != PRODUCTION_CAMPAIGN_LOCK.selection_file_sha256
    ):
        raise ValueError("sealed regime plan/selection files differ")
    plan = _read_json(plan_path)
    selection = _read_json(selection_path)
    if (
        _verify_embedded_hash(plan, "plan_sha256")
        != PRODUCTION_CAMPAIGN_LOCK.plan_sha256
        or _verify_embedded_hash(selection, "selection_sha256")
        != PRODUCTION_CAMPAIGN_LOCK.selection_sha256
    ):
        raise ValueError("sealed regime plan/selection identity differs")
    regime_inventory = plan.get("source_files_sha256")
    if not isinstance(regime_inventory, Mapping) or not regime_inventory:
        raise ValueError("sealed regime source inventory is missing")
    for relative, expected in regime_inventory.items():
        path = _safe_project_file(str(relative))
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"sealed regime source differs: {relative}")


def _verify_tracked_paths(root: Path, hashes: Any) -> None:
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("anchor_ready tracked file inventory is missing")
    for relative, expected in hashes.items():
        candidate = Path(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("anchor_ready contains an unsafe tracked path")
        path = (root / candidate).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"anchor_ready tracked file is missing: {relative}")
        if _sha256_file(path) != expected:
            raise ValueError(f"anchor_ready tracked file hash differs: {relative}")


def verify_confirmed_anchor(anchor_directory: str | Path) -> VerifiedFinalAnchor:
    """Verify the complete persisted confirmation without opening full train."""

    root = Path(anchor_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"anchor confirmation directory is missing: {root}")

    promotion_path = root / "promotion_lock.json"
    selection_path = root / "epoch_selection.json"
    ready_path = root / "anchor_ready.json"
    access_path = root / "access_started.json"
    prediction_commit_path = root / "predictions_committed.json"
    prediction_path = root / "anchor_predictions.csv"
    metrics_started_path = root / "metrics_started.json"
    metrics_path = root / "metrics.json"
    confirmation_path = root / "confirmation.json"
    checkpoint_record_path = root / "checkpoint" / "training_record.json"

    required = (
        promotion_path,
        selection_path,
        ready_path,
        access_path,
        prediction_commit_path,
        prediction_path,
        metrics_started_path,
        metrics_path,
        confirmation_path,
        checkpoint_record_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"confirmed anchor artifact is incomplete: {missing}")

    promotion = _read_json(promotion_path)
    selection = _read_json(selection_path)
    ready = _read_json(ready_path)
    access = _read_json(access_path)
    prediction_commit = _read_json(prediction_commit_path)
    metrics_started = _read_json(metrics_started_path)
    metrics = _read_json(metrics_path)
    confirmation = _read_json(confirmation_path)
    checkpoint_record = _read_json(checkpoint_record_path)

    promotion_sha = _verify_embedded_hash(promotion, "promotion_lock_sha256")
    _verify_embedded_hash(selection, "epoch_selection_sha256")
    ready_sha = _verify_embedded_hash(ready, "anchor_ready_sha256")
    access_sha = _verify_embedded_hash(access, "access_ledger_sha256")
    prediction_commit_sha = _verify_embedded_hash(
        prediction_commit, "predictions_commit_sha256"
    )
    metrics_started_sha = _verify_embedded_hash(
        metrics_started, "metrics_started_sha256"
    )
    metrics_sha = _verify_embedded_hash(metrics, "metrics_sha256")
    confirmation_sha = _verify_embedded_hash(confirmation, "confirmation_sha256")

    if confirmation_sha != EXPECTED_ANCHOR_CONFIRMATION_SHA256:
        raise ValueError("anchor terminal decision is not the sealed production decision")
    if ready_sha != EXPECTED_ANCHOR_READY_SHA256:
        raise ValueError("anchor_ready is not the sealed production checkpoint")
    if (
        _sha256_file(confirmation_path)
        != EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256
        or _sha256_file(metrics_path) != EXPECTED_ANCHOR_METRICS_FILE_SHA256
        or metrics_sha != EXPECTED_ANCHOR_METRICS_SHA256
        or _sha256_file(ready_path) != EXPECTED_ANCHOR_READY_FILE_SHA256
    ):
        raise ValueError("anchor file identity differs from the sealed production files")
    if confirmation.get("status") != "confirmed":
        raise ValueError("final training requires a confirmed anchor")
    if confirmation.get("previous_anchor_ready_sha256") != ready_sha:
        raise ValueError("anchor terminal does not reference anchor_ready")
    if (
        confirmation.get("previous_metrics_sha256") != metrics_sha
        or confirmation.get("previous_metrics_file_sha256") != _sha256_file(metrics_path)
    ):
        raise ValueError("anchor terminal does not reference persisted metrics")

    if promotion.get("expected_campaign_lock") != PRODUCTION_CAMPAIGN_LOCK.to_dict():
        raise ValueError("anchor promotion lock differs from the production campaign")
    _verify_sealed_dependency_sources(root, promotion)
    expected_spec = _expected_candidate_spec()
    if (
        promotion.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID
        or promotion.get("selected_model_config") != expected_spec["model_config"]
        or promotion.get("anchor_seed_policy") != ANCHOR_SEEDS.to_dict()
        or promotion.get("expected_full_source_sha256") != EXPECTED_FULL_SOURCE_SHA256
    ):
        raise ValueError("anchor promotion lock differs from the final training contract")
    optimization = OptimizationProcedure.from_dict(promotion.get("optimization", {}))
    if optimization.to_dict() != _expected_optimization().to_dict():
        raise ValueError("anchor optimization differs from the fixed final procedure")

    if (
        selection.get("promotion_lock_sha256") != promotion_sha
        or selection.get("record", {}).get("selected_epoch")
        != EXPECTED_SELECTED_EPOCH
        or selection.get("selection_model_discarded") is not True
    ):
        raise ValueError("anchor epoch selection differs from fixed epoch 1")
    if promotion_sha != EXPECTED_PROMOTION_LOCK_SHA256:
        raise ValueError("anchor promotion lock is not the sealed production lock")
    if (
        ready.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID
        or ready.get("selected_epoch") != EXPECTED_SELECTED_EPOCH
        or ready.get("promotion_lock_sha256") != promotion_sha
        or ready.get("epoch_selection_sha256")
        != selection.get("epoch_selection_sha256")
    ):
        raise ValueError("anchor_ready differs from C2 epoch 1")
    _verify_tracked_paths(root, ready.get("files_sha256"))
    if (
        checkpoint_record.get("selected_epoch") != EXPECTED_SELECTED_EPOCH
        or checkpoint_record.get("selection_model_reused") is not False
        or checkpoint_record.get("fresh_model_refit") is not True
        or checkpoint_record.get("anchor_2024_accessed") is not False
    ):
        raise ValueError("anchor checkpoint training record differs")

    if (
        access.get("previous_anchor_ready_sha256") != ready_sha
        or access.get("previous_anchor_ready_file_sha256") != _sha256_file(ready_path)
        or access.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID
        or access.get("expected_full_source_sha256") != EXPECTED_FULL_SOURCE_SHA256
    ):
        raise ValueError("anchor access ledger chain differs")
    if (
        prediction_commit.get("previous_access_ledger_sha256") != access_sha
        or prediction_commit.get("previous_access_file_sha256")
        != _sha256_file(access_path)
        or prediction_commit.get("selected_candidate_id")
        != EXPECTED_SELECTED_CANDIDATE_ID
        or prediction_commit.get("prediction_execution_count") != 1
        or prediction_commit.get("alternate_candidate_attempted") is not False
        or prediction_commit.get("prediction_file_sha256")
        != _sha256_file(prediction_path)
        or prediction_commit.get("source_preflight", {}).get("sha256")
        != EXPECTED_FULL_SOURCE_SHA256
        or prediction_commit_sha != EXPECTED_PREDICTIONS_COMMIT_SHA256
        or prediction_commit.get("prediction_file_sha256")
        != EXPECTED_ANCHOR_PREDICTION_FILE_SHA256
    ):
        raise ValueError("anchor prediction commit chain differs")
    prediction_table = _read_prediction_table(prediction_path)
    if prediction_commit.get("prediction_rows") != len(prediction_table):
        raise ValueError("anchor prediction row count differs")
    if prediction_commit.get("prediction_row_id_sha256") != row_id_sha256(
        prediction_table[ID_COL].to_numpy(dtype=object, copy=False)
    ):
        raise ValueError("anchor prediction row identity differs")

    if (
        metrics_started.get("previous_predictions_commit_sha256")
        != prediction_commit_sha
        or metrics_started.get("previous_predictions_commit_file_sha256")
        != _sha256_file(prediction_commit_path)
        or metrics_started.get("prediction_file_sha256")
        != _sha256_file(prediction_path)
        or metrics_started.get("decision_execution_budget") != 1
        or metrics_started.get("decision_reexecution_allowed_after_this_record")
        is not False
    ):
        raise ValueError("anchor metric intent chain differs")
    if (
        metrics.get("previous_metrics_started_sha256") != metrics_started_sha
        or metrics.get("previous_metrics_started_file_sha256")
        != _sha256_file(metrics_started_path)
        or metrics.get("previous_predictions_commit_sha256")
        != prediction_commit_sha
        or metrics.get("decision_metric_execution_count") != 1
        or metrics.get("audit_metric_recomputation_is_decision") is not False
    ):
        raise ValueError("anchor metrics chain differs")

    recomputed_metrics = _metrics_from_prediction_table(prediction_table)
    selected_metrics = promotion.get("regime_selection", {}).get(
        "selected_candidate_metrics"
    )
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("anchor promotion lock lacks selected development metrics")
    recomputed_gates = evaluate_confirmation_gates(recomputed_metrics, selected_metrics)
    if metrics.get("metrics") != recomputed_metrics or metrics.get("gates") != recomputed_gates:
        raise ValueError("anchor metrics/gates do not recompute from persisted predictions")
    if recomputed_gates.get("passed") is not True:
        raise ValueError("anchor did not pass every fixed confirmation gate")

    source_preflight = prediction_commit.get("source_preflight", {})
    source_rows = source_preflight.get("rows")
    if (
        source_rows != EXPECTED_FULL_ROWS
        or source_preflight.get("season_counts") != EXPECTED_SEASON_COUNTS
    ):
        raise ValueError("anchor source row count differs from the sealed full train")
    if recomputed_gates != _expected_anchor_gates():
        raise ValueError("anchor gates differ from the sealed production result")
    return VerifiedFinalAnchor(
        anchor_directory=str(root),
        anchor_ready_sha256=ready_sha,
        anchor_ready_file_sha256=EXPECTED_ANCHOR_READY_FILE_SHA256,
        confirmation_sha256=confirmation_sha,
        confirmation_file_sha256=EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256,
        metrics_sha256=metrics_sha,
        metrics_file_sha256=EXPECTED_ANCHOR_METRICS_FILE_SHA256,
        predictions_commit_sha256=prediction_commit_sha,
        prediction_file_sha256=_sha256_file(prediction_path),
        promotion_lock_sha256=promotion_sha,
        selected_candidate_id=EXPECTED_SELECTED_CANDIDATE_ID,
        selected_epoch=EXPECTED_SELECTED_EPOCH,
        selected_model_config=dict(expected_spec["model_config"]),
        optimization=optimization.to_dict(),
        seeds=ANCHOR_SEEDS.to_dict(),
        expected_full_source_sha256=EXPECTED_FULL_SOURCE_SHA256,
        anchor_rows=len(prediction_table),
        gates=dict(recomputed_gates),
    )


def _final_code_inventory() -> dict[str, str]:
    paths = tuple(_safe_project_file(relative) for relative in FINAL_CODE_RELATIVE_FILES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"final training code inventory is incomplete: {missing}")
    root = Path(__file__).resolve().parents[2]
    return {path.relative_to(root).as_posix(): _sha256_file(path) for path in paths}


def _load_full_training_frame(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"full training input is missing: {path}")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("full training input differs from the confirmed source")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("full training columns differ from the official schema")
    validate_input_frame(frame, require_target=True, allow_target=True)
    if frame.empty or not frame[ID_COL].is_unique:
        raise ValueError("full training row IDs must be nonempty and unique")
    seasons = pd.to_numeric(frame["season"], errors="raise").astype(int)
    if tuple(sorted(seasons.unique())) != FINAL_SEASONS:
        raise ValueError("final training input must contain exactly 2019..2024")
    target = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("final training target must be finite binary 0/1")
    counts = seasons.value_counts().sort_index()
    observed_counts = {str(int(key)): int(value) for key, value in counts.items()}
    observed_row_id_sha256 = row_id_sha256(
        frame[ID_COL].to_numpy(dtype=object, copy=False)
    )
    observed_target_rate = float(target.mean())
    if (
        len(frame) != EXPECTED_FULL_ROWS
        or observed_counts != EXPECTED_SEASON_COUNTS
        or observed_row_id_sha256 != EXPECTED_FULL_ROW_ID_SHA256
        or observed_target_rate != EXPECTED_FULL_TARGET_RATE
    ):
        raise ValueError("full training source semantics differ from the sealed contract")
    provenance = {
        "path": str(path),
        "sha256": observed_sha256,
        "rows": len(frame),
        "row_id_sha256": observed_row_id_sha256,
        "seasons": list(FINAL_SEASONS),
        "season_counts": observed_counts,
        "target_rate": observed_target_rate,
    }
    return frame, provenance


def _train_fresh_model(
    frame: pd.DataFrame,
    *,
    anchor: VerifiedFinalAnchor,
    source_provenance: Mapping[str, Any],
    verbose: bool,
) -> tuple[DeepTabularPreprocessor, ContextGatedResidualModel, dict[str, Any]]:
    if anchor.selected_epoch != EXPECTED_SELECTED_EPOCH:
        raise ValueError("final epoch must be the pre-access selected epoch 1")
    config = anchor.model_config()
    expected_config = ContextGatedModelConfig.from_dict(
        _expected_candidate_spec()["model_config"]
    )
    if config.to_dict() != expected_config.to_dict() or config.game_type_mode != "neutral":
        raise ValueError("final model must use neutral game_type mode")
    optimization = anchor.optimization_procedure()
    if (
        optimization.to_dict() != _expected_optimization().to_dict()
        or anchor.seeds != ANCHOR_SEEDS.to_dict()
        or anchor.expected_full_source_sha256 != EXPECTED_FULL_SOURCE_SHA256
        or int(source_provenance.get("rows", -1)) != len(frame)
        or source_provenance.get("row_id_sha256")
        != row_id_sha256(frame[ID_COL].to_numpy(dtype=object, copy=False))
    ):
        raise ValueError("final training procedure/provenance differs from the fixed contract")
    preprocessor = DeepTabularPreprocessor(
        min_category_frequency=1,
        clip_value=10.0,
    ).fit(frame)
    arrays = preprocessor.transform(frame)
    model = build_context_gated_model(
        preprocessor,
        config=config,
        initialization_seed=ANCHOR_SEEDS.initialization_seed,
    )
    categorical, numerical, target = _validate_training_arrays(
        model,
        arrays,
        expected_row_ids=frame[ID_COL].to_numpy(dtype=object, copy=False),
    )
    initial_state_sha256 = _model_state_sha256(model)
    device = torch.device("cpu")
    _set_deterministic_runtime(
        ANCHOR_SEEDS.training_seed,
        cpu_threads=optimization.cpu_threads,
    )
    model.to(device)
    train_loader = _loader(
        _dataset(categorical, numerical, target),
        batch_size=optimization.batch_size,
        shuffle=True,
        loader_seed=ANCHOR_SEEDS.loader_seed,
        num_workers=optimization.num_workers,
        device=device,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, optimization)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history: list[dict[str, Any]] = []
    for epoch in range(1, anchor.selected_epoch + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            procedure=optimization,
            amp_enabled=False,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
            }
        )
        if verbose:
            print(
                "[final_context][train] "
                f"epoch={epoch} train_loss={train_loss:.8f} lr={learning_rate:.3g}",
                flush=True,
            )
        scheduler.step()
    model.to("cpu").eval()
    final_state_sha256 = _model_state_sha256(model)
    record = _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "final_context_full_train",
            "checkpoint_kind": FINAL_CHECKPOINT_KIND,
            "selected_candidate_id": anchor.selected_candidate_id,
            "selected_epoch": anchor.selected_epoch,
            "train_seasons": list(FINAL_SEASONS),
            "train_rows": int(source_provenance["rows"]),
            "train_row_id_sha256": source_provenance["row_id_sha256"],
            "source_input_sha256": source_provenance["sha256"],
            "anchor_confirmation_sha256": anchor.confirmation_sha256,
            "anchor_ready_sha256": anchor.anchor_ready_sha256,
            "model_config": model.get_config(),
            "architecture_procedure_sha256": _architecture_procedure_sha256(config),
            "parameter_report": model.parameter_report(),
            "preprocessor_fingerprint": model.feature_layout.preprocessor_fingerprint,
            "preprocessor_state_sha256": _preprocessor_state_sha256(preprocessor),
            "optimization": optimization.to_dict(),
            "seeds": ANCHOR_SEEDS.to_dict(),
            "history": history,
            "initial_model_state_sha256": initial_state_sha256,
            "final_model_state_sha256": final_state_sha256,
            "model_count": 1,
            "fresh_model": True,
            "warm_start": False,
            "validation_performed": False,
            "early_stopping_performed": False,
            "recency_or_group_weighting": False,
            "trackman_used": False,
            "calibration_used": False,
            "ensemble_used": False,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "device": "cpu",
                "amp_enabled": False,
            },
        },
        "training_record_sha256",
    )
    return preprocessor, model, record


def _save_final_artifact(
    destination: Path,
    *,
    preprocessor: DeepTabularPreprocessor,
    model: ContextGatedResidualModel,
    training_record: Mapping[str, Any],
    anchor: VerifiedFinalAnchor,
    source_provenance: Mapping[str, Any],
) -> None:
    if destination.exists():
        raise FileExistsError(f"final artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert_preprocessor_compatible(model.feature_layout, preprocessor)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    ).resolve()
    try:
        preprocessor.save(staging / "preprocessor.json")
        (staging / "model_config.json").write_bytes(_serialize_json(model.get_config()))
        (staging / "training_record.json").write_bytes(
            _serialize_json(training_record)
        )
        anchor_provenance = _payload_with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "final_context_anchor_provenance",
                "verified_anchor": anchor.to_dict(),
                "source_provenance": dict(source_provenance),
            },
            "anchor_provenance_sha256",
        )
        (staging / "anchor_provenance.json").write_bytes(
            _serialize_json(anchor_provenance)
        )
        state_arrays = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in sorted(model.state_dict().items())
        }
        if any(not np.isfinite(array).all() for array in state_arrays.values()):
            raise ValueError("final model state contains a non-finite tensor")
        np.savez_compressed(staging / "model_state.npz", **state_arrays)
        manifest = _payload_with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_kind": FINAL_CHECKPOINT_KIND,
                "files_sha256": {
                    name: _sha256_file(staging / name)
                    for name in sorted(FINAL_TRACKED_FILES)
                },
                "model_state_schema": {
                    name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                    for name, array in state_arrays.items()
                },
                "model_state_sha256": _model_state_sha256(model),
                "preprocessor_fingerprint": model.feature_layout.preprocessor_fingerprint,
                "preprocessor_state_sha256": _preprocessor_state_sha256(preprocessor),
                "source_input_sha256": source_provenance["sha256"],
                "source_rows": source_provenance["rows"],
                "source_row_id_sha256": source_provenance["row_id_sha256"],
                "source_season_counts": source_provenance["season_counts"],
                "anchor_confirmation_sha256": anchor.confirmation_sha256,
                "anchor_confirmation_file_sha256": anchor.confirmation_file_sha256,
                "anchor_ready_sha256": anchor.anchor_ready_sha256,
                "anchor_metrics_sha256": anchor.metrics_sha256,
                "anchor_provenance_sha256": anchor_provenance[
                    "anchor_provenance_sha256"
                ],
                "training_record_sha256": training_record[
                    "training_record_sha256"
                ],
                "selected_candidate_id": anchor.selected_candidate_id,
                "selected_epoch": anchor.selected_epoch,
                "model_count": 1,
                "calibration_used": False,
                "ensemble_used": False,
                "trackman_used": False,
                "code_files_sha256": _final_code_inventory(),
            },
            "artifact_manifest_sha256",
        )
        (staging / "manifest.json").write_bytes(_serialize_json(manifest))
        os.replace(staging, destination)
    finally:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)


def load_final_context_artifact(
    directory: str | Path,
    *,
    verify_code: bool = True,
) -> LoadedFinalContextArtifact:
    root = Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"final artifact directory is missing: {root}")
    entries = list(root.iterdir())
    expected_entries = FINAL_TRACKED_FILES.union({"manifest.json"})
    if (
        {path.name for path in entries} != expected_entries
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError("final artifact file inventory differs")

    manifest = _read_json(root / "manifest.json")
    _verify_embedded_hash(manifest, "artifact_manifest_sha256")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("checkpoint_kind") != FINAL_CHECKPOINT_KIND
        or manifest.get("source_input_sha256") != EXPECTED_FULL_SOURCE_SHA256
        or manifest.get("source_rows") != EXPECTED_FULL_ROWS
        or manifest.get("source_row_id_sha256") != EXPECTED_FULL_ROW_ID_SHA256
        or manifest.get("source_season_counts") != EXPECTED_SEASON_COUNTS
        or manifest.get("anchor_confirmation_sha256")
        != EXPECTED_ANCHOR_CONFIRMATION_SHA256
        or manifest.get("anchor_confirmation_file_sha256")
        != EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256
        or manifest.get("anchor_ready_sha256") != EXPECTED_ANCHOR_READY_SHA256
        or manifest.get("anchor_metrics_sha256") != EXPECTED_ANCHOR_METRICS_SHA256
        or manifest.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID
        or manifest.get("selected_epoch") != EXPECTED_SELECTED_EPOCH
        or manifest.get("model_count") != 1
        or manifest.get("calibration_used") is not False
        or manifest.get("ensemble_used") is not False
        or manifest.get("trackman_used") is not False
    ):
        raise ValueError("final artifact manifest differs from the fixed contract")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != FINAL_TRACKED_FILES:
        raise ValueError("final artifact hash inventory differs")
    for name, expected in hashes.items():
        if _sha256_file(root / name) != expected:
            raise ValueError(f"final artifact file hash differs: {name}")
    if verify_code and manifest.get("code_files_sha256") != _final_code_inventory():
        raise ValueError("final training code differs from the artifact manifest")

    anchor_provenance = _read_json(root / "anchor_provenance.json")
    anchor_provenance_sha = _verify_embedded_hash(
        anchor_provenance, "anchor_provenance_sha256"
    )
    if manifest.get("anchor_provenance_sha256") != anchor_provenance_sha:
        raise ValueError("final manifest anchor provenance link differs")
    verified_anchor = anchor_provenance.get("verified_anchor", {})
    source_provenance = anchor_provenance.get("source_provenance", {})
    if (
        anchor_provenance.get("schema_version") != SCHEMA_VERSION
        or anchor_provenance.get("phase") != "final_context_anchor_provenance"
        or verified_anchor.get("confirmation_sha256")
        != EXPECTED_ANCHOR_CONFIRMATION_SHA256
        or verified_anchor.get("confirmation_file_sha256")
        != EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256
        or verified_anchor.get("anchor_ready_sha256") != EXPECTED_ANCHOR_READY_SHA256
        or verified_anchor.get("anchor_ready_file_sha256")
        != EXPECTED_ANCHOR_READY_FILE_SHA256
        or verified_anchor.get("metrics_sha256") != EXPECTED_ANCHOR_METRICS_SHA256
        or verified_anchor.get("metrics_file_sha256")
        != EXPECTED_ANCHOR_METRICS_FILE_SHA256
        or verified_anchor.get("predictions_commit_sha256")
        != EXPECTED_PREDICTIONS_COMMIT_SHA256
        or verified_anchor.get("prediction_file_sha256")
        != EXPECTED_ANCHOR_PREDICTION_FILE_SHA256
        or verified_anchor.get("promotion_lock_sha256")
        != EXPECTED_PROMOTION_LOCK_SHA256
        or verified_anchor.get("selected_candidate_id")
        != EXPECTED_SELECTED_CANDIDATE_ID
        or verified_anchor.get("selected_epoch") != EXPECTED_SELECTED_EPOCH
        or verified_anchor.get("selected_model_config")
        != _expected_candidate_spec()["model_config"]
        or verified_anchor.get("optimization") != _expected_optimization().to_dict()
        or verified_anchor.get("seeds") != ANCHOR_SEEDS.to_dict()
        or verified_anchor.get("expected_full_source_sha256")
        != EXPECTED_FULL_SOURCE_SHA256
        or verified_anchor.get("anchor_rows") != EXPECTED_SEASON_COUNTS["2024"]
        or verified_anchor.get("gates") != _expected_anchor_gates()
    ):
        raise ValueError("final artifact anchor provenance differs")
    if (
        not isinstance(source_provenance, Mapping)
        or source_provenance.get("sha256") != EXPECTED_FULL_SOURCE_SHA256
        or source_provenance.get("rows") != EXPECTED_FULL_ROWS
        or source_provenance.get("row_id_sha256") != EXPECTED_FULL_ROW_ID_SHA256
        or source_provenance.get("seasons") != list(FINAL_SEASONS)
        or source_provenance.get("season_counts") != EXPECTED_SEASON_COUNTS
        or source_provenance.get("target_rate") != EXPECTED_FULL_TARGET_RATE
    ):
        raise ValueError("final artifact source provenance differs")

    preprocessor = DeepTabularPreprocessor.load(root / "preprocessor.json")
    model_payload = _read_json(root / "model_config.json")
    expected_config = ContextGatedModelConfig.from_dict(
        _expected_candidate_spec()["model_config"]
    )
    model = ContextGatedResidualModel.from_config(model_payload)
    assert_preprocessor_compatible(model.feature_layout, preprocessor)
    if model.config.to_dict() != expected_config.to_dict():
        raise ValueError("final artifact model config is not C2 neutral")
    if (
        preprocessor.min_category_frequency != 1
        or preprocessor.clip_value != 10.0
        or
        manifest.get("preprocessor_fingerprint")
        != model.feature_layout.preprocessor_fingerprint
        or manifest.get("preprocessor_state_sha256")
        != _preprocessor_state_sha256(preprocessor)
    ):
        raise ValueError("final artifact preprocessor state differs")

    expected_state = model.state_dict()
    state_schema = manifest.get("model_state_schema")
    if not isinstance(state_schema, Mapping) or set(state_schema) != set(expected_state):
        raise ValueError("final artifact tensor schema differs")
    with np.load(root / "model_state.npz", allow_pickle=False) as archive:
        if (
            len(archive.files) != len(expected_state)
            or len(set(archive.files)) != len(archive.files)
            or set(archive.files) != set(expected_state)
        ):
            raise ValueError("final artifact tensor keys differ")
        state: dict[str, torch.Tensor] = {}
        for name in archive.files:
            array = archive[name]
            architecture = expected_state[name].detach().cpu().numpy()
            schema = state_schema[name]
            if (
                not isinstance(schema, Mapping)
                or schema.get("shape") != list(array.shape)
                or schema.get("dtype") != str(array.dtype)
                or array.shape != architecture.shape
                or array.dtype != architecture.dtype
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"final artifact tensor differs: {name}")
            state[name] = torch.from_numpy(array.copy())
    model.load_state_dict(state, strict=True)
    model.to("cpu").eval()
    if manifest.get("model_state_sha256") != _model_state_sha256(model):
        raise ValueError("final artifact model state hash differs")

    record = _read_json(root / "training_record.json")
    record_sha = _verify_embedded_hash(record, "training_record_sha256")
    if manifest.get("training_record_sha256") != record_sha:
        raise ValueError("final manifest training record link differs")
    required_record = {
        "schema_version": SCHEMA_VERSION,
        "phase": "final_context_full_train",
        "checkpoint_kind": FINAL_CHECKPOINT_KIND,
        "selected_candidate_id": EXPECTED_SELECTED_CANDIDATE_ID,
        "selected_epoch": EXPECTED_SELECTED_EPOCH,
        "train_seasons": list(FINAL_SEASONS),
        "train_rows": EXPECTED_FULL_ROWS,
        "train_row_id_sha256": EXPECTED_FULL_ROW_ID_SHA256,
        "source_input_sha256": EXPECTED_FULL_SOURCE_SHA256,
        "anchor_confirmation_sha256": EXPECTED_ANCHOR_CONFIRMATION_SHA256,
        "anchor_ready_sha256": EXPECTED_ANCHOR_READY_SHA256,
        "model_count": 1,
        "fresh_model": True,
        "warm_start": False,
        "validation_performed": False,
        "early_stopping_performed": False,
        "recency_or_group_weighting": False,
        "trackman_used": False,
        "calibration_used": False,
        "ensemble_used": False,
    }
    for name, value in required_record.items():
        if record.get(name) != value:
            raise ValueError(f"final training record differs: {name}")
    history = record.get("history")
    if (
        record.get("model_config") != model.get_config()
        or record.get("architecture_procedure_sha256")
        != _architecture_procedure_sha256(expected_config)
        or record.get("parameter_report") != model.parameter_report()
        or record.get("preprocessor_fingerprint")
        != model.feature_layout.preprocessor_fingerprint
        or record.get("preprocessor_state_sha256")
        != _preprocessor_state_sha256(preprocessor)
        or record.get("final_model_state_sha256") != _model_state_sha256(model)
        or record.get("optimization")
        != _expected_optimization().to_dict()
        or OptimizationProcedure.from_dict(verified_anchor["optimization"]).to_dict()
        != _expected_optimization().to_dict()
        or record.get("seeds") != ANCHOR_SEEDS.to_dict()
        or not isinstance(history, list)
        or len(history) != EXPECTED_SELECTED_EPOCH
        or history[0].get("epoch") != EXPECTED_SELECTED_EPOCH
        or history[0].get("learning_rate") != 0.001
        or not np.isfinite(float(history[0].get("train_loss", float("nan"))))
        or float(history[0].get("train_loss", -1.0)) < 0.0
    ):
        raise ValueError("final training record does not reconcile")
    fresh_model = build_context_gated_model(
        preprocessor,
        config=expected_config,
        initialization_seed=ANCHOR_SEEDS.initialization_seed,
    )
    expected_initial_sha = _model_state_sha256(fresh_model)
    if (
        record.get("initial_model_state_sha256") != expected_initial_sha
        or record.get("final_model_state_sha256") == expected_initial_sha
    ):
        raise ValueError("final artifact does not prove a fresh trained model")
    return LoadedFinalContextArtifact(
        directory=str(root),
        preprocessor=preprocessor,
        model=model,
        training_record=record,
        anchor_provenance=anchor_provenance,
        manifest=manifest,
    )


def train_final_context(
    *,
    train_input: str | Path,
    anchor_directory: str | Path,
    output_directory: str | Path,
    resume: bool = False,
    verbose: bool = True,
) -> LoadedFinalContextArtifact:
    anchor = verify_confirmed_anchor(anchor_directory)
    train_path = Path(train_input).resolve()
    output = Path(output_directory).resolve()
    anchor_root = Path(anchor_directory).resolve()
    if (
        output == train_path
        or output == anchor_root
        or anchor_root in output.parents
        or output in train_path.parents
    ):
        raise ValueError("final output must be separate from train and anchor artifacts")
    if output.exists():
        if not resume:
            raise FileExistsError(f"final output exists; use resume: {output}")
        if _sha256_file(train_path) != anchor.expected_full_source_sha256:
            raise ValueError("resume train source differs from confirmed source")
        return load_final_context_artifact(output, verify_code=True)

    frame, source_provenance = _load_full_training_frame(
        train_path,
        expected_sha256=anchor.expected_full_source_sha256,
    )
    preprocessor, model, record = _train_fresh_model(
        frame,
        anchor=anchor,
        source_provenance=source_provenance,
        verbose=verbose,
    )
    _save_final_artifact(
        output,
        preprocessor=preprocessor,
        model=model,
        training_record=record,
        anchor=anchor,
        source_provenance=source_provenance,
    )
    return load_final_context_artifact(output, verify_code=True)


def predict_final_context(
    loaded: LoadedFinalContextArtifact,
    frame: pd.DataFrame,
    *,
    batch_size: int = 16_384,
) -> np.ndarray:
    if not isinstance(loaded, LoadedFinalContextArtifact):
        raise TypeError("loaded must be a LoadedFinalContextArtifact")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validate_input_frame(frame, require_target=False, allow_target=False)
    if frame.empty or not frame[ID_COL].is_unique:
        raise ValueError("inference row IDs must be nonempty and unique")
    pre_before = _preprocessor_state_sha256(loaded.preprocessor)
    model_before = _model_state_sha256(loaded.model)
    transformed = loaded.preprocessor.transform(frame)
    if "target" in transformed:
        raise RuntimeError("target entered final inference preprocessing")
    validate_transformed_arrays(
        loaded.model.feature_layout,
        transformed,
        expected_row_ids=frame[ID_COL].to_numpy(dtype=object, copy=False),
    )
    probabilities = _predict(
        loaded.model,
        transformed["categorical"],
        transformed["numerical"],
        config=EvaluationConfig(
            eval_batch_size=batch_size,
            num_workers=0,
            cpu_threads=loaded.training_record["optimization"]["cpu_threads"],
        ),
        device=torch.device("cpu"),
    )
    if pre_before != _preprocessor_state_sha256(loaded.preprocessor):
        raise RuntimeError("final inference mutated the preprocessor")
    if model_before != _model_state_sha256(loaded.model):
        raise RuntimeError("final inference mutated the model")
    return validate_predictions(frame[ID_COL], probabilities)


__all__ = [
    "FINAL_CHECKPOINT_KIND",
    "FINAL_SEASONS",
    "EXPECTED_ANCHOR_CONFIRMATION_SHA256",
    "EXPECTED_ANCHOR_READY_SHA256",
    "EXPECTED_FULL_SOURCE_SHA256",
    "LoadedFinalContextArtifact",
    "VerifiedFinalAnchor",
    "load_final_context_artifact",
    "predict_final_context",
    "train_final_context",
    "verify_confirmed_anchor",
]
