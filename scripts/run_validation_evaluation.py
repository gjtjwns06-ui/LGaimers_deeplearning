"""Evaluate the completed 2022/2023 development checkpoints, and stop.

The command creates outer-validation probabilities and fixed performance
reports.  It does not train, select, tune, calibrate, ensemble, evaluate the
2024 anchor, run test inference, or create a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import ID_COL, TRAIN_COLUMNS
from lg_control.evaluation import (
    EvaluationConfig,
    FoldEvaluation,
    _partition_masks,
    _slice_metrics,
    aggregate_development_evaluations,
    evaluate_outer_validation,
    probability_metrics,
)
from lg_control.learning import row_id_sha256
from lg_control.learning_artifacts import load_learning_checkpoint
from lg_control.validation import build_validation_plan


DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_LEARNING_DIR = PROJECT_ROOT / "artifacts" / "learning"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "validation_evaluation"
DEVELOPMENT_SEASONS = (2022, 2023)
PREDICTION_COLUMNS = (ID_COL, "season", "y_true", "prediction")
EXPECTED_EVALUATION_FILES = frozenset(
    {
        "aggregate_metrics.json",
        "development_oof_predictions.csv",
        "fold_metrics.csv",
        "slice_metrics.csv",
        "rolling_origin_2022/evaluation_record.json",
        "rolling_origin_2022/predictions.csv",
        "rolling_origin_2023/evaluation_record.json",
        "rolling_origin_2023/predictions.csv",
    }
)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_learning_provenance(
    *,
    learning_dir: Path,
    input_path: Path,
    input_sha256: str,
    input_rows: int,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest_path = learning_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    recorded_input = manifest.get("input")
    if not isinstance(recorded_input, Mapping):
        raise ValueError("learning manifest input provenance is missing")
    if recorded_input.get("sha256") != input_sha256:
        raise ValueError("learning manifest input SHA-256 differs from evaluation input")
    if recorded_input.get("size_bytes") != input_path.stat().st_size:
        raise ValueError("learning manifest input size differs from evaluation input")
    if recorded_input.get("rows") != input_rows:
        raise ValueError("learning manifest input rows differ from evaluation input")
    boundary = manifest.get("stage_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("learning manifest stage boundary is missing")
    for name in (
        "outer_validation_metrics_computed",
        "outer_validation_predictions_created",
        "hyperparameters_tuned",
        "anchor_2024_trained_or_evaluated",
        "calibration_or_ensemble_executed",
        "inference_or_submission_executed",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"learning provenance violates stage boundary: {name}")

    raw_folds = manifest.get("folds")
    if not isinstance(raw_folds, list):
        raise ValueError("learning manifest fold records are missing")
    by_season: dict[int, dict[str, Any]] = {}
    for raw in raw_folds:
        if not isinstance(raw, dict):
            raise ValueError("learning manifest fold record is malformed")
        season = int(raw.get("reserved_outer_validation_season"))
        if season in by_season:
            raise ValueError("learning manifest contains duplicate fold seasons")
        by_season[season] = raw
    if tuple(sorted(by_season)) != DEVELOPMENT_SEASONS:
        raise ValueError("learning manifest must contain exactly 2022/2023 checkpoints")

    for season, raw in by_season.items():
        fold_name = f"rolling_origin_{season}"
        if raw.get("fold") != fold_name:
            raise ValueError("learning fold name and validation season disagree")
        checkpoint_manifest = learning_dir / fold_name / "checkpoint" / "manifest.json"
        selection_record = learning_dir / fold_name / "selection_record.json"
        if _sha256_file(checkpoint_manifest) != raw.get(
            "checkpoint_manifest_sha256"
        ):
            raise ValueError(f"learning checkpoint manifest hash mismatch: {fold_name}")
        if _sha256_file(selection_record) != raw.get("selection_record_sha256"):
            raise ValueError(f"learning selection record hash mismatch: {fold_name}")
        if raw.get("outer_validation_rows_accessed") is not False:
            raise ValueError("learning checkpoint used outer validation before evaluation")
    return manifest, by_season


def _metric_values_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    atol: float = 1e-14,
) -> bool:
    if set(expected) != set(actual):
        return False
    for key in expected:
        left, right = expected[key], actual[key]
        if left is None or right is None:
            if left is not None or right is not None:
                return False
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not np.isclose(float(left), float(right), rtol=0.0, atol=atol):
                return False
        elif left != right:
            return False
    return True


def _nested_values_match(expected: Any, actual: Any, *, atol: float = 1e-14) -> bool:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(
            _nested_values_match(expected[key], actual[key], atol=atol)
            for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _nested_values_match(left, right, atol=atol)
            for left, right in zip(expected, actual, strict=True)
        )
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return bool(np.isclose(float(expected), float(actual), rtol=0.0, atol=atol))
    return expected == actual


def verify_evaluation_artifacts(
    directory: str | Path,
    *,
    input_path: str | Path | None = None,
    learning_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hash-check and semantically reconcile a completed evaluation artifact."""

    source = Path(directory).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"evaluation directory does not exist: {source}")
    root_manifest_path = (source / "manifest.json").resolve()
    manifest = _read_json(root_manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported evaluation artifact schema")
    if manifest.get("phase") != "development_outer_validation_evaluation":
        raise ValueError("evaluation artifact phase is invalid")
    file_hashes = manifest.get("files_sha256")
    if not isinstance(file_hashes, dict):
        raise ValueError("evaluation file hash inventory is missing")
    actual_files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.resolve() != root_manifest_path
    }
    if actual_files != EXPECTED_EVALUATION_FILES:
        raise ValueError("evaluation artifact contains a missing or unexpected file")
    if set(file_hashes) != EXPECTED_EVALUATION_FILES:
        raise ValueError("evaluation file hash inventory is incomplete or unexpected")
    for relative in sorted(EXPECTED_EVALUATION_FILES):
        expected_hash = file_hashes[relative]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"invalid SHA-256 for {relative}")
        path = (source / relative).resolve()
        if source not in path.parents:
            raise ValueError("evaluation hash inventory contains an unsafe path")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"evaluation file hash mismatch: {relative}")

    aggregate = _read_json(source / "aggregate_metrics.json")
    if not _nested_values_match(manifest.get("aggregate_metrics"), aggregate):
        raise ValueError("root manifest aggregate differs from aggregate_metrics.json")
    manifest_folds = manifest.get("folds")
    if not isinstance(manifest_folds, list) or len(manifest_folds) != 2:
        raise ValueError("root manifest must contain exactly two evaluation folds")
    manifest_folds_by_season = {
        int(item.get("validation_season")): item
        for item in manifest_folds
        if isinstance(item, dict)
    }
    if tuple(sorted(manifest_folds_by_season)) != DEVELOPMENT_SEASONS:
        raise ValueError("root manifest fold seasons differ from evaluation contract")

    fold_frames: list[pd.DataFrame] = []
    fold_results: list[FoldEvaluation] = []
    records_by_season: dict[int, dict[str, Any]] = {}
    predictions_by_season: dict[int, pd.DataFrame] = {}
    for season in DEVELOPMENT_SEASONS:
        fold_name = f"rolling_origin_{season}"
        prediction_relative = f"{fold_name}/predictions.csv"
        record_relative = f"{fold_name}/evaluation_record.json"
        predictions = pd.read_csv(
            source / prediction_relative,
            low_memory=False,
            float_precision="round_trip",
        )
        if tuple(predictions.columns) != PREDICTION_COLUMNS:
            raise ValueError(f"prediction columns differ from contract: {fold_name}")
        if predictions.empty or not predictions[ID_COL].is_unique:
            raise ValueError(f"prediction IDs are empty or duplicated: {fold_name}")
        if not predictions["season"].eq(season).all():
            raise ValueError(f"prediction season mismatch: {fold_name}")
        record = _read_json(source / record_relative)
        if record.get("phase") != "development_outer_validation_evaluation":
            raise ValueError(f"evaluation record phase mismatch: {fold_name}")
        if record.get("fold") != fold_name or record.get("validation_season") != season:
            raise ValueError(f"evaluation record fold identity mismatch: {fold_name}")
        if record.get("validation_rows") != len(predictions):
            raise ValueError(f"prediction row count mismatch: {fold_name}")
        if record.get("validation_row_id_sha256") != row_id_sha256(
            predictions[ID_COL].to_numpy(dtype=object)
        ):
            raise ValueError(f"prediction row ID hash mismatch: {fold_name}")
        for flag in (
            "target_removed_before_preprocessing",
            "prediction_order_preserved",
            "model_state_unchanged",
            "preprocessor_state_unchanged",
        ):
            if record.get(flag) is not True:
                raise ValueError(f"evaluation record invariant failed: {fold_name}/{flag}")
        if record.get("target_consumed_by_model") is not False:
            raise ValueError(f"validation target entered the model: {fold_name}")
        for flag in (
            "selection_or_tuning_performed",
            "calibration_or_ensemble_performed",
            "anchor_2024_evaluated",
            "inference_or_submission_performed",
        ):
            if record.get(flag) is not False:
                raise ValueError(f"evaluation scope flag failed: {fold_name}/{flag}")
        recomputed = probability_metrics(
            predictions["y_true"].to_numpy(),
            predictions["prediction"].to_numpy(),
        )
        if not _metric_values_match(record.get("metrics", {}), recomputed):
            raise ValueError(f"prediction metrics do not reconcile: {fold_name}")

        root_fold = manifest_folds_by_season[season]
        expected_root_fields = {
            "fold": fold_name,
            "validation_season": season,
            "validation_rows": len(predictions),
            "validation_row_id_sha256": record["validation_row_id_sha256"],
            "selected_epoch": root_fold.get("selected_epoch"),
            "checkpoint_manifest_sha256": record["checkpoint_manifest_sha256"],
            "prediction_path": prediction_relative,
            "evaluation_record_path": record_relative,
            "metrics": record["metrics"],
        }
        if not _nested_values_match(expected_root_fields, root_fold):
            raise ValueError(f"root manifest fold record does not reconcile: {fold_name}")
        if not isinstance(root_fold.get("selected_epoch"), int) or root_fold[
            "selected_epoch"
        ] < 1:
            raise ValueError(f"root manifest selected epoch is invalid: {fold_name}")

        fold_frames.append(predictions)
        records_by_season[season] = record
        predictions_by_season[season] = predictions
        fold_results.append(
            FoldEvaluation(
                fold_name=fold_name,
                validation_season=season,
                predictions=predictions,
                metrics=record["metrics"],
                slice_metrics=pd.DataFrame(),
                evaluation_record=record,
            )
        )

    combined = pd.concat(fold_frames, ignore_index=True)
    saved_oof = pd.read_csv(
        source / "development_oof_predictions.csv",
        low_memory=False,
        float_precision="round_trip",
    )
    try:
        pd.testing.assert_frame_equal(
            saved_oof, combined, check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError("saved OOF is not the exact chronological fold concatenation") from error
    if not saved_oof[ID_COL].is_unique:
        raise ValueError("development OOF row IDs overlap across folds")

    reconstructed = aggregate_development_evaluations(fold_results)
    if not _nested_values_match(aggregate, reconstructed.aggregate_metrics):
        raise ValueError("aggregate macro/pooled/weighted/worst metrics do not reconcile")
    saved_fold_metrics = pd.read_csv(
        source / "fold_metrics.csv", low_memory=False, float_precision="round_trip"
    )
    if set(saved_fold_metrics.columns) != set(reconstructed.fold_metrics.columns):
        raise ValueError("fold_metrics.csv columns differ from the metric contract")
    expected_fold_metrics = reconstructed.fold_metrics.loc[
        :, saved_fold_metrics.columns
    ]
    try:
        pd.testing.assert_frame_equal(
            saved_fold_metrics,
            expected_fold_metrics,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-14,
        )
    except AssertionError as error:
        raise ValueError("fold_metrics.csv does not reconcile with predictions") from error

    saved_slices = pd.read_csv(
        source / "slice_metrics.csv", low_memory=False, float_precision="round_trip"
    )
    required_slice_values = {
        "overall": {"all"},
        "game_type": {"F", "R"},
        "entity_status": {"both_seen", "pitcher_only", "batter_only", "neither_seen"},
        "recent_status": {"recent_missing", "recent_observed"},
        "pitcher_history": {
            "cold_start_0", "low_history_1_49", "history_50_plus", "history_missing"
        },
        "batter_history": {
            "cold_start_0", "low_history_1_49", "history_50_plus", "history_missing"
        },
    }
    metric_columns = tuple(probability_metrics(np.array([0.0, 1.0]), np.array([0.5, 0.5])))
    for season in DEVELOPMENT_SEASONS:
        fold_name = f"rolling_origin_{season}"
        fold_slices = saved_slices.loc[saved_slices["fold"].eq(fold_name)]
        for family, values in required_slice_values.items():
            part = fold_slices.loc[fold_slices["slice"].eq(family)]
            if set(part["value"]) != values:
                raise ValueError(f"slice values differ from contract: {fold_name}/{family}")
            if int(part["rows"].sum()) != len(predictions_by_season[season]):
                raise ValueError(f"slice rows do not cover fold: {fold_name}/{family}")
            if not np.isclose(float(part["share"].sum()), 1.0, rtol=0.0, atol=1e-14):
                raise ValueError(f"slice shares do not cover fold: {fold_name}/{family}")
        overall = fold_slices.loc[
            fold_slices["slice"].eq("overall") & fold_slices["value"].eq("all")
        ]
        if len(overall) != 1:
            raise ValueError(f"overall slice missing or duplicated: {fold_name}")
        overall_metrics = {
            name: (
                None if pd.isna(overall.iloc[0][name]) else overall.iloc[0][name]
            )
            for name in metric_columns
        }
        if not _metric_values_match(records_by_season[season]["metrics"], overall_metrics):
            raise ValueError(f"overall slice metrics differ from fold metrics: {fold_name}")
        empty = fold_slices.loc[fold_slices["rows"].eq(0)]
        nullable_metrics = [name for name in metric_columns if name != "rows"]
        if len(empty) and not empty.loc[:, nullable_metrics].isna().all().all():
            raise ValueError(f"empty slice contains non-null metrics: {fold_name}")

    if (input_path is None) != (learning_dir is None):
        raise ValueError("input_path and learning_dir must be supplied together")
    if input_path is not None and learning_dir is not None:
        verified_input = Path(input_path).resolve()
        verified_learning = Path(learning_dir).resolve()
        input_record = manifest.get("input")
        if not isinstance(input_record, Mapping):
            raise ValueError("evaluation input provenance is missing")
        if _sha256_file(verified_input) != input_record.get("sha256"):
            raise ValueError("evaluation input file hash differs from manifest")
        frame = pd.read_csv(verified_input, low_memory=False)
        if list(frame.columns) != TRAIN_COLUMNS or len(frame) != input_record.get("rows"):
            raise ValueError("evaluation input schema or row count differs from manifest")
        if row_id_sha256(frame[ID_COL].to_numpy(dtype=object)) != input_record.get(
            "source_row_id_sha256"
        ):
            raise ValueError("evaluation input row identity differs from manifest")
        learning_record = manifest.get("learning_provenance")
        if not isinstance(learning_record, Mapping) or _sha256_file(
            verified_learning / "manifest.json"
        ) != learning_record.get("manifest_sha256"):
            raise ValueError("learning manifest provenance differs from evaluation")

        plan = build_validation_plan(frame, include_recent_only=False)
        folds_by_season = {
            fold.validation_season: fold for fold in plan.development_folds
        }
        expected_slice_parts: list[pd.DataFrame] = []
        for season in DEVELOPMENT_SEASONS:
            fold = folds_by_season[season]
            prediction_frame = predictions_by_season[season]
            validation = fold.validation_frame(frame)
            if not np.array_equal(
                prediction_frame[ID_COL].to_numpy(dtype=object), fold.validation_row_ids
            ):
                raise ValueError(f"saved prediction row order differs from source: {fold.name}")
            source_target = pd.to_numeric(
                validation["control_success"], errors="coerce"
            ).to_numpy(dtype=np.int8)
            if not np.array_equal(
                prediction_frame["y_true"].to_numpy(dtype=np.int8), source_target
            ):
                raise ValueError(f"saved targets differ from source: {fold.name}")
            checkpoint_dir = verified_learning / fold.name / "checkpoint"
            if _sha256_file(checkpoint_dir / "manifest.json") != records_by_season[
                season
            ]["checkpoint_manifest_sha256"]:
                raise ValueError(f"checkpoint provenance differs from evaluation: {fold.name}")
            loaded = load_learning_checkpoint(checkpoint_dir, device="cpu")
            transformed = loaded.preprocessor.transform(
                validation.drop(columns=["control_success"])
            )
            masks = _partition_masks(
                validation,
                transformed["categorical"],
                categorical_names=loaded.model.feature_layout.categorical_names,
            )
            expected_slice_parts.append(
                _slice_metrics(
                    fold=fold,
                    target=source_target.astype(np.float64),
                    probabilities=prediction_frame["prediction"].to_numpy(),
                    masks=masks,
                )
            )
        expected_slices = pd.concat(expected_slice_parts, ignore_index=True)
        if set(saved_slices.columns) != set(expected_slices.columns):
            raise ValueError("slice_metrics.csv columns differ from the slice contract")
        expected_slices = expected_slices.loc[:, saved_slices.columns]
        try:
            pd.testing.assert_frame_equal(
                saved_slices,
                expected_slices,
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=1e-14,
            )
        except AssertionError as error:
            raise ValueError("slice_metrics.csv does not reconcile with source rows") from error

    boundary = manifest.get("stage_boundary")
    required_false = (
        "training_performed",
        "checkpoint_or_epoch_selected",
        "hyperparameters_tuned",
        "calibration_or_ensemble_performed",
        "anchor_2024_predicted_or_scored",
        "test_inference_or_submission_performed",
    )
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("evaluated_outer_validation_seasons") != list(DEVELOPMENT_SEASONS)
        or any(boundary.get(name) is not False for name in required_false)
    ):
        raise ValueError("evaluation stage boundary flags are invalid")
    return manifest


def run_validation_evaluation(
    *,
    input_path: Path,
    learning_dir: Path,
    output_dir: Path,
    config: EvaluationConfig,
    device: str,
    verbose: bool,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    learning_dir = learning_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    input_sha256 = _sha256_file(input_path)
    frame = pd.read_csv(input_path, low_memory=False)
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("train.csv column order does not match the official schema")
    learning_manifest, learning_folds = _assert_learning_provenance(
        learning_dir=learning_dir,
        input_path=input_path,
        input_sha256=input_sha256,
        input_rows=len(frame),
    )
    learning_hashes_before = _tree_hashes(learning_dir)
    plan = build_validation_plan(frame, include_recent_only=False)
    folds_by_season = {
        fold.validation_season: fold for fold in plan.development_folds
    }
    if tuple(sorted(folds_by_season)) != DEVELOPMENT_SEASONS:
        raise AssertionError("development validation plan changed unexpectedly")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    ).resolve()
    if staging.parent != output_dir.parent:
        raise RuntimeError("evaluation staging directory escaped output parent")
    try:
        fold_evaluations = []
        fold_manifest_records: list[dict[str, Any]] = []
        for season in DEVELOPMENT_SEASONS:
            fold = folds_by_season[season]
            checkpoint_dir = learning_dir / fold.name / "checkpoint"
            if verbose:
                print(f"[{fold.name}] transforming and predicting outer validation", flush=True)
            result = evaluate_outer_validation(
                frame,
                fold,
                checkpoint_dir,
                config=config,
                device=device,
                verbose=verbose,
            )
            fold_dir = staging / fold.name
            prediction_path = fold_dir / "predictions.csv"
            record_path = fold_dir / "evaluation_record.json"
            _write_csv(result.predictions, prediction_path)
            _write_json(record_path, result.evaluation_record)
            fold_evaluations.append(result)
            fold_manifest_records.append(
                {
                    "fold": fold.name,
                    "validation_season": season,
                    "validation_rows": fold.validation_row_count,
                    "validation_row_id_sha256": result.evaluation_record[
                        "validation_row_id_sha256"
                    ],
                    "selected_epoch": learning_folds[season]["selected_epoch"],
                    "checkpoint_manifest_sha256": result.evaluation_record[
                        "checkpoint_manifest_sha256"
                    ],
                    "prediction_path": prediction_path.relative_to(staging).as_posix(),
                    "evaluation_record_path": record_path.relative_to(staging).as_posix(),
                    "metrics": result.metrics,
                }
            )

        combined = aggregate_development_evaluations(fold_evaluations)
        _write_csv(combined.oof_predictions, staging / "development_oof_predictions.csv")
        _write_csv(combined.fold_metrics, staging / "fold_metrics.csv")
        _write_csv(combined.slice_metrics, staging / "slice_metrics.csv")
        _write_json(staging / "aggregate_metrics.json", combined.aggregate_metrics)

        learning_hashes_after = _tree_hashes(learning_dir)
        if learning_hashes_after != learning_hashes_before:
            raise RuntimeError("evaluation modified a learning checkpoint artifact")
        tracked_files = {
            path.relative_to(staging).as_posix(): _sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "phase": "development_outer_validation_evaluation",
            "scope": (
                "predict and score only the existing 2022/2023 development "
                "outer-validation checkpoints"
            ),
            "input": {
                "path": str(input_path),
                "size_bytes": input_path.stat().st_size,
                "rows": int(len(frame)),
                "sha256": input_sha256,
                "source_row_id_sha256": row_id_sha256(
                    frame[ID_COL].to_numpy(dtype=object, copy=False)
                ),
            },
            "learning_provenance": {
                "directory": str(learning_dir),
                "manifest_sha256": _sha256_file(learning_dir / "manifest.json"),
                "input_sha256": learning_manifest["input"]["sha256"],
                "artifact_files_before_and_after_identical": True,
            },
            "evaluation_config": config.to_dict(),
            "folds": fold_manifest_records,
            "aggregate_metrics": combined.aggregate_metrics,
            "files_sha256": tracked_files,
            "stage_boundary": {
                "evaluated_outer_validation_seasons": list(DEVELOPMENT_SEASONS),
                "training_performed": False,
                "checkpoint_or_epoch_selected": False,
                "hyperparameters_tuned": False,
                "calibration_or_ensemble_performed": False,
                "anchor_2024_predicted_or_scored": False,
                "test_inference_or_submission_performed": False,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        # Re-read hashes, row contracts, and metrics while the directory is
        # still unpublished.  Only a complete verified artifact is renamed
        # into the user-visible destination.
        verify_evaluation_artifacts(
            staging,
            input_path=input_path,
            learning_dir=learning_dir,
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.parent == output_dir.parent:
            shutil.rmtree(staging)
        raise

    verified = verify_evaluation_artifacts(
        output_dir,
        input_path=input_path,
        learning_dir=learning_dir,
    )
    if verbose:
        print(f"[evaluation] artifact hashes and metrics verified: {output_dir}", flush=True)
    return verified


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--learning-dir", type=Path, default=DEFAULT_LEARNING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Only hash-check and reconcile an existing evaluation artifact.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verify_existing:
        manifest = verify_evaluation_artifacts(
            args.output_dir,
            input_path=args.input,
            learning_dir=args.learning_dir,
        )
    else:
        manifest = run_validation_evaluation(
            input_path=args.input,
            learning_dir=args.learning_dir,
            output_dir=args.output_dir,
            config=EvaluationConfig(
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                cpu_threads=args.cpu_threads,
            ),
            device=args.device,
            verbose=not args.quiet,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "phase": manifest["phase"],
                "fold_brier": {
                    fold["fold"]: fold["metrics"]["brier"]
                    for fold in manifest["folds"]
                },
                "pooled_brier": manifest["aggregate_metrics"]["row_pooled"][
                    "brier"
                ],
                "anchor_2024_predicted_or_scored": False,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
