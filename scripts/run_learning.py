"""Run only the leakage-safe learning stage for development folds.

For each requested development fold, this runner first selects an epoch count
on the fold-contained inner train/stop partition.  It then discards that model
and fits a newly initialized model for exactly that many epochs on the complete
outer-training partition.  Outer-validation rows are never transformed,
predicted, scored, or used for checkpoint selection here.

The 2024 anchor, recent-only diagnostic, hyperparameter tuning, calibration,
ensembling, inference, and submission are deliberately outside this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import TRAIN_COLUMNS
from lg_control.learning import (
    EpochSelectionRecord,
    OptimizationProcedure,
    SeedSpec,
    refit_outer_fixed_epochs,
    row_id_sha256,
    select_epoch_on_inner_stop,
)
from lg_control.learning_artifacts import (
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from lg_control.model import ContextGatedModelConfig
from lg_control.validation import SeasonFold, build_validation_plan


DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "learning"
DEVELOPMENT_SEASONS = (2022, 2023)
SEED_STRIDE_PER_FOLD = 100
SEED_OFFSETS = {
    "initialization_seed": 0,
    "training_seed": 1,
    "loader_seed": 2,
}


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"object of type {type(value).__name__} is not JSON-recordable")
    if not isinstance(payload, dict):
        raise TypeError("to_dict() must return a dictionary")
    # Validate serializability and prohibit NaN/Infinity before touching disk.
    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably create one JSON file without overwriting an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _write_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a deterministic manifest, or verify an identical existing one."""

    expected = dict(payload)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise FileExistsError(
                f"existing manifest differs from this invocation: {path}"
            )
        return
    _write_json_exclusive(path, expected)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must resolve to cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _normalise_fold_seasons(values: Sequence[int]) -> tuple[int, ...]:
    requested = tuple(int(value) for value in values)
    if not requested:
        raise ValueError("at least one development fold is required")
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate --folds values are not allowed: {requested}")
    prohibited = sorted(set(requested).difference(DEVELOPMENT_SEASONS))
    if prohibited:
        raise ValueError(
            "learning-stage execution is restricted to development seasons "
            f"{DEVELOPMENT_SEASONS}; received {prohibited}"
        )
    # Canonical chronological execution is enforced even if CLI order differs.
    return tuple(season for season in DEVELOPMENT_SEASONS if season in requested)


def _seeds_for_fold(base_seed: int, season: int) -> SeedSpec:
    fold_offset = (season - DEVELOPMENT_SEASONS[0]) * SEED_STRIDE_PER_FOLD
    return SeedSpec(
        initialization_seed=base_seed + fold_offset + SEED_OFFSETS["initialization_seed"],
        training_seed=base_seed + fold_offset + SEED_OFFSETS["training_seed"],
        loader_seed=base_seed + fold_offset + SEED_OFFSETS["loader_seed"],
    )


def _read_selection(path: Path) -> EpochSelectionRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EpochSelectionRecord.from_dict(payload)


def _verify_selection_for_fold(
    selection: EpochSelectionRecord,
    *,
    fold: SeasonFold,
    optimization: OptimizationProcedure,
    seeds: SeedSpec,
) -> None:
    inner = fold.inner_stop_split
    if inner is None:
        raise ValueError(f"{fold.name} has no inner stop split")
    if selection.outer_fold_name != fold.name:
        raise ValueError("selection record belongs to a different outer fold")
    if selection.outer_validation_season != fold.validation_season:
        raise ValueError("selection record outer season differs from fold")
    if selection.inner_partition_name != inner.name:
        raise ValueError("selection record inner partition differs from fold")
    if selection.fit_row_id_sha256 != row_id_sha256(inner.train_row_ids):
        raise ValueError("selection record inner-train row identity differs from source")
    if selection.stop_row_id_sha256 != row_id_sha256(inner.validation_row_ids):
        raise ValueError("selection record inner-stop row identity differs from source")
    if selection.optimization != optimization.to_dict():
        raise ValueError("selection record optimization procedure differs from CLI")
    if selection.seeds != seeds.to_dict():
        raise ValueError("selection record seeds differ from deterministic fold seeds")


def _verify_checkpoint_for_resume(
    checkpoint_dir: Path,
    *,
    fold: SeasonFold,
    selection: EpochSelectionRecord,
    optimization: OptimizationProcedure,
    seeds: SeedSpec,
    model_config: ContextGatedModelConfig,
    execution_device: torch.device,
) -> dict[str, Any]:
    """Hash-load a checkpoint and cross-check its learning-stage identity."""

    loaded = load_learning_checkpoint(checkpoint_dir, device="cpu")
    manifest = loaded.manifest
    record = loaded.training_record
    if manifest.get("contains_outer_validation_predictions") is not False:
        raise ValueError("resumed checkpoint does not prohibit outer-validation predictions")
    if manifest.get("contains_outer_validation_metrics") is not False:
        raise ValueError("resumed checkpoint does not prohibit outer-validation metrics")
    if record.get("outer_fold_name") != fold.name:
        raise ValueError(
            "resumed checkpoint fold mismatch: "
            f"expected {fold.name!r}, found {record.get('outer_fold_name')!r}"
        )
    if record.get("selected_epoch") != selection.selected_epoch:
        raise ValueError("resumed checkpoint selected epoch differs from selection record")
    if record.get("selection_record") != selection.to_dict():
        raise ValueError("resumed checkpoint embeds a different selection record")
    if record.get("optimization") != optimization.to_dict():
        raise ValueError("resumed checkpoint optimization procedure differs from CLI")
    if record.get("seeds") != seeds.to_dict():
        raise ValueError("resumed checkpoint seeds differ from deterministic fold seeds")
    saved_model_config = record.get("model_config")
    if (
        not isinstance(saved_model_config, Mapping)
        or saved_model_config.get("architecture") != "context_gated_residual_v1"
        or saved_model_config.get("model") != model_config.to_dict()
    ):
        raise ValueError("resumed checkpoint model config differs from V1 architecture")
    if record.get("outer_train_row_id_sha256") != row_id_sha256(fold.train_row_ids):
        raise ValueError("resumed checkpoint outer-train row identity differs from source")
    if record.get("source_row_id_sha256") != row_id_sha256(fold._source.row_ids):
        raise ValueError("resumed checkpoint source row identity differs from input")
    if record.get("preprocessor_fingerprint") != manifest.get(
        "preprocessor_state_sha256"
    ):
        raise ValueError("resumed checkpoint preprocessor fingerprints disagree")
    if record.get("outer_validation_rows_accessed") is not False:
        raise ValueError("resumed checkpoint does not certify untouched outer validation")
    if record.get("contains_outer_validation_predictions") is not False:
        raise ValueError("resumed training record contains outer-validation predictions")
    if record.get("contains_outer_validation_metrics") is not False:
        raise ValueError("resumed training record contains outer-validation metrics")
    if record.get("outer_validation_access") != {
        "transform_calls": 0,
        "predict_calls": 0,
        "metric_calls": 0,
    }:
        raise ValueError("resumed checkpoint has a nonzero outer-validation access audit")
    if record.get("anchor_accessed") is not False:
        raise ValueError("resumed checkpoint accessed the 2024 anchor")
    if record.get("diagnostic_accessed") is not False:
        raise ValueError("resumed checkpoint accessed a diagnostic fold")
    environment = record.get("environment")
    if not isinstance(environment, Mapping) or environment.get("device") != str(
        execution_device
    ):
        raise ValueError("resumed checkpoint execution device differs from this run")
    return {
        "checkpoint_manifest_sha256": _sha256_file(checkpoint_dir / "manifest.json"),
        "preprocessor_state_sha256": manifest["preprocessor_state_sha256"],
    }


def _prepare_fold_directory(fold_dir: Path) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"selection_record.json", "checkpoint"}
    unexpected = sorted(path.name for path in fold_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise FileExistsError(
            f"unexpected artifacts in fold directory {fold_dir}: {unexpected}"
        )


def run_learning(
    *,
    input_path: Path,
    output_dir: Path,
    fold_seasons: Sequence[int],
    optimization: OptimizationProcedure,
    base_seed: int,
    device: str | torch.device,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Execute epoch selection and fixed-epoch outer-train refits only."""

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    selected_seasons = _normalise_fold_seasons(fold_seasons)
    target_device = _resolve_device(str(device))
    if target_device.type == "cpu" and optimization.use_amp:
        raise ValueError("AMP is disabled for the fixed CPU learning procedure")

    source_sha256 = _sha256_file(input_path)
    frame = pd.read_csv(input_path, low_memory=False)
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("train.csv column order does not match the official training schema")

    plan = build_validation_plan(frame, include_recent_only=False)
    development_by_season = {
        fold.validation_season: fold for fold in plan.development_folds
    }
    if tuple(development_by_season) != DEVELOPMENT_SEASONS:
        raise AssertionError("development-fold contract changed unexpectedly")
    if plan.anchor_fold.validation_season in selected_seasons:
        raise AssertionError("the 2024 anchor must not enter learning-stage execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists() and not resume:
        raise FileExistsError(
            "learning manifest already exists; pass --resume to verify completed artifacts"
        )
    model_config = ContextGatedModelConfig()
    fold_summaries: list[dict[str, Any]] = []

    for season in selected_seasons:
        fold = development_by_season[season]
        seeds = _seeds_for_fold(base_seed, season)
        fold_dir = output_dir / fold.name
        selection_path = fold_dir / "selection_record.json"
        checkpoint_dir = fold_dir / "checkpoint"
        _prepare_fold_directory(fold_dir)

        if selection_path.exists():
            if not resume:
                raise FileExistsError(
                    f"selection record already exists; pass --resume to verify/reuse: {selection_path}"
                )
            selection = _read_selection(selection_path)
            _verify_selection_for_fold(
                selection,
                fold=fold,
                optimization=optimization,
                seeds=seeds,
            )
            print(f"[{fold.name}] verified saved inner selection", flush=True)
        else:
            if checkpoint_dir.exists():
                raise FileExistsError(
                    f"checkpoint exists without its selection record: {checkpoint_dir}"
                )
            print(f"[{fold.name}] selecting epoch on inner train/stop", flush=True)
            selection = select_epoch_on_inner_stop(
                frame,
                fold,
                model_config=model_config,
                optimization=optimization,
                seeds=seeds,
                device=target_device,
                verbose=verbose,
            )
            _verify_selection_for_fold(
                selection,
                fold=fold,
                optimization=optimization,
                seeds=seeds,
            )
            _write_json_exclusive(selection_path, _json_payload(selection))
            print(
                f"[{fold.name}] selected_epoch={selection.selected_epoch}; "
                "inner model discarded",
                flush=True,
            )

        if checkpoint_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"checkpoint already exists; pass --resume to verify/skip: {checkpoint_dir}"
                )
            checkpoint_summary = _verify_checkpoint_for_resume(
                checkpoint_dir,
                fold=fold,
                selection=selection,
                optimization=optimization,
                seeds=seeds,
                model_config=model_config,
                execution_device=target_device,
            )
            print(f"[{fold.name}] checkpoint hashes verified; refit skipped", flush=True)
        else:
            print(
                f"[{fold.name}] fitting fresh outer-train model for "
                f"{selection.selected_epoch} fixed epoch(s)",
                flush=True,
            )
            fitted = refit_outer_fixed_epochs(
                frame,
                fold,
                selection,
                model_config=model_config,
                optimization=optimization,
                seeds=seeds,
                device=target_device,
                verbose=verbose,
            )
            save_learning_checkpoint(
                checkpoint_dir,
                preprocessor=fitted.preprocessor,
                model=fitted.model,
                training_record=fitted.training_record,
            )
            checkpoint_summary = _verify_checkpoint_for_resume(
                checkpoint_dir,
                fold=fold,
                selection=selection,
                optimization=optimization,
                seeds=seeds,
                model_config=model_config,
                execution_device=target_device,
            )
            print(f"[{fold.name}] checkpoint saved and hash-verified", flush=True)

        fold_summaries.append(
            {
                "fold": fold.name,
                "outer_train_seasons": list(fold.train_seasons),
                "reserved_outer_validation_season": fold.validation_season,
                "outer_train_rows": fold.train_row_count,
                "outer_validation_rows_accessed": False,
                "selected_epoch": selection.selected_epoch,
                "scheduler_horizon_epochs": optimization.scheduler_horizon_epochs,
                "selection_record_sha256": _sha256_file(selection_path),
                "checkpoint_directory": str(checkpoint_dir),
                "execution_status": "checkpoint_complete_and_verified",
                **checkpoint_summary,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scope": (
            "learning only: inner epoch selection and fresh fixed-epoch outer-train refit; "
            "no outer validation prediction or performance evaluation"
        ),
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": source_sha256,
            "rows": int(len(frame)),
        },
        "architecture": {
            "name": "context_gated_residual_v1",
            "config": model_config.to_dict(),
        },
        "optimization": optimization.to_dict(),
        "execution_device": str(target_device),
        "seed_policy": {
            "base_seed": base_seed,
            "fold_stride": SEED_STRIDE_PER_FOLD,
            "field_offsets": SEED_OFFSETS,
            "by_fold": {
                development_by_season[season].name: _seeds_for_fold(
                    base_seed, season
                ).to_dict()
                for season in selected_seasons
            },
        },
        "folds": fold_summaries,
        "stage_boundary": {
            "allowed_development_validation_seasons": list(DEVELOPMENT_SEASONS),
            "outer_validation_rows_accessed": False,
            "outer_validation_predictions_created": False,
            "outer_validation_metrics_computed": False,
            "anchor_2024_trained_or_evaluated": False,
            "recent_only_diagnostic_executed": False,
            "hyperparameters_tuned": False,
            "calibration_or_ensemble_executed": False,
            "inference_or_submission_executed": False,
        },
        "outputs": [
            f"{summary['fold']}/selection_record.json" for summary in fold_summaries
        ]
        + [f"{summary['fold']}/checkpoint/" for summary in fold_summaries]
        + ["manifest.json"],
    }
    _write_or_verify_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(DEVELOPMENT_SEASONS),
        help="Development validation seasons only (allowed: 2022 2023).",
    )
    parser.add_argument("--max-selection-epochs", type=int, default=8)
    parser.add_argument("--scheduler-horizon-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP. The fixed CPU default leaves this disabled.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse selection records; hash-load complete checkpoints and skip them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch messages (fold-level progress is still printed).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    optimization = OptimizationProcedure(
        max_selection_epochs=args.max_selection_epochs,
        scheduler_horizon_epochs=args.scheduler_horizon_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        brier_weight=args.brier_weight,
        patience=args.patience,
        min_delta=args.min_delta,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=args.num_workers,
        use_amp=args.amp,
        cpu_threads=args.cpu_threads,
    )
    manifest = run_learning(
        input_path=args.input,
        output_dir=args.output_dir,
        fold_seasons=args.folds,
        optimization=optimization,
        base_seed=args.base_seed,
        device=args.device,
        resume=args.resume,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "scope": manifest["scope"],
                "folds": [fold["fold"] for fold in manifest["folds"]],
                "selected_epochs": {
                    fold["fold"]: fold["selected_epoch"]
                    for fold in manifest["folds"]
                },
                "outer_validation_metrics_computed": False,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
