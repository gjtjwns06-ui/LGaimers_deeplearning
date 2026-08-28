"""Leakage-safe regime ablation orchestration for the 2022/2023 folds.

This module deliberately sits on top of the already audited learning and
evaluation APIs.  It does not implement a second trainer.  The only model
choices are the three pre-registered ``game_type`` policies declared in
``CANDIDATE_SPECS``; the 2024 anchor is rejected during the input preflight and
is never represented by a fold in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .constants import ID_COL, TARGET_COL, TRAIN_COLUMNS
from .evaluation import (
    EvaluationConfig,
    aggregate_development_evaluations,
    evaluate_outer_validation,
    probability_metrics,
)
from .learning import (
    EpochSelectionRecord,
    OptimizationProcedure,
    SeedSpec,
    refit_outer_fixed_epochs,
    row_id_sha256,
    select_epoch_on_inner_stop,
)
from .learning_artifacts import load_learning_checkpoint, save_learning_checkpoint
from .model import ContextGatedModelConfig, ContextGatedResidualModel
from .validation import SeasonFold, build_development_plan


SCHEMA_VERSION = 1
TUNING_SEASONS = (2019, 2020, 2021, 2022, 2023)
DEVELOPMENT_SEASONS = (2022, 2023)
DEVELOPMENT_SNAPSHOT_ARTIFACT = "development_snapshot_v2"
DEVELOPMENT_SNAPSHOT_POLICY = {
    "included_seasons": list(TUNING_SEASONS),
    "excluded_seasons": [2024],
    "excluded_row_fields_lexically_parsed": ["row_id", "season"],
    "excluded_target_lexically_parsed": False,
    "excluded_target_semantically_parsed": False,
    "byte_dialect_contract": (
        "official header order; unquoted ASCII row_id/season prefix; "
        "one official record per physical line"
    ),
}
BASE_SEED = 2026
SEED_STRIDE_PER_FOLD = 100
HARD_GATE_2022_TOLERANCE = 0.0025
HARD_GATE_2023_R_TOLERANCE = 0.0005


@dataclass(frozen=True)
class CandidateSpec:
    """One immutable, named regime-policy ablation."""

    candidate_id: str
    label: str
    game_type_mode: str

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        label = str(self.label).strip()
        mode = str(self.game_type_mode).strip().lower()
        if not candidate_id or not label:
            raise ValueError("candidate id and label must be non-empty")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in candidate_id):
            raise ValueError("candidate id must contain only lowercase letters, digits, and underscores")
        # ContextGatedModelConfig remains the authoritative mode validator.
        ContextGatedModelConfig(game_type_mode=mode)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "game_type_mode", mode)

    def model_config(self) -> ContextGatedModelConfig:
        return ContextGatedModelConfig(game_type_mode=self.game_type_mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "game_type_mode": self.game_type_mode,
            "model_config": self.model_config().to_dict(),
        }


CANDIDATE_SPECS = (
    CandidateSpec("c1_no_bias", "C1 no_bias", "no_bias"),
    CandidateSpec("c2_neutral", "C2 neutral", "neutral"),
    CandidateSpec(
        "c3_bounded_history_residual",
        "C3 bounded_history_residual",
        "bounded_history_residual",
    ),
)


@dataclass(frozen=True)
class BaselineReference:
    learning_directory: str
    evaluation_directory: str
    learning_manifest_sha256: str
    evaluation_manifest_sha256: str
    source_input_sha256: str
    checkpoint_manifest_sha256: dict[str, str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise ValueError(f"payload already contains embedded hash field {field!r}")
    result = dict(payload)
    result[field] = _canonical_sha256(result)
    return result


def _verify_embedded_hash(payload: Mapping[str, Any], field: str) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("hashed payload must be a mapping")
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"embedded hash {field!r} is missing or malformed")
    unhashed = dict(payload)
    del unhashed[field]
    expected = _canonical_sha256(unhashed)
    if observed != expected:
        raise ValueError(f"embedded hash mismatch for {field}")
    return observed


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _serialize_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes_exclusive(path: Path, content: bytes) -> None:
    """Atomically create ``path`` and never replace a committed artifact."""

    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"artifact appeared while writing: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes_exclusive(path, _serialize_json(payload))


def _write_or_verify_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = dict(payload)
    if path.exists():
        if _read_json(path) != expected:
            raise FileExistsError(f"immutable artifact differs from this run: {path}")
        return
    _atomic_write_json_exclusive(path, expected)


def _safe_inventory_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe manifest path: {relative!r}")
    resolved = (root / candidate).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"manifest path escapes artifact root: {relative!r}")
    return resolved


def _tree_file_hashes(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _remove_known_temporary_tree(path: Path, *, expected_parent: Path, prefix: str) -> None:
    """Remove only an orchestrator-owned interrupted staging tree."""

    resolved = path.resolve()
    parent = expected_parent.resolve()
    if resolved.parent != parent or not resolved.name.startswith(prefix):
        raise RuntimeError(f"refusing to remove an unverified temporary path: {resolved}")
    if not resolved.exists():
        return
    for child in sorted(resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    resolved.rmdir()


def _verify_file_inventory(
    root: Path,
    inventory: Mapping[str, Any],
    *,
    manifest_name: str = "manifest.json",
) -> None:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError(f"file hash inventory is missing: {root}")
    expected_names = set()
    for relative, expected_hash in inventory.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("file hash inventory is malformed")
        path = _safe_inventory_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f"artifact hash mismatch: {relative}")
        expected_names.add(relative)
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != manifest_name
    }
    if actual_names != expected_names:
        raise ValueError(
            f"artifact inventory differs from files on disk: {root}; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )


def validate_tuning_seasons(values: Sequence[Any] | pd.Series | np.ndarray) -> tuple[int, ...]:
    """Require exactly 2019--2023 and reject 2024 before any tuning work."""

    numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    if numeric.size == 0 or not np.isfinite(numeric).all() or not np.equal(
        numeric, np.floor(numeric)
    ).all():
        raise ValueError("season must contain finite integer years")
    observed = tuple(sorted(int(value) for value in np.unique(numeric.astype(np.int64))))
    if 2024 in observed:
        raise ValueError("2024 is a sealed anchor and is prohibited from regime tuning input")
    if observed != TUNING_SEASONS:
        raise ValueError(
            "regime tuning input must contain exactly seasons 2019..2023; "
            f"observed={observed}"
        )
    return observed


def load_tuning_frame(input_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Season-preflight a CSV, then load the complete development-only frame."""

    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"tuning input does not exist: {source}")
    # This is intentionally the first CSV read.  A file containing 2024 is
    # rejected before its remaining columns, baseline artifacts, or outputs are touched.
    season_only = pd.read_csv(source, usecols=["season"], low_memory=False)
    observed = validate_tuning_seasons(season_only["season"])
    frame = pd.read_csv(source, low_memory=False)
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("tuning CSV column order differs from the official training schema")
    validate_tuning_seasons(frame["season"])
    if not frame[ID_COL].is_unique:
        raise ValueError("tuning input row_id values must be unique")
    provenance = {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "rows": int(len(frame)),
        "seasons": list(observed),
        "sha256": _sha256_file(source),
        "source_row_id_sha256": row_id_sha256(
            frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
    }
    snapshot_manifest_path = source.with_suffix(".manifest.json")
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError(
            f"development snapshot v2 manifest is required: {snapshot_manifest_path}"
        )
    snapshot_manifest = _read_json(snapshot_manifest_path)
    contract_hash = _verify_embedded_hash(snapshot_manifest, "contract_sha256")
    if snapshot_manifest.get("artifact") != DEVELOPMENT_SNAPSHOT_ARTIFACT:
        raise ValueError("input is not governed by the development_snapshot_v2 contract")
    policy = snapshot_manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("development snapshot v2 policy is missing")
    if dict(policy) != DEVELOPMENT_SNAPSHOT_POLICY:
        raise ValueError("development snapshot v2 policy differs from the exact contract")
    output_record = snapshot_manifest.get("output")
    source_record = snapshot_manifest.get("source")
    if not isinstance(output_record, Mapping) or not isinstance(source_record, Mapping):
        raise ValueError("development snapshot manifest provenance is missing")
    if output_record.get("sha256") != provenance["sha256"]:
        raise ValueError("development snapshot CSV hash differs from its manifest")
    if output_record.get("rows") != provenance["rows"]:
        raise ValueError("development snapshot row count differs from its manifest")
    if output_record.get("season_counts") != {
        str(season): int((frame["season"] == season).sum())
        for season in TUNING_SEASONS
    }:
        raise ValueError("development snapshot season counts differ from its manifest")
    if not isinstance(source_record.get("sha256"), str):
        raise ValueError("development snapshot source hash is missing")
    provenance["snapshot_manifest"] = {
        "artifact": DEVELOPMENT_SNAPSHOT_ARTIFACT,
        "path": str(snapshot_manifest_path.resolve()),
        "file_sha256": _sha256_file(snapshot_manifest_path),
        "contract_sha256": contract_hash,
        "source_sha256": str(source_record["sha256"]),
        "policy": dict(policy),
    }
    return frame, provenance


def build_tuning_folds(frame: pd.DataFrame) -> tuple[SeasonFold, SeasonFold]:
    validate_tuning_seasons(frame["season"])
    plan = build_development_plan(frame)
    folds = tuple(plan.development_folds)
    for fold in folds:
        if fold.validation_season not in DEVELOPMENT_SEASONS or fold.role != "rolling":
            raise AssertionError("development fold construction escaped its contract")
        if fold.inner_stop_split is None:
            raise AssertionError("each tuning fold must have an inner stop partition")
    return folds  # type: ignore[return-value]


def _seed_for_fold(base_seed: int, season: int) -> SeedSpec:
    if isinstance(base_seed, bool) or int(base_seed) != base_seed or int(base_seed) < 0:
        raise ValueError("base_seed must be a non-negative integer")
    offset = (season - DEVELOPMENT_SEASONS[0]) * SEED_STRIDE_PER_FOLD
    return SeedSpec(
        initialization_seed=int(base_seed) + offset,
        training_seed=int(base_seed) + offset + 1,
        loader_seed=int(base_seed) + offset + 2,
    )


def _architecture_hash(config: ContextGatedModelConfig) -> str:
    return _canonical_sha256(
        {
            "architecture": ContextGatedResidualModel.architecture_name,
            "model": config.to_dict(),
        }
    )


def _verify_selection_record(
    record: EpochSelectionRecord,
    *,
    fold: SeasonFold,
    config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    seeds: SeedSpec,
) -> None:
    inner = fold.inner_stop_split
    assert inner is not None
    checks = {
        "outer_fold_name": fold.name,
        "outer_validation_season": fold.validation_season,
        "inner_partition_name": inner.name,
        "fit_row_id_sha256": row_id_sha256(inner.train_row_ids),
        "stop_row_id_sha256": row_id_sha256(inner.validation_row_ids),
        "architecture_procedure_sha256": _architecture_hash(config),
    }
    for name, expected in checks.items():
        if getattr(record, name) != expected:
            raise ValueError(f"selection record {name} differs from candidate plan")
    if record.optimization != optimization.to_dict():
        raise ValueError("selection optimization differs from candidate plan")
    if record.seeds != seeds.to_dict():
        raise ValueError("selection seeds differ from candidate plan")


def _verify_candidate_checkpoint(
    checkpoint_dir: Path,
    *,
    frame: pd.DataFrame,
    fold: SeasonFold,
    selection: EpochSelectionRecord,
    config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    seeds: SeedSpec,
) -> dict[str, Any]:
    loaded = load_learning_checkpoint(checkpoint_dir, device="cpu")
    record = loaded.training_record
    expected = {
        "outer_fold_name": fold.name,
        "outer_validation_season": fold.validation_season,
        "outer_train_seasons": list(fold.train_seasons),
        "outer_train_rows": fold.train_row_count,
        "outer_train_row_id_sha256": row_id_sha256(fold.train_row_ids),
        "source_row_id_sha256": row_id_sha256(
            frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
        "selected_epoch": selection.selected_epoch,
        "optimization": optimization.to_dict(),
        "seeds": seeds.to_dict(),
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f"candidate checkpoint {name} differs from plan")
    if record.get("selection_record") != selection.to_dict():
        raise ValueError("candidate checkpoint embeds a different selection record")
    model_config = record.get("model_config")
    if not isinstance(model_config, Mapping) or model_config.get("model") != config.to_dict():
        raise ValueError("candidate checkpoint model config differs from candidate")
    if model_config.get("architecture") != ContextGatedResidualModel.architecture_name:
        raise ValueError("candidate checkpoint architecture is unexpected")
    for name in (
        "outer_validation_rows_accessed",
        "contains_outer_validation_predictions",
        "contains_outer_validation_metrics",
        "anchor_accessed",
        "diagnostic_accessed",
    ):
        if record.get(name) is not False:
            raise ValueError(f"candidate checkpoint scope flag must be false: {name}")
    access = record.get("outer_validation_access")
    if access != {"transform_calls": 0, "predict_calls": 0, "metric_calls": 0}:
        raise ValueError("candidate checkpoint outer-validation audit is nonzero")
    report = record.get("parameter_report")
    if not isinstance(report, Mapping) or not isinstance(report.get("total_parameters"), int):
        raise ValueError("candidate checkpoint parameter report is missing")
    return {
        "checkpoint_manifest_sha256": _sha256_file(checkpoint_dir / "manifest.json"),
        "model_state_sha256": loaded.manifest["files_sha256"]["model_state.npz"],
        "preprocessor_fingerprint": loaded.manifest["preprocessor_state_sha256"],
        "parameter_count": int(report["total_parameters"]),
    }


def _select_unique_slice(
    table: pd.DataFrame,
    *,
    season: int,
    slice_name: str,
    value: str,
) -> dict[str, float | int]:
    selected = table[
        table["validation_season"].eq(season)
        & table["slice"].eq(slice_name)
        & table["value"].astype(str).eq(value)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one slice row for season={season}, slice={slice_name}, value={value}"
        )
    row = selected.iloc[0]
    result: dict[str, float | int] = {"rows": int(row["rows"])}
    for name in ("target_rate", "prediction_mean", "brier"):
        numeric = float(row[name])
        if not np.isfinite(numeric):
            raise ValueError(f"slice metric is not finite: {name}")
        result[name] = numeric
    result["absolute_bias"] = abs(
        float(result["prediction_mean"]) - float(result["target_rate"])
    )
    return result


def verify_baseline_reference(
    *,
    frame: pd.DataFrame,
    folds: Sequence[SeasonFold],
    learning_dir: str | Path,
    evaluation_dir: str | Path,
) -> BaselineReference:
    """Strictly hash-verify the existing V1 development baseline."""

    learning_root = Path(learning_dir).resolve()
    evaluation_root = Path(evaluation_dir).resolve()
    learning_manifest_path = learning_root / "manifest.json"
    evaluation_manifest_path = evaluation_root / "manifest.json"
    learning_manifest = _read_json(learning_manifest_path)
    evaluation_manifest = _read_json(evaluation_manifest_path)
    boundary = learning_manifest.get("stage_boundary")
    if not isinstance(boundary, Mapping) or boundary.get(
        "anchor_2024_trained_or_evaluated"
    ) is not False:
        raise ValueError("baseline learning manifest does not certify a sealed anchor")
    eval_boundary = evaluation_manifest.get("stage_boundary")
    if not isinstance(eval_boundary, Mapping) or eval_boundary.get(
        "anchor_2024_predicted_or_scored"
    ) is not False:
        raise ValueError("baseline evaluation manifest does not certify a sealed anchor")
    if evaluation_manifest.get("phase") != "development_outer_validation_evaluation":
        raise ValueError("baseline evaluation phase is unexpected")
    inventory = evaluation_manifest.get("files_sha256")
    _verify_file_inventory(evaluation_root, inventory)
    if any("2024" in str(relative) for relative in inventory):
        raise ValueError("baseline development evaluation inventory references 2024")
    if evaluation_manifest.get("learning_provenance", {}).get(
        "manifest_sha256"
    ) != _sha256_file(learning_manifest_path):
        raise ValueError("baseline evaluation points to a different learning manifest")

    learning_folds = learning_manifest.get("folds")
    evaluation_folds = evaluation_manifest.get("folds")
    if not isinstance(learning_folds, list) or not isinstance(evaluation_folds, list):
        raise ValueError("baseline fold manifests are missing")
    learning_by_name = {item.get("fold"): item for item in learning_folds}
    evaluation_by_name = {item.get("fold"): item for item in evaluation_folds}
    expected_names = {fold.name for fold in folds}
    if set(learning_by_name) != expected_names or set(evaluation_by_name) != expected_names:
        raise ValueError("baseline must contain exactly the 2022/2023 development folds")

    checkpoint_hashes: dict[str, str] = {}
    for fold in folds:
        learning_record = learning_by_name[fold.name]
        evaluation_record = evaluation_by_name[fold.name]
        checkpoint_dir = learning_root / fold.name / "checkpoint"
        checkpoint_hash = _sha256_file(checkpoint_dir / "manifest.json")
        if checkpoint_hash != learning_record.get("checkpoint_manifest_sha256"):
            raise ValueError(f"baseline checkpoint manifest hash mismatch: {fold.name}")
        if checkpoint_hash != evaluation_record.get("checkpoint_manifest_sha256"):
            raise ValueError(f"baseline evaluation checkpoint hash mismatch: {fold.name}")
        selection_path = learning_root / fold.name / "selection_record.json"
        if _sha256_file(selection_path) != learning_record.get("selection_record_sha256"):
            raise ValueError(f"baseline selection record hash mismatch: {fold.name}")
        loaded = load_learning_checkpoint(checkpoint_dir, device="cpu")
        record = loaded.training_record
        if record.get("outer_train_seasons") != list(fold.train_seasons):
            raise ValueError(f"baseline train seasons differ: {fold.name}")
        if record.get("outer_train_row_id_sha256") != row_id_sha256(fold.train_row_ids):
            raise ValueError(f"baseline train row identity differs: {fold.name}")
        selection = record.get("selection_record")
        inner = fold.inner_stop_split
        assert inner is not None
        if not isinstance(selection, Mapping):
            raise ValueError(f"baseline selection is missing: {fold.name}")
        if selection.get("fit_row_id_sha256") != row_id_sha256(inner.train_row_ids):
            raise ValueError(f"baseline inner-fit rows differ: {fold.name}")
        if selection.get("stop_row_id_sha256") != row_id_sha256(inner.validation_row_ids):
            raise ValueError(f"baseline inner-stop rows differ: {fold.name}")
        if evaluation_record.get("validation_row_id_sha256") != row_id_sha256(
            fold.validation_row_ids
        ):
            raise ValueError(f"baseline validation rows differ: {fold.name}")
        checkpoint_hashes[fold.name] = checkpoint_hash

    slice_path = evaluation_root / "slice_metrics.csv"
    slices = pd.read_csv(slice_path)
    overall_2022 = float(evaluation_by_name["rolling_origin_2022"]["metrics"]["brier"])
    overall_2023 = float(evaluation_by_name["rolling_origin_2023"]["metrics"]["brier"])
    f_2023 = _select_unique_slice(
        slices, season=2023, slice_name="game_type", value="F"
    )
    r_2023 = _select_unique_slice(
        slices, season=2023, slice_name="game_type", value="R"
    )
    metrics = {
        "fold_brier": {"2022": overall_2022, "2023": overall_2023},
        "max_fold_brier": max(overall_2022, overall_2023),
        "game_type_2023": {"F": f_2023, "R": r_2023},
    }
    for value in (
        overall_2022,
        overall_2023,
        float(f_2023["brier"]),
        float(r_2023["brier"]),
    ):
        if not np.isfinite(value):
            raise ValueError("baseline metrics contain a non-finite value")
    source_input = learning_manifest.get("input")
    if not isinstance(source_input, Mapping) or not isinstance(
        source_input.get("sha256"), str
    ):
        raise ValueError("baseline source provenance is missing")
    return BaselineReference(
        learning_directory=str(learning_root),
        evaluation_directory=str(evaluation_root),
        learning_manifest_sha256=_sha256_file(learning_manifest_path),
        evaluation_manifest_sha256=_sha256_file(evaluation_manifest_path),
        source_input_sha256=str(source_input["sha256"]),
        checkpoint_manifest_sha256=checkpoint_hashes,
        metrics=metrics,
    )


def evaluate_hard_gates(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the four pre-registered, non-configurable promotion gates."""

    candidate_fold = candidate_metrics["fold_brier"]
    baseline_fold = baseline_metrics["fold_brier"]
    candidate_f = candidate_metrics["game_type_2023"]["F"]
    candidate_r = candidate_metrics["game_type_2023"]["R"]
    baseline_f = baseline_metrics["game_type_2023"]["F"]
    baseline_r = baseline_metrics["game_type_2023"]["R"]
    checks = {
        "brier_2022_within_baseline_plus_0_0025": {
            "value": float(candidate_fold["2022"]),
            "threshold": float(baseline_fold["2022"]) + HARD_GATE_2022_TOLERANCE,
            "operator": "<=",
        },
        "brier_2023_r_within_baseline_plus_0_0005": {
            "value": float(candidate_r["brier"]),
            "threshold": float(baseline_r["brier"]) + HARD_GATE_2023_R_TOLERANCE,
            "operator": "<=",
        },
        "brier_2023_f_strictly_improves": {
            "value": float(candidate_f["brier"]),
            "threshold": float(baseline_f["brier"]),
            "operator": "<",
        },
        "absolute_bias_2023_f_strictly_improves": {
            "value": float(candidate_f["absolute_bias"]),
            "threshold": float(baseline_f["absolute_bias"]),
            "operator": "<",
        },
    }
    for check in checks.values():
        value = float(check["value"])
        threshold = float(check["threshold"])
        if not np.isfinite(value) or not np.isfinite(threshold):
            raise ValueError("hard-gate metric is not finite")
        check["passed"] = value <= threshold if check["operator"] == "<=" else value < threshold
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def candidate_ranking_key(
    candidate_id: str,
    candidate_metrics: Mapping[str, Any],
) -> tuple[float, float, float, float, int, str]:
    """Return the immutable lexicographic selection key."""

    fold = candidate_metrics["fold_brier"]
    f_2023 = candidate_metrics["game_type_2023"]["F"]
    key = (
        max(float(fold["2022"]), float(fold["2023"])),
        float(fold["2023"]),
        float(f_2023["absolute_bias"]),
        float(f_2023["brier"]),
        int(candidate_metrics["parameter_count"]),
        str(candidate_id),
    )
    if not all(np.isfinite(value) for value in key[:4]):
        raise ValueError("candidate ranking metric is not finite")
    if key[4] <= 0:
        raise ValueError("candidate parameter count must be positive")
    return key


def choose_candidate(
    candidate_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select only among hard-gate passers, otherwise retain the baseline."""

    ranked: list[tuple[tuple[float, float, float, float, int, str], str]] = []
    records = []
    for manifest in candidate_manifests:
        candidate_id = str(manifest["candidate"]["candidate_id"])
        metrics = manifest["metrics"]
        gates = manifest["hard_gates"]
        key = candidate_ranking_key(candidate_id, metrics)
        records.append(
            {
                "candidate_id": candidate_id,
                "hard_gates_passed": bool(gates["passed"]),
                "ranking_key": [*key[:-1], key[-1]],
            }
        )
        if gates["passed"]:
            ranked.append((key, candidate_id))
    ranked.sort(key=lambda item: item[0])
    records.sort(key=lambda item: item["candidate_id"])
    if not ranked:
        return {
            "selected_candidate_id": "baseline_v1",
            "outcome": "baseline_retained_no_candidate_passed_all_hard_gates",
            "eligible_ranking": [],
            "candidates": records,
        }
    return {
        "selected_candidate_id": ranked[0][1],
        "outcome": "candidate_selected",
        "eligible_ranking": [candidate_id for _, candidate_id in ranked],
        "candidates": records,
    }


def _source_inventory() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    paths = (
        *sorted(package_root.glob("*.py")),
        project_root / "scripts" / "build_development_snapshot.py",
        project_root / "scripts" / "run_regime_tuning.py",
    )
    return {
        str(path.relative_to(project_root).as_posix()): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _environment_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_execution_only": True,
    }


def _candidate_spec_payload(
    *, candidate: CandidateSpec, plan_sha256: str
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "plan_sha256": plan_sha256,
            "candidate": candidate.to_dict(),
        },
        "candidate_spec_sha256",
    )


def _read_selection(path: Path) -> EpochSelectionRecord:
    return EpochSelectionRecord.from_dict(_read_json(path))


def _run_candidate_learning(
    *,
    candidate_root: Path,
    frame: pd.DataFrame,
    folds: Sequence[SeasonFold],
    candidate: CandidateSpec,
    optimization: OptimizationProcedure,
    base_seed: int,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    config = candidate.model_config()
    records: dict[str, Any] = {}
    for fold in folds:
        seeds = _seed_for_fold(base_seed, fold.validation_season)
        fold_root = candidate_root / "learning" / fold.name
        selection_path = fold_root / "selection_record.json"
        checkpoint_dir = fold_root / "checkpoint"
        fold_root.mkdir(parents=True, exist_ok=True)
        interrupted_checkpoints = sorted(
            path
            for path in fold_root.iterdir()
            if path.is_dir() and path.name.startswith(".checkpoint.staging-")
        )
        unexpected = {
            path.name for path in fold_root.iterdir()
        }.difference(
            {
                "selection_record.json",
                "checkpoint",
                *(path.name for path in interrupted_checkpoints),
            }
        )
        if unexpected:
            raise FileExistsError(
                f"unexpected candidate learning artifacts in {fold_root}: {sorted(unexpected)}"
            )
        if interrupted_checkpoints and not resume:
            raise FileExistsError(
                f"interrupted checkpoint staging exists; use resume: {fold_root}"
            )
        if selection_path.exists():
            if not resume:
                raise FileExistsError(f"selection exists; use resume: {selection_path}")
            selection = _read_selection(selection_path)
            _verify_selection_record(
                selection,
                fold=fold,
                config=config,
                optimization=optimization,
                seeds=seeds,
            )
            if verbose:
                print(f"[{candidate.candidate_id}][{fold.name}] selection verified", flush=True)
        else:
            if checkpoint_dir.exists():
                raise FileExistsError(f"checkpoint exists without selection: {checkpoint_dir}")
            if verbose:
                print(f"[{candidate.candidate_id}][{fold.name}] inner epoch selection", flush=True)
            selection = select_epoch_on_inner_stop(
                frame,
                fold,
                model_config=config,
                optimization=optimization,
                seeds=seeds,
                device="cpu",
                verbose=verbose,
            )
            _verify_selection_record(
                selection,
                fold=fold,
                config=config,
                optimization=optimization,
                seeds=seeds,
            )
            _atomic_write_json_exclusive(selection_path, selection.to_dict())
        if interrupted_checkpoints:
            if checkpoint_dir.exists() or len(interrupted_checkpoints) != 1:
                for interrupted in interrupted_checkpoints:
                    _remove_known_temporary_tree(
                        interrupted,
                        expected_parent=fold_root,
                        prefix=".checkpoint.staging-",
                    )
            else:
                interrupted = interrupted_checkpoints[0]
                try:
                    _verify_candidate_checkpoint(
                        interrupted,
                        frame=frame,
                        fold=fold,
                        selection=selection,
                        config=config,
                        optimization=optimization,
                        seeds=seeds,
                    )
                except Exception:
                    _remove_known_temporary_tree(
                        interrupted,
                        expected_parent=fold_root,
                        prefix=".checkpoint.staging-",
                    )
                else:
                    os.replace(interrupted, checkpoint_dir)
                    if verbose:
                        print(
                            f"[{candidate.candidate_id}][{fold.name}] "
                            "completed checkpoint staging recovered",
                            flush=True,
                        )
        if checkpoint_dir.exists():
            if not resume:
                raise FileExistsError(f"checkpoint exists; use resume: {checkpoint_dir}")
            checkpoint_record = _verify_candidate_checkpoint(
                checkpoint_dir,
                frame=frame,
                fold=fold,
                selection=selection,
                config=config,
                optimization=optimization,
                seeds=seeds,
            )
            if verbose:
                print(f"[{candidate.candidate_id}][{fold.name}] checkpoint verified", flush=True)
        else:
            if verbose:
                print(
                    f"[{candidate.candidate_id}][{fold.name}] outer refit "
                    f"for {selection.selected_epoch} epoch(s)",
                    flush=True,
                )
            fitted = refit_outer_fixed_epochs(
                frame,
                fold,
                selection,
                model_config=config,
                optimization=optimization,
                seeds=seeds,
                device="cpu",
                verbose=verbose,
            )
            save_learning_checkpoint(
                checkpoint_dir,
                preprocessor=fitted.preprocessor,
                model=fitted.model,
                training_record=fitted.training_record,
            )
            checkpoint_record = _verify_candidate_checkpoint(
                checkpoint_dir,
                frame=frame,
                fold=fold,
                selection=selection,
                config=config,
                optimization=optimization,
                seeds=seeds,
            )
        records[fold.name] = {
            "selected_epoch": selection.selected_epoch,
            "selection_record_sha256": _sha256_file(selection_path),
            **checkpoint_record,
        }
    return records


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _evaluation_summary_from_tables(
    *,
    fold_metrics: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    parameter_count: int,
) -> dict[str, Any]:
    fold_brier: dict[str, float] = {}
    for season in DEVELOPMENT_SEASONS:
        selected = fold_metrics[fold_metrics["validation_season"].eq(season)]
        if len(selected) != 1:
            raise ValueError(f"evaluation must contain one fold row for {season}")
        fold_brier[str(season)] = float(selected.iloc[0]["brier"])
    f_2023 = _select_unique_slice(
        slice_metrics, season=2023, slice_name="game_type", value="F"
    )
    r_2023 = _select_unique_slice(
        slice_metrics, season=2023, slice_name="game_type", value="R"
    )
    return {
        "fold_brier": fold_brier,
        "max_fold_brier": max(fold_brier.values()),
        "game_type_2023": {"F": f_2023, "R": r_2023},
        "parameter_count": int(parameter_count),
    }


def _nested_values_match(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-14,
) -> bool:
    """Compare persisted metric trees while tolerating CSV float round-trips."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_values_match(left[key], right[key], atol=atol) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_values_match(lvalue, rvalue, atol=atol)
            for lvalue, rvalue in zip(left, right, strict=True)
        )
    numeric_types = (int, float, np.integer, np.floating)
    if (
        isinstance(left, numeric_types)
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, numeric_types)
        and not isinstance(right, (bool, np.bool_))
    ):
        return bool(
            np.isclose(float(left), float(right), rtol=0.0, atol=atol, equal_nan=True)
        )
    return left == right


def _verify_saved_candidate_evaluation(
    evaluation_root: Path,
    *,
    candidate_learning: Mapping[str, Any],
    folds: Sequence[SeasonFold],
) -> dict[str, Any]:
    manifest = _read_json(evaluation_root / "manifest.json")
    if manifest.get("phase") != "regime_candidate_development_evaluation":
        raise ValueError("candidate evaluation phase is unexpected")
    if manifest.get("anchor_2024_evaluated") is not False:
        raise ValueError("candidate evaluation does not certify a sealed anchor")
    expected_folds = {fold.name: fold for fold in folds}
    if set(expected_folds) != {
        f"rolling_origin_{season}" for season in DEVELOPMENT_SEASONS
    }:
        raise ValueError("candidate evaluation folds differ from the development plan")
    if manifest.get("evaluated_seasons") != list(DEVELOPMENT_SEASONS):
        raise ValueError("candidate evaluation seasons must be exactly 2022/2023")
    checkpoint_map = manifest.get("checkpoint_manifest_sha256")
    if not isinstance(checkpoint_map, Mapping) or set(checkpoint_map) != set(
        expected_folds
    ):
        raise ValueError("candidate evaluation checkpoint map differs from folds")
    inventory = manifest.get("files_sha256")
    _verify_file_inventory(evaluation_root, inventory)
    if any("2024" in str(relative) for relative in inventory):
        raise ValueError("candidate development evaluation inventory references 2024")
    if set(candidate_learning) != set(expected_folds):
        raise ValueError("candidate learning records differ from evaluation folds")
    for fold_name, record in candidate_learning.items():
        if checkpoint_map.get(fold_name) != record["checkpoint_manifest_sha256"]:
            raise ValueError("candidate evaluation checkpoint provenance differs")
    fold_metrics = pd.read_csv(evaluation_root / "fold_metrics.csv")
    slice_metrics = pd.read_csv(evaluation_root / "slice_metrics.csv")
    fold_pairs = {
        (str(row.fold), int(row.validation_season))
        for row in fold_metrics.itertuples(index=False)
    }
    expected_pairs = {
        (fold.name, int(fold.validation_season)) for fold in expected_folds.values()
    }
    if len(fold_metrics) != len(expected_pairs) or fold_pairs != expected_pairs:
        raise ValueError("candidate fold metrics are not exactly the 2022/2023 folds")
    parameter_count = max(
        int(record["parameter_count"]) for record in candidate_learning.values()
    )
    summary = _evaluation_summary_from_tables(
        fold_metrics=fold_metrics,
        slice_metrics=slice_metrics,
        parameter_count=parameter_count,
    )
    if not _nested_values_match(manifest.get("metrics"), summary):
        raise ValueError("candidate evaluation summary differs from saved tables")
    predictions = pd.read_csv(evaluation_root / "development_oof_predictions.csv")
    if set(pd.to_numeric(predictions["season"], errors="raise").astype(int)) != set(
        DEVELOPMENT_SEASONS
    ):
        raise ValueError("candidate OOF predictions contain unexpected seasons")
    expected_ids = np.concatenate(
        [
            fold.validation_row_ids
            for fold in sorted(expected_folds.values(), key=lambda item: item.validation_season)
        ]
    )
    observed_ids = predictions[ID_COL].to_numpy(dtype=object, copy=False)
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError("candidate OOF prediction rows differ from development folds")
    for fold in expected_folds.values():
        fold_root = evaluation_root / fold.name
        fold_record = _read_json(fold_root / "evaluation_record.json")
        if (
            fold_record.get("fold") != fold.name
            or fold_record.get("validation_season") != fold.validation_season
            or fold_record.get("validation_rows") != fold.validation_row_count
            or fold_record.get("validation_row_id_sha256")
            != row_id_sha256(fold.validation_row_ids)
            or fold_record.get("anchor_2024_evaluated") is not False
        ):
            raise ValueError(f"candidate evaluation record differs from fold: {fold.name}")
        fold_predictions = pd.read_csv(fold_root / "predictions.csv")
        if not np.array_equal(
            fold_predictions[ID_COL].to_numpy(dtype=object, copy=False),
            fold.validation_row_ids,
        ):
            raise ValueError(f"candidate fold predictions differ: {fold.name}")
    recalculated = probability_metrics(
        predictions["y_true"].to_numpy(dtype=np.float64),
        predictions["prediction"].to_numpy(dtype=np.float64),
    )
    saved_aggregate = _read_json(evaluation_root / "aggregate_metrics.json")
    for name in ("brier", "target_rate", "prediction_mean", "rows"):
        if not np.isclose(
            float(recalculated[name]),
            float(saved_aggregate["row_pooled"][name]),
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError(f"candidate pooled evaluation does not reconcile: {name}")
    return summary


def _run_candidate_evaluation(
    *,
    candidate_root: Path,
    frame: pd.DataFrame,
    folds: Sequence[SeasonFold],
    candidate_learning: Mapping[str, Any],
    evaluation_config: EvaluationConfig,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    evaluation_root = candidate_root / "evaluation"
    if evaluation_root.exists():
        if not resume:
            raise FileExistsError(f"candidate evaluation exists; use resume: {evaluation_root}")
        return _verify_saved_candidate_evaluation(
            evaluation_root,
            candidate_learning=candidate_learning,
            folds=folds,
        )
    for interrupted in candidate_root.glob(".evaluation.tmp-*"):
        if not resume:
            raise FileExistsError(
                f"interrupted candidate evaluation exists; use resume: {interrupted}"
            )
        _remove_known_temporary_tree(
            interrupted,
            expected_parent=candidate_root,
            prefix=".evaluation.tmp-",
        )
    temporary = candidate_root / f".evaluation.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        fold_results = []
        for fold in folds:
            checkpoint_dir = candidate_root / "learning" / fold.name / "checkpoint"
            if verbose:
                print(f"[{candidate_root.name}][{fold.name}] outer evaluation", flush=True)
            result = evaluate_outer_validation(
                frame,
                fold,
                checkpoint_dir,
                config=evaluation_config,
                device="cpu",
                verbose=verbose,
            )
            fold_results.append(result)
            fold_root = temporary / fold.name
            _write_dataframe(fold_root / "predictions.csv", result.predictions)
            _write_dataframe(fold_root / "slice_metrics.csv", result.slice_metrics)
            saved_record = dict(result.evaluation_record)
            saved_record["checkpoint_directory"] = (
                f"../../learning/{fold.name}/checkpoint"
            )
            saved_record["selection_or_tuning_performed"] = True
            saved_record["orchestrated_for_regime_tuning"] = True
            _atomic_write_json_exclusive(
                fold_root / "evaluation_record.json", saved_record
            )
        aggregate = aggregate_development_evaluations(fold_results)
        _write_dataframe(temporary / "development_oof_predictions.csv", aggregate.oof_predictions)
        _write_dataframe(temporary / "fold_metrics.csv", aggregate.fold_metrics)
        _write_dataframe(temporary / "slice_metrics.csv", aggregate.slice_metrics)
        _atomic_write_json_exclusive(
            temporary / "aggregate_metrics.json", aggregate.aggregate_metrics
        )
        parameter_count = max(
            int(record["parameter_count"]) for record in candidate_learning.values()
        )
        summary = _evaluation_summary_from_tables(
            fold_metrics=aggregate.fold_metrics,
            slice_metrics=aggregate.slice_metrics,
            parameter_count=parameter_count,
        )
        inventory = _tree_file_hashes(temporary)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": "regime_candidate_development_evaluation",
            "evaluated_seasons": list(DEVELOPMENT_SEASONS),
            "anchor_2024_evaluated": False,
            "evaluation_config": evaluation_config.to_dict(),
            "checkpoint_manifest_sha256": {
                fold_name: record["checkpoint_manifest_sha256"]
                for fold_name, record in candidate_learning.items()
            },
            "metrics": summary,
            "files_sha256": inventory,
        }
        _atomic_write_json_exclusive(temporary / "manifest.json", manifest)
        _verify_saved_candidate_evaluation(
            temporary,
            candidate_learning=candidate_learning,
            folds=folds,
        )
        os.replace(temporary, evaluation_root)
    finally:
        # Do not recursively delete an unknown path.  A known temporary created
        # by this function is removed file-by-file only when publication failed.
        if temporary.exists() and temporary.parent == candidate_root:
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
    return _verify_saved_candidate_evaluation(
        evaluation_root,
        candidate_learning=candidate_learning,
        folds=folds,
    )


def _verify_candidate_directory(
    candidate_root: Path,
    *,
    plan_sha256: str,
    expected_candidate: CandidateSpec,
    folds: Sequence[SeasonFold],
    baseline: BaselineReference,
) -> dict[str, Any]:
    manifest = _read_json(candidate_root / "manifest.json")
    _verify_embedded_hash(manifest, "candidate_manifest_sha256")
    if manifest.get("phase") != "regime_candidate_complete":
        raise ValueError("candidate manifest phase is unexpected")
    if manifest.get("plan_sha256") != plan_sha256:
        raise ValueError("candidate manifest belongs to a different immutable plan")
    if manifest.get("anchor_2024_accessed") is not False:
        raise ValueError("candidate manifest does not certify a sealed anchor")
    allowed_names = {
        expected_candidate.candidate_id,
        f".{expected_candidate.candidate_id}.staging",
    }
    if candidate_root.name not in allowed_names:
        raise ValueError("candidate directory name differs from expected candidate")
    _verify_file_inventory(candidate_root, manifest.get("files_sha256"))
    spec = _read_json(candidate_root / "candidate_spec.json")
    _verify_embedded_hash(spec, "candidate_spec_sha256")
    expected_spec = _candidate_spec_payload(
        candidate=expected_candidate, plan_sha256=plan_sha256
    )
    if spec != expected_spec:
        raise ValueError("candidate spec differs from the registered candidate")
    if manifest.get("candidate") != expected_candidate.to_dict():
        raise ValueError("candidate manifest differs from the registered candidate")
    learning = manifest.get("learning")
    if not isinstance(learning, Mapping) or set(learning) != {
        fold.name for fold in folds
    }:
        raise ValueError("candidate learning summary differs from development folds")
    metrics = _verify_saved_candidate_evaluation(
        candidate_root / "evaluation",
        candidate_learning=learning,
        folds=folds,
    )
    if not _nested_values_match(manifest.get("metrics"), metrics):
        raise ValueError("candidate manifest metrics differ from saved evaluation")
    gates = evaluate_hard_gates(metrics, baseline.metrics)
    if not _nested_values_match(manifest.get("hard_gates"), gates):
        raise ValueError("candidate hard gates differ from baseline recomputation")
    ranking_key = list(
        candidate_ranking_key(expected_candidate.candidate_id, metrics)
    )
    if not _nested_values_match(manifest.get("ranking_key"), ranking_key):
        raise ValueError("candidate ranking key differs from metric recomputation")
    return manifest


def _run_one_candidate(
    *,
    candidates_root: Path,
    frame: pd.DataFrame,
    folds: Sequence[SeasonFold],
    candidate: CandidateSpec,
    plan_sha256: str,
    baseline: BaselineReference,
    optimization: OptimizationProcedure,
    evaluation_config: EvaluationConfig,
    base_seed: int,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    final_root = candidates_root / candidate.candidate_id
    staging_root = candidates_root / f".{candidate.candidate_id}.staging"
    if final_root.exists():
        if staging_root.exists():
            raise FileExistsError(
                f"both committed and staging candidate directories exist: {candidate.candidate_id}"
            )
        if not resume:
            raise FileExistsError(f"candidate already exists; use resume: {final_root}")
        return _verify_candidate_directory(
            final_root,
            plan_sha256=plan_sha256,
            expected_candidate=candidate,
            folds=folds,
            baseline=baseline,
        )
    if not staging_root.exists():
        staging_root.mkdir(parents=True, exist_ok=False)
    elif not resume:
        raise FileExistsError(f"candidate staging exists; use resume: {staging_root}")
    elif (staging_root / "manifest.json").exists():
        # The candidate was fully sealed and verified, but the process stopped
        # before the final directory rename.  Publish it without recomputing an
        # inventory that would accidentally include its own manifest.
        manifest = _verify_candidate_directory(
            staging_root,
            plan_sha256=plan_sha256,
            expected_candidate=candidate,
            folds=folds,
            baseline=baseline,
        )
        os.replace(staging_root, final_root)
        _verify_candidate_directory(
            final_root,
            plan_sha256=plan_sha256,
            expected_candidate=candidate,
            folds=folds,
            baseline=baseline,
        )
        return manifest
    spec_payload = _candidate_spec_payload(candidate=candidate, plan_sha256=plan_sha256)
    _write_or_verify_immutable_json(staging_root / "candidate_spec.json", spec_payload)
    learning = _run_candidate_learning(
        candidate_root=staging_root,
        frame=frame,
        folds=folds,
        candidate=candidate,
        optimization=optimization,
        base_seed=base_seed,
        resume=resume,
        verbose=verbose,
    )
    metrics = _run_candidate_evaluation(
        candidate_root=staging_root,
        frame=frame,
        folds=folds,
        candidate_learning=learning,
        evaluation_config=evaluation_config,
        resume=resume,
        verbose=verbose,
    )
    gates = evaluate_hard_gates(metrics, baseline.metrics)
    ranking_key = candidate_ranking_key(candidate.candidate_id, metrics)
    inventory = _tree_file_hashes(staging_root)
    manifest = _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "regime_candidate_complete",
            "plan_sha256": plan_sha256,
            "candidate": candidate.to_dict(),
            "learning": learning,
            "metrics": metrics,
            "hard_gates": gates,
            "ranking_key": [*ranking_key[:-1], ranking_key[-1]],
            "anchor_2024_accessed": False,
            "files_sha256": inventory,
        },
        "candidate_manifest_sha256",
    )
    _atomic_write_json_exclusive(staging_root / "manifest.json", manifest)
    _verify_candidate_directory(
        staging_root,
        plan_sha256=plan_sha256,
        expected_candidate=candidate,
        folds=folds,
        baseline=baseline,
    )
    os.replace(staging_root, final_root)
    return _verify_candidate_directory(
        final_root,
        plan_sha256=plan_sha256,
        expected_candidate=candidate,
        folds=folds,
        baseline=baseline,
    )


def _build_plan(
    *,
    input_provenance: Mapping[str, Any],
    baseline: BaselineReference,
    optimization: OptimizationProcedure,
    evaluation_config: EvaluationConfig,
    base_seed: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "regime_tuning_plan",
        "scope": "fixed C1/C2/C3 game_type ablations on development seasons 2022/2023 only",
        "input": dict(input_provenance),
        "allowed_input_seasons": list(TUNING_SEASONS),
        "development_validation_seasons": list(DEVELOPMENT_SEASONS),
        "candidates": [candidate.to_dict() for candidate in CANDIDATE_SPECS],
        "optimization": optimization.to_dict(),
        "evaluation_config": evaluation_config.to_dict(),
        "seed_policy": {
            "base_seed": int(base_seed),
            "fold_stride": SEED_STRIDE_PER_FOLD,
            "by_fold": {
                f"rolling_origin_{season}": _seed_for_fold(base_seed, season).to_dict()
                for season in DEVELOPMENT_SEASONS
            },
        },
        "baseline_reference": baseline.to_dict(),
        "hard_gates": {
            "2022_overall": "candidate <= baseline + 0.0025",
            "2023_game_type_R": "candidate <= baseline + 0.0005",
            "2023_game_type_F_brier": "candidate < baseline",
            "2023_game_type_F_absolute_bias": "candidate < baseline",
        },
        "selection_order": [
            "max(brier_2022,brier_2023)",
            "brier_2023",
            "absolute_bias_2023_F",
            "brier_2023_F",
            "max_parameter_count_across_2022_2023_checkpoints",
            "candidate_id",
        ],
        "execution": {
            "device": "cpu",
            "parallel_candidates": False,
            "resume": "candidate staging plus strict hashes",
        },
        "source_files_sha256": _source_inventory(),
        "environment": _environment_record(),
        "stage_boundary": {
            "anchor_2024_allowed": False,
            "calibration_or_ensemble_allowed": False,
            "test_inference_or_submission_allowed": False,
        },
    }
    return _payload_with_hash(payload, "plan_sha256")


def _verify_selection_artifact(
    payload: Mapping[str, Any],
    *,
    plan_sha256: str,
    candidate_manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    _verify_embedded_hash(payload, "selection_sha256")
    if payload.get("plan_sha256") != plan_sha256:
        raise ValueError("selection belongs to a different plan")
    if payload.get("candidate_manifest_sha256") != dict(candidate_manifest_hashes):
        raise ValueError("selection candidate provenance differs")
    return dict(payload)


def run_regime_tuning(
    *,
    input_path: str | Path,
    baseline_learning_dir: str | Path,
    baseline_evaluation_dir: str | Path,
    output_dir: str | Path,
    optimization: OptimizationProcedure | None = None,
    evaluation_config: EvaluationConfig | None = None,
    base_seed: int = BASE_SEED,
    resume: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the fixed candidates sequentially and publish an immutable selection."""

    if optimization is None:
        optimization = OptimizationProcedure()
    if evaluation_config is None:
        evaluation_config = EvaluationConfig(
            eval_batch_size=optimization.eval_batch_size,
            num_workers=0,
            cpu_threads=optimization.cpu_threads,
        )
    if not isinstance(optimization, OptimizationProcedure):
        raise TypeError("optimization must be OptimizationProcedure")
    if not isinstance(evaluation_config, EvaluationConfig):
        raise TypeError("evaluation_config must be EvaluationConfig")
    if optimization.use_amp:
        raise ValueError("regime tuning is fixed to deterministic CPU execution without AMP")
    if optimization.num_workers != 0:
        raise ValueError("regime tuning requires num_workers=0 for sequential CPU execution")

    # Input preflight occurs before output creation or baseline traversal.
    frame, input_provenance = load_tuning_frame(input_path)
    folds = build_tuning_folds(frame)
    baseline = verify_baseline_reference(
        frame=frame,
        folds=folds,
        learning_dir=baseline_learning_dir,
        evaluation_dir=baseline_evaluation_dir,
    )
    snapshot_source_sha256 = input_provenance["snapshot_manifest"]["source_sha256"]
    if snapshot_source_sha256 != baseline.source_input_sha256:
        raise ValueError(
            "development snapshot v2 and baseline artifacts reference different source CSVs"
        )
    root = Path(output_dir).resolve()
    if root.exists() and not resume:
        raise FileExistsError(f"tuning output already exists; use resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(
        input_provenance=input_provenance,
        baseline=baseline,
        optimization=optimization,
        evaluation_config=evaluation_config,
        base_seed=base_seed,
    )
    _write_or_verify_immutable_json(root / "plan.json", plan)
    plan_sha256 = _verify_embedded_hash(plan, "plan_sha256")
    candidates_root = root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for candidate in CANDIDATE_SPECS:
        manifests.append(
            _run_one_candidate(
                candidates_root=candidates_root,
                frame=frame,
                folds=folds,
                candidate=candidate,
                plan_sha256=plan_sha256,
                baseline=baseline,
                optimization=optimization,
                evaluation_config=evaluation_config,
                base_seed=base_seed,
                resume=resume,
                verbose=verbose,
            )
        )
    expected_candidate_ids = [candidate.candidate_id for candidate in CANDIDATE_SPECS]
    observed_candidate_ids = [
        str(manifest.get("candidate", {}).get("candidate_id"))
        for manifest in manifests
    ]
    if observed_candidate_ids != expected_candidate_ids or len(
        set(observed_candidate_ids)
    ) != len(expected_candidate_ids):
        raise ValueError("completed candidates differ from the exact C1/C2/C3 registry")
    actual_candidate_directories = {
        path.name for path in candidates_root.iterdir() if path.is_dir()
    }
    if actual_candidate_directories != set(expected_candidate_ids):
        raise ValueError("candidate output directories differ from the exact registry")
    candidate_hashes = {
        candidate.candidate_id: _sha256_file(
            candidates_root / candidate.candidate_id / "manifest.json"
        )
        for candidate in CANDIDATE_SPECS
    }
    choice = choose_candidate(manifests)
    selection_payload = _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "regime_tuning_selection",
            "plan_sha256": plan_sha256,
            "candidate_manifest_sha256": candidate_hashes,
            "baseline_reference": baseline.to_dict(),
            "decision": choice,
            "anchor_2024_accessed": False,
            "calibration_or_ensemble_performed": False,
            "test_inference_or_submission_performed": False,
        },
        "selection_sha256",
    )
    selection_path = root / "selection.json"
    _write_or_verify_immutable_json(selection_path, selection_payload)
    selection = _verify_selection_artifact(
        _read_json(selection_path),
        plan_sha256=plan_sha256,
        candidate_manifest_hashes=candidate_hashes,
    )
    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "regime_tuning_complete",
        "plan_sha256": plan_sha256,
        "plan_file_sha256": _sha256_file(root / "plan.json"),
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": _sha256_file(selection_path),
        "candidate_manifest_sha256": candidate_hashes,
        "selected_candidate_id": selection["decision"]["selected_candidate_id"],
        "anchor_2024_accessed": False,
    }
    _write_or_verify_immutable_json(root / "manifest.json", root_manifest)
    return {
        "plan": plan,
        "selection": selection,
        "manifest": root_manifest,
    }


__all__ = [
    "SCHEMA_VERSION",
    "TUNING_SEASONS",
    "DEVELOPMENT_SEASONS",
    "CANDIDATE_SPECS",
    "CandidateSpec",
    "BaselineReference",
    "validate_tuning_seasons",
    "load_tuning_frame",
    "build_tuning_folds",
    "verify_baseline_reference",
    "evaluate_hard_gates",
    "candidate_ranking_key",
    "choose_candidate",
    "run_regime_tuning",
]
