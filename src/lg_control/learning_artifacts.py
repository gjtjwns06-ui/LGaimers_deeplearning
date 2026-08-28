"""Strict persistence for the new learning-stage model checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import (
    ContextGatedResidualModel,
    assert_preprocessor_compatible,
)
from .preprocessing import DeepTabularPreprocessor


LEARNING_CHECKPOINT_VERSION = 1
CHECKPOINT_KIND = "development_outer_train_fixed_epoch"
TRACKED_FILES = frozenset(
    {
        "preprocessor.json",
        "model_config.json",
        "model_state.npz",
        "training_record.json",
    }
)


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_training_record_scope(record: Any) -> dict[str, Any]:
    """Require a completed learning-stage record with no evaluation access."""

    if not isinstance(record, dict):
        raise TypeError("training_record must be a dictionary")
    expected_false = (
        "outer_validation_rows_accessed",
        "contains_outer_validation_predictions",
        "contains_outer_validation_metrics",
        "anchor_accessed",
        "diagnostic_accessed",
        "promotion_allowed",
    )
    for name in expected_false:
        if record.get(name) is not False:
            raise ValueError(f"training_record must set {name}=false")
    if record.get("phase") != "development_outer_train_fixed_epoch_refit":
        raise ValueError("training_record has the wrong learning phase")
    if record.get("evaluation_status") != "not_evaluated":
        raise ValueError("training_record must remain not_evaluated")
    access = record.get("outer_validation_access")
    expected_access_keys = {"transform_calls", "predict_calls", "metric_calls"}
    if not isinstance(access, dict) or set(access) != expected_access_keys:
        raise ValueError("training_record outer_validation_access is malformed")
    for name in expected_access_keys:
        if type(access[name]) is not int or access[name] != 0:
            raise ValueError(f"training_record outer validation {name} must be 0")
    return record


@dataclass(frozen=True)
class LoadedLearningCheckpoint:
    preprocessor: DeepTabularPreprocessor
    model: ContextGatedResidualModel
    training_record: dict[str, Any]
    manifest: dict[str, Any]


def save_learning_checkpoint(
    directory: str | Path,
    *,
    preprocessor: DeepTabularPreprocessor,
    model: ContextGatedResidualModel,
    training_record: dict[str, Any],
) -> Path:
    """Atomically save a checkpoint; refuse to overwrite an existing target."""

    destination = Path(directory).resolve()
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    if not isinstance(preprocessor, DeepTabularPreprocessor):
        raise TypeError("preprocessor must be DeepTabularPreprocessor")
    if not isinstance(model, ContextGatedResidualModel):
        raise TypeError("model must be ContextGatedResidualModel")
    _validate_training_record_scope(training_record)
    assert_preprocessor_compatible(model.feature_layout, preprocessor)
    if training_record.get("preprocessor_fingerprint") != (
        model.feature_layout.preprocessor_fingerprint
    ):
        raise ValueError("training_record preprocessor fingerprint differs from model")
    if training_record.get("model_config") != model.get_config():
        raise ValueError("training_record model config differs from model")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    ).resolve()
    if staging.parent != destination.parent:
        raise RuntimeError("checkpoint staging directory escaped destination parent")
    try:
        preprocessor_path = staging / "preprocessor.json"
        model_config_path = staging / "model_config.json"
        model_state_path = staging / "model_state.npz"
        training_record_path = staging / "training_record.json"
        preprocessor.save(preprocessor_path)
        _write_json(model_config_path, model.get_config())
        _write_json(training_record_path, training_record)

        state_arrays = {
            name: tensor.detach().to("cpu").numpy().copy()
            for name, tensor in sorted(model.state_dict().items())
        }
        np.savez_compressed(model_state_path, **state_arrays)
        state_schema = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in state_arrays.items()
        }
        manifest = {
            "checkpoint_version": LEARNING_CHECKPOINT_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND,
            "files_sha256": {
                name: _sha256(staging / name) for name in sorted(TRACKED_FILES)
            },
            "model_state_schema": state_schema,
            "preprocessor_state_sha256": model.feature_layout.preprocessor_fingerprint,
            "outer_validation_rows_accessed": False,
            "contains_outer_validation_predictions": False,
            "contains_outer_validation_metrics": False,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise
    return destination / "manifest.json"


def load_learning_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedLearningCheckpoint:
    """Load and cross-check every checkpoint component and state tensor."""

    source = Path(directory).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {source}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    if manifest.get("checkpoint_version") != LEARNING_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported learning checkpoint version: {manifest.get('checkpoint_version')}"
        )
    if manifest.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise ValueError("checkpoint kind does not match the learning-stage contract")
    file_hashes = manifest.get("files_sha256")
    if not isinstance(file_hashes, dict) or set(file_hashes) != TRACKED_FILES:
        raise ValueError("checkpoint file hash inventory is incomplete or unexpected")
    if not all(_is_sha256(value) for value in file_hashes.values()):
        raise ValueError("checkpoint file hash inventory contains an invalid SHA-256")
    for name in sorted(TRACKED_FILES):
        expected = file_hashes[name]
        actual = _sha256(source / name)
        if actual != expected:
            raise ValueError(f"checkpoint hash mismatch for {name}")

    preprocessor = DeepTabularPreprocessor.load(source / "preprocessor.json")
    model_config = json.loads(
        (source / "model_config.json").read_text(encoding="utf-8")
    )
    model = ContextGatedResidualModel.from_config(model_config)
    assert_preprocessor_compatible(model.feature_layout, preprocessor)
    fingerprint = manifest.get("preprocessor_state_sha256")
    if not _is_sha256(fingerprint):
        raise ValueError("checkpoint preprocessor fingerprint is invalid")
    if fingerprint != model.feature_layout.preprocessor_fingerprint:
        raise ValueError("checkpoint preprocessor fingerprint differs from model")

    expected_schema = manifest.get("model_state_schema")
    if not isinstance(expected_schema, dict):
        raise ValueError("checkpoint model state schema is missing")
    expected_state = model.state_dict()
    if set(expected_schema) != set(expected_state):
        raise ValueError("checkpoint model state schema differs from architecture")
    with np.load(source / "model_state.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected_schema):
            raise ValueError("checkpoint model state keys differ from manifest")
        arrays: dict[str, np.ndarray] = {}
        for name in archive.files:
            array = archive[name]
            schema = expected_schema[name]
            if not isinstance(schema, dict) or set(schema) != {"shape", "dtype"}:
                raise ValueError(f"checkpoint state schema is malformed for {name}")
            if not isinstance(schema["shape"], list) or not isinstance(
                schema["dtype"], str
            ):
                raise ValueError(f"checkpoint state schema is malformed for {name}")
            if list(array.shape) != schema["shape"]:
                raise ValueError(f"checkpoint tensor shape mismatch for {name}")
            if str(array.dtype) != schema["dtype"]:
                raise ValueError(f"checkpoint tensor dtype mismatch for {name}")
            architecture_tensor = expected_state[name].detach().cpu().numpy()
            if array.shape != architecture_tensor.shape:
                raise ValueError(
                    f"checkpoint tensor shape differs from architecture for {name}"
                )
            if array.dtype != architecture_tensor.dtype:
                raise ValueError(
                    f"checkpoint tensor dtype differs from architecture for {name}"
                )
            arrays[name] = array.copy()
    if set(arrays) != set(expected_state):
        raise ValueError("checkpoint model state keys differ from architecture")
    state = {name: torch.from_numpy(array) for name, array in arrays.items()}
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()

    training_record = json.loads(
        (source / "training_record.json").read_text(encoding="utf-8")
    )
    _validate_training_record_scope(training_record)
    for name in (
        "outer_validation_rows_accessed",
        "contains_outer_validation_predictions",
        "contains_outer_validation_metrics",
    ):
        if manifest.get(name) is not False or training_record.get(name) is not False:
            raise ValueError(f"checkpoint scope flag {name} must be false")
    if training_record.get("preprocessor_fingerprint") != fingerprint:
        raise ValueError("training record preprocessor fingerprint differs from manifest")
    if training_record.get("model_config") != model.get_config():
        raise ValueError("training record model config differs from checkpoint model")
    return LoadedLearningCheckpoint(
        preprocessor=preprocessor,
        model=model,
        training_record=training_record,
        manifest=manifest,
    )


__all__ = [
    "LEARNING_CHECKPOINT_VERSION",
    "CHECKPOINT_KIND",
    "LoadedLearningCheckpoint",
    "save_learning_checkpoint",
    "load_learning_checkpoint",
]
