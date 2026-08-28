"""Version-light model artifact persistence for training and submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import AffineLogitCalibrator
from .features import FeaturePreprocessor
from .model import TabularResidualModel


ARTIFACT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_bundle(
    directory: str | Path,
    *,
    preprocessor: FeaturePreprocessor,
    models: list[TabularResidualModel],
    calibrator: AffineLogitCalibrator | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save one preprocessing state and one or more architecture-compatible models."""

    if not models:
        raise ValueError("at least one model is required")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    preprocessor_path = destination / "preprocessor.json"
    model_config_path = destination / "model_config.json"
    preprocessor.save(preprocessor_path)

    config = models[0].get_config()
    for model in models[1:]:
        if model.get_config() != config:
            raise ValueError("all ensemble models must have the same architecture")
    model_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    model_files: list[str] = []
    for index, model in enumerate(models):
        filename = f"model_{index:02d}.npz"
        state_arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in model.cpu().state_dict().items()
        }
        np.savez_compressed(destination / filename, **state_arrays)
        model_files.append(filename)

    calibrator_file: str | None = None
    if calibrator is not None:
        calibrator_file = "calibrator.json"
        calibrator.save(destination / calibrator_file)

    tracked = ["preprocessor.json", "model_config.json", *model_files]
    if calibrator_file:
        tracked.append(calibrator_file)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "model_files": model_files,
        "calibrator_file": calibrator_file,
        "files_sha256": {name: _sha256(destination / name) for name in tracked},
        "metadata": metadata or {},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_model_bundle(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_hashes: bool = True,
) -> tuple[FeaturePreprocessor, list[TabularResidualModel], AffineLogitCalibrator | None, dict[str, Any]]:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version: {manifest.get('artifact_version')}")
    if verify_hashes:
        for name, expected in manifest["files_sha256"].items():
            actual = _sha256(source / name)
            if actual != expected:
                raise ValueError(f"artifact hash mismatch for {name}")

    preprocessor = FeaturePreprocessor.load(source / "preprocessor.json")
    config = json.loads((source / "model_config.json").read_text(encoding="utf-8"))
    target_device = torch.device(device)
    models: list[TabularResidualModel] = []
    for filename in manifest["model_files"]:
        model = TabularResidualModel.from_config(config)
        model_path = source / filename
        if model_path.suffix.lower() == ".npz":
            with np.load(model_path, allow_pickle=False) as archive:
                state = {name: torch.from_numpy(archive[name]) for name in archive.files}
        elif model_path.suffix.lower() == ".pt":
            # Backward-compatible loader for development checkpoints created
            # before the version-independent NPZ deployment format.
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"unsupported model weight format: {filename}")
        model.load_state_dict(state, strict=True)
        model.to(target_device).eval()
        models.append(model)

    calibrator_file = manifest.get("calibrator_file")
    calibrator = (
        AffineLogitCalibrator.load(source / calibrator_file) if calibrator_file else None
    )
    return preprocessor, models, calibrator, dict(manifest.get("metadata", {}))
