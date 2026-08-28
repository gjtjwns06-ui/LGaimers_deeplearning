"""Runtime for the row-local combined-signal R complement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd
import torch

from .combined_signal_features import CombinedSignalAugmentedPreprocessor
from .regime_model import (
    GatedCrossModelConfig,
    MultiTaskGatedCrossRegimeModel,
    MultiTaskHeadConfig,
)
from .training import predict_arrays


@dataclass
class CombinedRComplementArtifact:
    preprocessor: CombinedSignalAugmentedPreprocessor
    model: MultiTaskGatedCrossRegimeModel
    gate: dict[str, Any]


def _load_state(model: torch.nn.Module, path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    model.load_state_dict(state, strict=True)


def load_combined_r_complement(
    model_directory: Path,
    gate_path: Path,
    *,
    device: str = "cpu",
) -> CombinedRComplementArtifact:
    preprocessor = CombinedSignalAugmentedPreprocessor.from_state_dict(
        json.loads((model_directory / "preprocessor.json").read_text(encoding="utf-8"))
    )
    config = json.loads((model_directory / "model_config.json").read_text(encoding="utf-8"))
    if config.get("architecture") != "dcnv2_gated_experts_multitask_v1":
        raise ValueError("unexpected combined complement architecture")
    multitask = dict(config["multitask"])
    if tuple(multitask.pop("auxiliary_targets")) != ("reverse", "middle", "ball", "strike"):
        raise ValueError("combined complement auxiliary targets differ")
    model = MultiTaskGatedCrossRegimeModel(
        preprocessor.feature_layout(),
        GatedCrossModelConfig(**config["model"]),
        MultiTaskHeadConfig(**multitask),
    )
    _load_state(model, model_directory / "model_state.npz")
    model.to(torch.device(device)).eval()
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        payload.get("kind") != "combined_signal_r_complement_gate_v1"
        or payload.get("decision") != "PASS"
        or payload.get("scope") != "R"
    ):
        raise ValueError("combined R complement gate is invalid")
    gate = payload.get("gate")
    if not isinstance(gate, dict) or gate.get("columns") != ["uncertainty", "direction"]:
        raise ValueError("combined R complement gate columns differ")
    return CombinedRComplementArtifact(preprocessor=preprocessor, model=model, gate=gate)


def predict_combined_r_complement(
    artifact: CombinedRComplementArtifact,
    frame: pd.DataFrame,
    base_probability: np.ndarray,
    *,
    batch_size: int = 32768,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_probability, dtype=np.float64)
    if base.shape != (len(frame),) or not np.isfinite(base).all():
        raise ValueError("combined complement base probability is invalid")
    arrays = artifact.preprocessor.transform(frame)
    specialist = predict_arrays(
        artifact.model,
        arrays["categorical"],
        arrays["numerical"],
        batch_size=batch_size,
        device=torch.device(device),
    ).astype(np.float64)
    distance = np.abs(base - 0.5)
    uncertainty = np.where(distance <= 0.025, "u025", np.where(distance <= 0.050, "u050", "far"))
    direction = np.where(specialist >= base, "up", "down")
    keys = np.char.add(np.char.add(uncertainty.astype(str), "|"), direction.astype(str))
    groups = artifact.gate["groups"]
    fallback = float(artifact.gate["global_weight"])
    cap = float(artifact.gate["cap"])
    weights = np.fromiter(
        (float(groups.get(str(key), fallback)) for key in keys),
        dtype=np.float64,
        count=len(frame),
    )
    weights = np.clip(weights, 0.0, cap)
    r_mask = frame.game_type.astype(str).str.upper().eq("R").to_numpy()
    weights[~r_mask] = 0.0
    probability = np.clip(base + weights * (specialist - base), 1e-6, 1.0 - 1e-6)
    if not np.isfinite(probability).all():
        raise RuntimeError("combined R complement produced non-finite probabilities")
    return probability, specialist, weights


__all__ = [
    "CombinedRComplementArtifact",
    "load_combined_r_complement",
    "predict_combined_r_complement",
]
