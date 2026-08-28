"""Batch inference helpers whose transforms are independent across rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from .calibration import AffineLogitCalibrator
from .features import FeaturePreprocessor
from .schema import validate_predictions


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_probabilities(
    model: torch.nn.Module,
    preprocessor: FeaturePreprocessor,
    frame: pd.DataFrame,
    *,
    batch_size: int = 16_384,
    device: str | torch.device | None = None,
    calibrator: AffineLogitCalibrator | None = None,
) -> np.ndarray:
    """Predict one probability per row without cross-row feature operations."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    arrays = preprocessor.transform(frame)
    categorical = arrays["categorical"]
    numerical = arrays["numerical"]
    target_device = resolve_device(device)
    model = model.to(target_device)
    was_training = model.training
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            cat_batch = torch.from_numpy(categorical[start:stop]).to(
                target_device, dtype=torch.long, non_blocking=target_device.type == "cuda"
            )
            num_batch = torch.from_numpy(numerical[start:stop]).to(
                target_device, dtype=torch.float32, non_blocking=target_device.type == "cuda"
            )
            logits = model(cat_batch, num_batch)
            pieces.append(torch.sigmoid(logits).float().cpu().numpy())
    if was_training:
        model.train()
    probabilities = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)
    if calibrator is not None:
        probabilities = calibrator.transform(probabilities)
    return validate_predictions(frame["row_id"], probabilities)


def predict_ensemble(
    models: Sequence[torch.nn.Module],
    preprocessor: FeaturePreprocessor,
    frame: pd.DataFrame,
    *,
    batch_size: int = 16_384,
    device: str | torch.device | None = None,
    calibrator: AffineLogitCalibrator | None = None,
) -> np.ndarray:
    if not models:
        raise ValueError("at least one model is required")
    raw = [
        predict_probabilities(
            model,
            preprocessor,
            frame,
            batch_size=batch_size,
            device=device,
            calibrator=None,
        )
        for model in models
    ]
    probabilities = np.mean(np.stack(raw, axis=0), axis=0)
    if calibrator is not None:
        probabilities = calibrator.transform(probabilities)
    return validate_predictions(frame["row_id"], probabilities)

