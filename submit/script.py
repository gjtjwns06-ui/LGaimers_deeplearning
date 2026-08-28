#!/usr/bin/env python3
"""Run leakage-safe ensemble inference for the LG Aimers submission server."""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_ROOT = PACKAGE_ROOT / "model"
if str(LOCAL_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_MODEL_ROOT))

import numpy as np
import pandas as pd
import torch

from lg_control.artifacts import load_model_bundle
from lg_control.schema import validate_input_frame, validate_predictions


ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"
BATCH_SIZE = 16_384
BUNDLE_DIRECTORY = LOCAL_MODEL_ROOT / "bundle"
OUTPUT_PATH = Path("output") / "submission.csv"


def resolve_test_path() -> Path:
    """Use the official path first and the alternate open/ layout second."""

    preferred = Path("data") / "test.csv"
    fallback = Path("open") / "test.csv"
    if preferred.exists():
        if not preferred.is_file():
            raise FileNotFoundError(f"test path is not a file: {preferred}")
        return preferred
    if fallback.exists():
        if not fallback.is_file():
            raise FileNotFoundError(f"test path is not a file: {fallback}")
        return fallback
    raise FileNotFoundError(
        "test.csv was not found; checked ./data/test.csv and ./open/test.csv"
    )


def load_test_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    validate_input_frame(frame, require_target=False, allow_target=False)
    if frame.empty:
        raise ValueError("test.csv must contain at least one row")
    if not frame[ID_COLUMN].is_unique:
        # Kept here as an explicit submission-level contract even though the
        # shared schema validator already rejects duplicates.
        raise ValueError(f"{ID_COLUMN} must be unique")
    return frame


def transform_frame(preprocessor: object, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Apply the saved row-local transform with writable NumPy outputs.

    pandas 2.3 can expose a read-only view for a nullable Series even when
    ``to_numpy(dtype=...)`` is used.  The training preprocessor assigns its
    missing-category mask into that array, so inference makes the copy
    explicit here.  Feature construction and every fitted parameter still
    come from the packaged preprocessor.
    """

    preprocessor._require_fitted()
    categorical_frame = preprocessor._categorical_frame(frame)
    categorical_columns = list(preprocessor.categorical_columns)
    categorical = np.empty((len(frame), len(categorical_columns)), dtype=np.int64)
    for index, column in enumerate(categorical_columns):
        token_series = categorical_frame[column].astype("string")
        mapped = token_series.map(preprocessor.category_maps[column])
        values = np.array(mapped.fillna(1), dtype=np.int64, copy=True)
        values[token_series.isna().to_numpy(copy=True)] = 0
        categorical[:, index] = values

    numerical_frame = preprocessor._numerical_frame(frame)
    numerical_columns = list(preprocessor.numerical_columns)
    if list(numerical_frame.columns) != numerical_columns:
        raise RuntimeError("numeric feature order changed after fitting")
    numerical = np.empty((len(frame), len(numerical_columns)), dtype=np.float32)
    for index, column in enumerate(numerical_columns):
        values = pd.to_numeric(numerical_frame[column], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=True,
        )
        mean = preprocessor.numeric_means[column]
        std = preprocessor.numeric_stds[column]
        values = np.nan_to_num(values, nan=mean, posinf=mean, neginf=mean)
        numerical[:, index] = np.clip((values - mean) / std, -10.0, 10.0).astype(
            np.float32
        )
    return {"categorical": categorical, "numerical": numerical}


def predict_on_device(frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Transform once, average every bundled model, then calibrate once."""

    preprocessor, models, calibrator, _ = load_model_bundle(
        BUNDLE_DIRECTORY,
        device=device,
        verify_hashes=True,
    )
    if not models:
        raise RuntimeError("bundle must contain at least one model")
    if calibrator is None:
        raise RuntimeError("bundle must contain the fitted calibrator")

    arrays = transform_frame(preprocessor, frame)
    categorical = arrays["categorical"]
    numerical = arrays["numerical"]
    if categorical.shape[0] != len(frame) or numerical.shape[0] != len(frame):
        raise RuntimeError("preprocessor changed the number of test rows")

    probability_sum = np.zeros(len(frame), dtype=np.float64)
    with torch.inference_mode():
        for model in models:
            model.to(device).eval()
            for start in range(0, len(frame), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(frame))
                cat_batch = torch.from_numpy(categorical[start:stop]).to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=device.type == "cuda",
                )
                num_batch = torch.from_numpy(numerical[start:stop]).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                logits = model(cat_batch, num_batch)
                expected_shape = (stop - start,)
                if tuple(logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"model returned shape {tuple(logits.shape)}; "
                        f"expected {expected_shape}"
                    )
                probability_sum[start:stop] += (
                    torch.sigmoid(logits).float().cpu().numpy().astype(np.float64)
                )

    raw_probabilities = probability_sum / float(len(models))
    calibrated = calibrator.transform(raw_probabilities)
    return validate_predictions(frame[ID_COLUMN], calibrated)


def predict_with_fallback(frame: pd.DataFrame) -> tuple[np.ndarray, torch.device]:
    """Prefer CUDA when available and retry cleanly on CPU on CUDA failure."""

    preferred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred.type == "cpu":
        return predict_on_device(frame, preferred), preferred

    try:
        return predict_on_device(frame, preferred), preferred
    except RuntimeError as cuda_error:
        print(
            f"CUDA inference failed ({cuda_error}); retrying on CPU.",
            file=sys.stderr,
            flush=True,
        )
        with suppress(Exception):
            torch.cuda.empty_cache()
        gc.collect()
        cpu = torch.device("cpu")
        return predict_on_device(frame, cpu), cpu


def validate_submission(frame: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(
            f"submission columns must be {[ID_COLUMN, TARGET_COLUMN]}, "
            f"got {list(submission.columns)}"
        )
    if len(submission) != len(frame):
        raise ValueError(
            f"submission row count mismatch: test={len(frame)}, "
            f"submission={len(submission)}"
        )
    if not submission[ID_COLUMN].is_unique:
        raise ValueError(f"submission {ID_COLUMN} values must be unique")
    if not submission[ID_COLUMN].equals(frame[ID_COLUMN]):
        raise ValueError("submission row_id values or order differ from test.csv")
    validate_predictions(submission[ID_COLUMN], submission[TARGET_COLUMN].to_numpy())


def write_submission(frame: pd.DataFrame, probabilities: np.ndarray) -> None:
    submission = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].copy(),
            TARGET_COLUMN: probabilities,
        }
    )
    validate_submission(frame, submission)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix="submission_",
            suffix=".csv.tmp",
            dir=OUTPUT_PATH.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            submission.to_csv(temporary, index=False)
        os.replace(temporary_path, OUTPUT_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    test_path = resolve_test_path()
    print(f"Loading test data: {test_path}", flush=True)
    frame = load_test_frame(test_path)
    print(f"Running inference for {len(frame):,} rows...", flush=True)
    probabilities, device = predict_with_fallback(frame)
    write_submission(frame, probabilities)
    print(
        f"Saved {OUTPUT_PATH} ({len(frame):,} rows, device={device.type})",
        flush=True,
    )


if __name__ == "__main__":
    main()
