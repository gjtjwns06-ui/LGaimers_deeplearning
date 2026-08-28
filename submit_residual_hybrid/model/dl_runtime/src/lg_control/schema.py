"""Strict input/output validation used by training and submission code."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import ID_COL, RAW_FEATURE_COLUMNS, TARGET_COL


class SchemaError(ValueError):
    """Raised when an input or prediction violates the competition contract."""


def validate_input_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = False,
    allow_target: bool = True,
    allowed_extra_columns: Sequence[str] = (),
) -> None:
    """Validate official columns, identifiers, target, and numeric finiteness."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    required = {ID_COL, *RAW_FEATURE_COLUMNS}
    if require_target:
        required.add(TARGET_COL)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    allowed = set(required)
    if allow_target:
        allowed.add(TARGET_COL)
    allowed.update(allowed_extra_columns)
    extras = sorted(set(frame.columns).difference(allowed))
    if extras:
        raise SchemaError(f"unexpected columns: {extras}")

    if frame[ID_COL].isna().any():
        raise SchemaError(f"{ID_COL} contains missing values")
    if frame[ID_COL].duplicated().any():
        examples = frame.loc[frame[ID_COL].duplicated(keep=False), ID_COL].head(5).tolist()
        raise SchemaError(f"{ID_COL} must be unique; duplicates include {examples}")

    if TARGET_COL in frame.columns:
        target = frame[TARGET_COL]
        if target.isna().any() or not target.isin([0, 1]).all():
            raise SchemaError(f"{TARGET_COL} must contain only binary 0/1 values")

    numeric = frame[RAW_FEATURE_COLUMNS].select_dtypes(include=[np.number])
    if numeric.shape[1] and np.isinf(numeric.to_numpy(dtype=np.float64, copy=False)).any():
        raise SchemaError("numeric input contains +inf or -inf")


def validate_predictions(ids: Sequence[object], probabilities: np.ndarray) -> np.ndarray:
    """Return validated float64 probabilities with a strict one-to-one ID map."""

    id_series = pd.Series(ids, copy=False)
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(id_series) != len(values):
        raise SchemaError(
            f"prediction length mismatch: ids={len(id_series)}, probabilities={len(values)}"
        )
    if id_series.isna().any() or id_series.duplicated().any():
        raise SchemaError("prediction IDs must be non-missing and unique")
    if not np.isfinite(values).all():
        raise SchemaError("predictions contain NaN or infinite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise SchemaError("predictions must be inside [0, 1]")
    return values
