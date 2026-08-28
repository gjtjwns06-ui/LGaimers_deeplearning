"""Fold-train-only reconstruction of privileged auxiliary labels.

This module deliberately exposes no fitted state, serializer, lookup, or
inference transform.  It reconstructs labels only after the caller has made a
fold-train split.  In particular, validation and test rows must never be
concatenated to the frame passed here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FOLD_TRAIN_PARTITION = "fold_train"
ROUNDING_TOLERANCE = 0.25

_ROW_ID = "row_id"
_TARGET = "control_success"
_PITCHER_ID = "pitcher_id"
_BATTER_ID = "batter_id"
_PITCHER_COUNT = "asof_pitcher_n"
_BATTER_COUNT = "asof_batter_n"

_PITCHER_RATE_COLUMNS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
_BATTER_RATE_COLUMNS = {
    "success": "asof_batter_success_rate",
    "middle": "asof_batter_middle_rate",
}
_OUTPUT_COLUMNS = (
    _ROW_ID,
    "aux_reverse",
    "aux_middle",
    "aux_ball",
    "aux_strike",
    "auxiliary_valid_mask",
    "consistency_mask",
)
_REQUIRED_COLUMNS = (
    _ROW_ID,
    _TARGET,
    _PITCHER_ID,
    _BATTER_ID,
    _PITCHER_COUNT,
    *_PITCHER_RATE_COLUMNS.values(),
    _BATTER_COUNT,
    *_BATTER_RATE_COLUMNS.values(),
)


class AuxiliaryTargetError(ValueError):
    """Raised when fold-train rows cannot produce trustworthy labels."""


class AuxiliaryTargetScopeError(AuxiliaryTargetError):
    """Raised before any shift when the caller supplies a non-train scope."""


def _validate_scope(partition: str) -> None:
    if partition != FOLD_TRAIN_PARTITION:
        raise AuxiliaryTargetScopeError(
            "auxiliary labels may be reconstructed only from an already split "
            f"fold-train frame; received partition={partition!r}"
        )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    source_missing = frame[column].isna().to_numpy()
    invalid = ~source_missing & ~np.isfinite(values)
    if invalid.any():
        examples = frame.loc[invalid, _ROW_ID].head(5).tolist()
        raise AuxiliaryTargetError(
            f"{column} contains non-numeric or non-finite values at row_id {examples}"
        )
    return values


def _validate_count(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric(frame, column)
    invalid = ~np.isfinite(values) | (values < 0.0) | (values != np.floor(values))
    if invalid.any():
        examples = frame.loc[invalid, _ROW_ID].head(5).tolist()
        raise AuxiliaryTargetError(
            f"{column} must contain non-negative integers at row_id {examples}"
        )
    return values


def _validate_input(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if len(frame) == 0:
        raise AuxiliaryTargetError("fold-train frame must not be empty")
    missing = sorted(set(_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise AuxiliaryTargetError(
            "fold-train frame is missing required columns: " + ", ".join(missing)
        )
    if frame[_ROW_ID].isna().any() or frame[_ROW_ID].duplicated().any():
        raise AuxiliaryTargetError("row_id must be non-missing and unique")
    for entity_column in (_PITCHER_ID, _BATTER_ID):
        if frame[entity_column].isna().any():
            raise AuxiliaryTargetError(f"{entity_column} must not contain missing values")

    target = _numeric(frame, _TARGET)
    invalid_target = ~np.isfinite(target) | ~np.isin(target, (0.0, 1.0))
    if invalid_target.any():
        examples = frame.loc[invalid_target, _ROW_ID].head(5).tolist()
        raise AuxiliaryTargetError(
            f"control_success must be a complete binary training label at row_id {examples}"
        )
    pitcher_count = _validate_count(frame, _PITCHER_COUNT)
    batter_count = _validate_count(frame, _BATTER_COUNT)
    return target, pitcher_count, batter_count


def _shift_within_entity(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Return the next occurrence inside the supplied frame and nowhere else."""

    return frame.groupby(entity_column, sort=False, dropna=False)[
        list(value_columns)
    ].shift(-1)


def _transition_mask(count: np.ndarray, next_count: np.ndarray) -> np.ndarray:
    return np.isfinite(next_count) & (next_count == count + 1.0)


def _recover_binary_increment(
    frame: pd.DataFrame,
    *,
    label_name: str,
    count: np.ndarray,
    current_rate_column: str,
    next_rate: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    current_rate = _numeric(frame, current_rate_column)
    current_finite = np.isfinite(current_rate)
    next_finite = np.isfinite(next_rate)
    # At count zero the source legitimately stores a missing prior rate.  Its
    # cumulative numerator is nevertheless exactly zero.
    current_available = current_finite | ((count == 0.0) & np.isnan(current_rate))
    invalid_rate = transition & (~current_available | ~next_finite)
    invalid_rate |= transition & current_finite & (
        (current_rate < 0.0) | (current_rate > 1.0)
    )
    invalid_rate |= transition & next_finite & (
        (next_rate < 0.0) | (next_rate > 1.0)
    )
    if invalid_rate.any():
        examples = frame.loc[invalid_rate, _ROW_ID].head(5).tolist()
        raise AuxiliaryTargetError(
            f"{label_name} rates are missing or outside [0, 1] on n+1 transitions "
            f"at row_id {examples}"
        )

    safe_current = np.where(current_finite, current_rate, 0.0)
    raw = (count + 1.0) * next_rate - count * safe_current
    rounded = np.rint(raw)
    invalid_increment = transition & (
        ~np.isfinite(raw)
        | ~np.isin(rounded, (0.0, 1.0))
        | (np.abs(raw - rounded) > ROUNDING_TOLERANCE)
    )
    if invalid_increment.any():
        examples = frame.loc[invalid_increment, _ROW_ID].head(5).tolist()
        raise AuxiliaryTargetError(
            f"{label_name} cumulative-rate increment is not an unambiguous binary "
            f"label at row_id {examples}"
        )

    result = np.full(len(frame), np.nan, dtype=np.float32)
    result[transition] = rounded[transition].astype(np.float32)
    return result


def _assert_equal_on_mask(
    frame: pd.DataFrame,
    *,
    name: str,
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> None:
    mismatch = mask & (first != second)
    if mismatch.any():
        examples = frame.loc[mismatch, _ROW_ID].head(5).tolist()
        raise AuxiliaryTargetError(
            f"{name} cross-check failed at row_id {examples}; all reconstructed "
            "training labels must agree"
        )


def recover_fold_train_auxiliary_targets(
    frame: pd.DataFrame,
    *,
    partition: str,
) -> pd.DataFrame:
    """Recover privileged labels from an already isolated fold-train frame.

    The input order defines each entity's next occurrence.  A label is emitted
    only when that next occurrence exists *inside this frame* and its official
    cumulative count is exactly ``n + 1``.  Terminal and discontinuous rows
    remain missing.

    ``consistency_mask`` identifies valid rows where reconstructed reverse or
    middle is positive.  Those states imply a failed command in the official
    training data and may support a soft main/auxiliary consistency loss.

    The result is deliberately label-only: it contains no cumulative source
    values, next-row values, fitted state, or inference-time transform.
    """

    # Scope and label presence are checked before the first groupby/shift.  A
    # validation/test caller therefore cannot accidentally inspect another row.
    _validate_scope(partition)
    target, pitcher_count, batter_count = _validate_input(frame)

    pitcher_columns = (_PITCHER_COUNT, *_PITCHER_RATE_COLUMNS.values())
    next_pitcher = _shift_within_entity(
        frame,
        entity_column=_PITCHER_ID,
        value_columns=pitcher_columns,
    )
    next_pitcher_count = pd.to_numeric(
        next_pitcher[_PITCHER_COUNT], errors="coerce"
    ).to_numpy(dtype=np.float64)
    pitcher_transition = _transition_mask(pitcher_count, next_pitcher_count)

    pitcher_labels: dict[str, np.ndarray] = {}
    for label_name, rate_column in _PITCHER_RATE_COLUMNS.items():
        next_rate = pd.to_numeric(
            next_pitcher[rate_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        pitcher_labels[label_name] = _recover_binary_increment(
            frame,
            label_name=f"pitcher {label_name}",
            count=pitcher_count,
            current_rate_column=rate_column,
            next_rate=next_rate,
            transition=pitcher_transition,
        )

    next_batter = _shift_within_entity(
        frame,
        entity_column=_BATTER_ID,
        value_columns=(_BATTER_COUNT, *_BATTER_RATE_COLUMNS.values()),
    )
    next_batter_count = pd.to_numeric(
        next_batter[_BATTER_COUNT], errors="coerce"
    ).to_numpy(dtype=np.float64)
    batter_transition = _transition_mask(batter_count, next_batter_count)
    batter_labels: dict[str, np.ndarray] = {}
    for label_name, rate_column in _BATTER_RATE_COLUMNS.items():
        next_rate = pd.to_numeric(
            next_batter[rate_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        batter_labels[label_name] = _recover_binary_increment(
            frame,
            label_name=f"batter {label_name}",
            count=batter_count,
            current_rate_column=rate_column,
            next_rate=next_rate,
            transition=batter_transition,
        )

    _assert_equal_on_mask(
        frame,
        name="pitcher success versus control_success",
        first=pitcher_labels["success"],
        second=target,
        mask=pitcher_transition,
    )
    _assert_equal_on_mask(
        frame,
        name="batter success versus control_success",
        first=batter_labels["success"],
        second=target,
        mask=batter_transition,
    )
    shared_transition = pitcher_transition & batter_transition
    _assert_equal_on_mask(
        frame,
        name="pitcher middle versus batter middle",
        first=pitcher_labels["middle"],
        second=batter_labels["middle"],
        mask=shared_transition,
    )

    consistency_mask = pitcher_transition & (
        (pitcher_labels["reverse"] == 1.0)
        | (pitcher_labels["middle"] == 1.0)
    )
    _assert_equal_on_mask(
        frame,
        name="reverse/middle consistency versus failed control",
        first=target,
        second=np.zeros(len(frame), dtype=np.float64),
        mask=consistency_mask,
    )

    result = pd.DataFrame(index=frame.index.copy())
    result[_ROW_ID] = frame[_ROW_ID].copy()
    for label_name in ("reverse", "middle", "ball", "strike"):
        result[f"aux_{label_name}"] = pitcher_labels[label_name]
    result["auxiliary_valid_mask"] = pitcher_transition.astype(bool)
    result["consistency_mask"] = consistency_mask.astype(bool)
    return result.loc[:, _OUTPUT_COLUMNS]


__all__ = (
    "AuxiliaryTargetError",
    "AuxiliaryTargetScopeError",
    "recover_fold_train_auxiliary_targets",
)
