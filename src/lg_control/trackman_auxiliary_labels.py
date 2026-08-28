"""Leakage-safe current-pitch TrackMan labels for fold-train auxiliary losses.

These labels are *privileged training information*: they may supervise a
representation while fitting a fold, but they are never features and there is
intentionally no evaluation/test ``transform`` method.  Exact pitch identity is
inherited from :mod:`lg_control.strict_trackman`; ambiguous, reused, calendar-
mismatched, or context-mismatched games therefore receive missing-label masks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .strict_trackman import (
    MAIN_ALIGNMENT_REQUIRED_COLUMNS,
    StrictTrackmanError,
    _build_alignment_from_prepared,
    _prepare_trackman_history,
)
from .trackman import METRIC_COLUMNS, PITCH_TYPE_GROUPS


TRACKMAN_AUXILIARY_STATE_VERSION = 1
PITCH_TYPE_TO_INDEX = {name: index for index, name in enumerate(PITCH_TYPE_GROUPS)}


class TrackmanAuxiliaryBoundaryError(StrictTrackmanError):
    """Raised when evaluation-season main rows enter fold-train supervision."""


@dataclass(frozen=True)
class TrackmanAuxiliaryLabels:
    """Auxiliary arrays in exactly the caller's main-row order.

    ``continuous_targets`` are z-scored with statistics computed solely from
    exactly aligned, pre-validation-season pitches. Missing targets are zero;
    ``continuous_mask`` is the authoritative indicator for the loss.
    """

    row_ids: tuple[str, ...]
    pitch_type_targets: np.ndarray
    pitch_type_mask: np.ndarray
    continuous_targets: np.ndarray
    continuous_mask: np.ndarray
    continuous_location: np.ndarray
    continuous_scale: np.ndarray
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        row_count = len(self.row_ids)
        metric_count = len(METRIC_COLUMNS)
        expected_vectors = (
            (self.pitch_type_targets, (row_count,)),
            (self.pitch_type_mask, (row_count,)),
            (self.continuous_targets, (row_count, metric_count)),
            (self.continuous_mask, (row_count, metric_count)),
            (self.continuous_location, (metric_count,)),
            (self.continuous_scale, (metric_count,)),
        )
        for array, expected_shape in expected_vectors:
            if np.asarray(array).shape != expected_shape:
                raise ValueError(
                    f"TrackMan auxiliary array shape {np.asarray(array).shape} "
                    f"does not match {expected_shape}"
                )
        if len(set(self.row_ids)) != row_count:
            raise ValueError("TrackMan auxiliary row_ids must be unique")
        if not np.isfinite(self.continuous_targets).all():
            raise ValueError("continuous_targets must be finite")
        if not np.isfinite(self.continuous_location).all() or not np.isfinite(
            self.continuous_scale
        ).all():
            raise ValueError("continuous normalization statistics must be finite")
        if (self.continuous_scale <= 0).any():
            raise ValueError("continuous scales must be positive")


def _validated_seasons(values: pd.Series, *, name: str) -> np.ndarray:
    seasons = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(seasons) & (seasons == np.floor(seasons)) & (seasons >= 1900)
    if not valid.all():
        positions = np.flatnonzero(~valid)[:5].tolist()
        raise StrictTrackmanError(f"{name} has invalid seasons at positions {positions}")
    return seasons.astype(np.int64)


def _label_fingerprint(
    row_ids: tuple[str, ...],
    pitch_targets: np.ndarray,
    pitch_mask: np.ndarray,
    physical: np.ndarray,
    physical_mask: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for row_id in row_ids:
        encoded = row_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    digest.update(np.asarray(pitch_targets, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(pitch_mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(np.asarray(physical, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(physical_mask, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


class FoldTrainTrackmanAuxiliaryBuilder:
    """Build current-pitch labels only for rows strictly before a fold boundary."""

    def __init__(self, *, validation_year: int) -> None:
        if isinstance(validation_year, bool) or int(validation_year) != validation_year:
            raise TrackmanAuxiliaryBoundaryError("validation_year must be an integer")
        self.validation_year = int(validation_year)
        if self.validation_year < 1901:
            raise TrackmanAuxiliaryBoundaryError("validation_year must be >= 1901")
        self.continuous_location_: np.ndarray | None = None
        self.continuous_scale_: np.ndarray | None = None
        self.audit_: dict[str, Any] = {}
        self.is_fitted_ = False

    def fit_transform(
        self, fold_train: pd.DataFrame, trackman_history: pd.DataFrame
    ) -> TrackmanAuxiliaryLabels:
        missing_main = sorted(set(MAIN_ALIGNMENT_REQUIRED_COLUMNS) - set(fold_train))
        if missing_main:
            raise StrictTrackmanError(
                f"fold_train is missing required columns: {missing_main}"
            )
        if "season" not in trackman_history:
            raise StrictTrackmanError("TrackMan history is missing required column: season")
        main_seasons = _validated_seasons(fold_train["season"], name="fold_train")
        if (main_seasons >= self.validation_year).any():
            offending = sorted(set(main_seasons[main_seasons >= self.validation_year]))
            raise TrackmanAuxiliaryBoundaryError(
                "fold_train contains validation/future seasons "
                f"{offending}; all rows must be before {self.validation_year}"
            )
        history_seasons = _validated_seasons(
            trackman_history["season"], name="TrackMan history"
        )
        eligible_mask = history_seasons < self.validation_year
        eligible_history = trackman_history.loc[eligible_mask].copy()
        if len(eligible_history) == 0:
            raise TrackmanAuxiliaryBoundaryError(
                "no TrackMan rows exist before validation_year"
            )

        # Crucially, strict cleaning/alignment sees only the fold-train side of
        # the boundary. Validation/future current-pitch values never enter it.
        prepared, trackman_audit = _prepare_trackman_history(eligible_history)
        if (prepared["_source_season"].to_numpy(dtype=np.int64) >= self.validation_year).any():
            raise RuntimeError("TrackMan boundary filter failed")
        alignment = _build_alignment_from_prepared(
            fold_train, prepared, trackman_audit
        )

        label_pieces: list[pd.DataFrame] = []
        for match in alignment.matches.itertuples(index=False):
            aligned = alignment.aligned_rows.loc[
                alignment.aligned_rows["trackman_game_id"].eq(
                    str(match.trackman_game_id)
                )
            ].sort_values("main_position", kind="mergesort")
            physical = prepared.loc[
                prepared["_game_id"].eq(str(match.trackman_game_id)),
                ["pitch_no", "_trackman_id_int", "pitch_type_group", *METRIC_COLUMNS],
            ].sort_values(["pitch_no", "_trackman_id_int"], kind="mergesort")
            if len(aligned) != len(physical):
                raise RuntimeError("strict alignment label row count changed unexpectedly")
            piece = physical.loc[:, ["pitch_type_group", *METRIC_COLUMNS]].reset_index(
                drop=True
            )
            piece.insert(
                0, "main_row_id", aligned["main_row_id"].astype(str).to_numpy()
            )
            label_pieces.append(piece)

        columns = ["main_row_id", "pitch_type_group", *METRIC_COLUMNS]
        label_table = (
            pd.concat(label_pieces, ignore_index=True)
            if label_pieces
            else pd.DataFrame(columns=columns)
        )
        if label_table["main_row_id"].duplicated().any():
            raise RuntimeError("a main row received more than one TrackMan label")
        row_ids = tuple(fold_train["row_id"].astype(str).tolist())
        ordered = pd.DataFrame({"main_row_id": row_ids}).merge(
            label_table,
            on="main_row_id",
            how="left",
            sort=False,
            validate="one_to_one",
        )

        pitch_series = ordered["pitch_type_group"].map(PITCH_TYPE_TO_INDEX)
        pitch_mask = pitch_series.notna().to_numpy(dtype=bool)
        pitch_targets = pitch_series.fillna(-1).to_numpy(dtype=np.int64)

        raw_physical = ordered.loc[:, METRIC_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        physical_mask = np.isfinite(raw_physical)
        location = np.zeros(len(METRIC_COLUMNS), dtype=np.float64)
        scale = np.ones(len(METRIC_COLUMNS), dtype=np.float64)
        for index in range(len(METRIC_COLUMNS)):
            observed = raw_physical[physical_mask[:, index], index]
            if observed.size:
                location[index] = float(observed.mean())
                standard_deviation = float(observed.std(ddof=0))
                if np.isfinite(standard_deviation) and standard_deviation > 1e-8:
                    scale[index] = standard_deviation
        continuous = np.zeros_like(raw_physical, dtype=np.float64)
        for index in range(len(METRIC_COLUMNS)):
            observed_mask = physical_mask[:, index]
            continuous[observed_mask, index] = (
                raw_physical[observed_mask, index] - location[index]
            ) / scale[index]

        fingerprint = _label_fingerprint(
            row_ids, pitch_targets, pitch_mask, continuous, physical_mask
        )
        audit: dict[str, Any] = {
            "validation_year": self.validation_year,
            "fold_train_rows": int(len(fold_train)),
            "trackman_input_rows": int(len(trackman_history)),
            "trackman_boundary_excluded_rows": int((~eligible_mask).sum()),
            "trackman_pre_boundary_rows": int(eligible_mask.sum()),
            "pitch_type_labeled_rows": int(pitch_mask.sum()),
            "pitch_type_missing_rows": int((~pitch_mask).sum()),
            "pitch_type_counts": {
                group: int((pitch_targets[pitch_mask] == index).sum())
                for group, index in PITCH_TYPE_TO_INDEX.items()
            },
            "continuous_observed_counts": {
                metric: int(physical_mask[:, index].sum())
                for index, metric in enumerate(METRIC_COLUMNS)
            },
            "label_fingerprint_sha256": fingerprint,
            **{f"alignment_{key}": int(value) for key, value in alignment.audit.items()},
        }
        self.continuous_location_ = location.copy()
        self.continuous_scale_ = scale.copy()
        self.audit_ = audit.copy()
        self.is_fitted_ = True
        return TrackmanAuxiliaryLabels(
            row_ids=row_ids,
            pitch_type_targets=pitch_targets,
            pitch_type_mask=pitch_mask,
            continuous_targets=continuous.astype(np.float32),
            continuous_mask=physical_mask,
            continuous_location=location,
            continuous_scale=scale,
            audit=audit,
        )

    def to_state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("builder must be fitted before serialization")
        assert self.continuous_location_ is not None
        assert self.continuous_scale_ is not None
        return {
            "state_version": TRACKMAN_AUXILIARY_STATE_VERSION,
            "builder_class": type(self).__name__,
            "validation_year": self.validation_year,
            "pitch_type_groups": list(PITCH_TYPE_GROUPS),
            "metric_columns": list(METRIC_COLUMNS),
            "continuous_location": self.continuous_location_.tolist(),
            "continuous_scale": self.continuous_scale_.tolist(),
            "audit": self.audit_,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_state_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


__all__ = [
    "TRACKMAN_AUXILIARY_STATE_VERSION",
    "PITCH_TYPE_TO_INDEX",
    "TrackmanAuxiliaryBoundaryError",
    "TrackmanAuxiliaryLabels",
    "FoldTrainTrackmanAuxiliaryBuilder",
]
