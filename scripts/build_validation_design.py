"""Build and audit the fixed temporal validation design.

This script stops at validation partition design.  It creates no model,
prediction, metric-on-predictions, checkpoint, calibration, or submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import ID_COL, RECENT_RATE_COLUMNS, TARGET_COL, TRAIN_COLUMNS
from lg_control.validation import (
    EXPECTED_SEASONS,
    SeasonFold,
    ValidationPlan,
    build_validation_plan,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "validation_design"


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_identifiers(values: Iterable[object]) -> str:
    """Hash an ordered ID sequence with unambiguous length-prefix encoding."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def _rate(frame: pd.DataFrame) -> float:
    return float(frame[TARGET_COL].mean())


def _conditional_rate(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> float:
    selected = frame.loc[np.asarray(mask, dtype=bool), TARGET_COL]
    return float(selected.mean()) if len(selected) else float("nan")


def _entity_masks(
    train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, np.ndarray]:
    train_pitchers = set(train["pitcher_id"].dropna().tolist())
    train_batters = set(train["batter_id"].dropna().tolist())
    pitcher_seen = validation["pitcher_id"].isin(train_pitchers).to_numpy()
    batter_seen = validation["batter_id"].isin(train_batters).to_numpy()
    return {
        "both_seen": pitcher_seen & batter_seen,
        "pitcher_only": pitcher_seen & ~batter_seen,
        "batter_only": ~pitcher_seen & batter_seen,
        "neither_seen": ~pitcher_seen & ~batter_seen,
    }


def _recent_missing_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame[RECENT_RATE_COLUMNS].isna().any(axis=1).to_numpy()


def _fold_frames(frame: pd.DataFrame, fold: SeasonFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_columns = [
        ID_COL,
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_batter_n",
        *RECENT_RATE_COLUMNS,
        TARGET_COL,
    ]
    audit_frame = frame.loc[:, audit_columns]
    train = audit_frame.iloc[fold.train_positions]
    validation = audit_frame.iloc[fold.validation_positions]
    return train, validation


def _season_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in EXPECTED_SEASONS:
        part = frame.loc[frame["season"].eq(season)]
        recent_missing = _recent_missing_mask(part)
        game_f = part["game_type"].eq("F")
        game_r = part["game_type"].eq("R")
        rows.append(
            {
                "season": season,
                "rows": int(len(part)),
                "target_rate": _rate(part),
                "game_type_f_share": float(game_f.mean()),
                "game_type_f_target_rate": _conditional_rate(part, game_f),
                "game_type_r_target_rate": _conditional_rate(part, game_r),
                "unique_pitchers": int(part["pitcher_id"].nunique(dropna=True)),
                "unique_batters": int(part["batter_id"].nunique(dropna=True)),
                "recent_missing_rows": int(recent_missing.sum()),
                "recent_missing_rate": float(recent_missing.mean()),
            }
        )
    return pd.DataFrame(rows)


def _fold_summary(frame: pd.DataFrame, plan: ValidationPlan) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in plan.execution_folds():
        train, validation = _fold_frames(frame, fold)
        masks = _entity_masks(train, validation)
        train_f = train["game_type"].eq("F")
        validation_f = validation["game_type"].eq("F")
        train_r = train["game_type"].eq("R")
        validation_r = validation["game_type"].eq("R")
        train_target_rate = _rate(train)
        validation_target_rate = _rate(validation)
        train_f_rate = _conditional_rate(train, train_f)
        validation_f_rate = _conditional_rate(validation, validation_f)
        train_r_rate = _conditional_rate(train, train_r)
        validation_r_rate = _conditional_rate(validation, validation_r)
        inner = fold.inner_stop_split
        recent_missing = _recent_missing_mask(validation)
        rows.append(
            {
                "fold": fold.name,
                "role": fold.role,
                "family": "diagnostic" if fold.role == "diagnostic" else "canonical",
                "train_seasons": ",".join(map(str, fold.train_seasons)),
                "validation_season": fold.validation_season,
                "inner_train_seasons": "" if inner is None else ",".join(map(str, inner.train_seasons)),
                "inner_stop_season": np.nan if inner is None else inner.validation_season,
                "inner_train_rows": np.nan if inner is None else inner.train_row_count,
                "inner_stop_rows": np.nan if inner is None else inner.validation_row_count,
                "train_rows": fold.train_row_count,
                "validation_rows": fold.validation_row_count,
                "train_target_rate": train_target_rate,
                "validation_target_rate": validation_target_rate,
                "target_shift_pp": 100.0 * (validation_target_rate - train_target_rate),
                "train_game_type_f_share": float(train_f.mean()),
                "validation_game_type_f_share": float(validation_f.mean()),
                "train_game_type_f_target_rate": train_f_rate,
                "validation_game_type_f_target_rate": validation_f_rate,
                "game_type_f_target_shift_pp": 100.0 * (validation_f_rate - train_f_rate),
                "train_game_type_r_target_rate": train_r_rate,
                "validation_game_type_r_target_rate": validation_r_rate,
                "game_type_r_target_shift_pp": 100.0 * (validation_r_rate - train_r_rate),
                "pitcher_seen_rate": float((masks["both_seen"] | masks["pitcher_only"]).mean()),
                "batter_seen_rate": float((masks["both_seen"] | masks["batter_only"]).mean()),
                "both_seen_rate": float(masks["both_seen"].mean()),
                "neither_seen_rate": float(masks["neither_seen"].mean()),
                "recent_missing_rate": float(recent_missing.mean()),
            }
        )
    return pd.DataFrame(rows)


def _append_slice(
    rows: list[dict[str, object]],
    *,
    fold: SeasonFold,
    validation: pd.DataFrame,
    family: str,
    slice_name: str,
    value: str,
    mask: pd.Series | np.ndarray,
) -> None:
    selected = np.asarray(mask, dtype=bool)
    count = int(selected.sum())
    rows.append(
        {
            "fold": fold.name,
            "family": family,
            "slice": slice_name,
            "value": value,
            "rows": count,
            "share": count / len(validation),
            "target_rate": _conditional_rate(validation, selected),
        }
    )


def _subgroup_summary(frame: pd.DataFrame, plan: ValidationPlan) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in plan.execution_folds():
        train, validation = _fold_frames(frame, fold)
        family = "diagnostic" if fold.role == "diagnostic" else "canonical"
        _append_slice(
            rows,
            fold=fold,
            validation=validation,
            family=family,
            slice_name="overall",
            value="all",
            mask=np.ones(len(validation), dtype=bool),
        )
        for game_type in ("F", "R"):
            _append_slice(
                rows,
                fold=fold,
                validation=validation,
                family=family,
                slice_name="game_type",
                value=game_type,
                mask=validation["game_type"].eq(game_type),
            )
        for value, mask in _entity_masks(train, validation).items():
            _append_slice(
                rows,
                fold=fold,
                validation=validation,
                family=family,
                slice_name="entity_status",
                value=value,
                mask=mask,
            )
        recent_missing = _recent_missing_mask(validation)
        for value, mask in (
            ("recent_missing", recent_missing),
            ("recent_observed", ~recent_missing),
        ):
            _append_slice(
                rows,
                fold=fold,
                validation=validation,
                family=family,
                slice_name="recent_status",
                value=value,
                mask=mask,
            )
        for entity in ("pitcher", "batter"):
            counts = pd.to_numeric(validation[f"asof_{entity}_n"], errors="coerce")
            masks = {
                "cold_start_0": counts.eq(0),
                "low_history_1_49": counts.gt(0) & counts.lt(50),
                "history_50_plus": counts.ge(50),
                "history_missing": counts.isna(),
            }
            for value, mask in masks.items():
                _append_slice(
                    rows,
                    fold=fold,
                    validation=validation,
                    family=family,
                    slice_name=f"{entity}_history",
                    value=value,
                    mask=mask,
                )
    return pd.DataFrame(rows)


def _integrity_summary(frame: pd.DataFrame, plan: ValidationPlan) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in plan.execution_folds():
        train_seasons = frame.iloc[fold.train_positions]["season"]
        validation_seasons = frame.iloc[fold.validation_positions]["season"]
        inner = fold.inner_stop_split
        inner_rows = (
            np.empty(0, dtype=np.int64)
            if inner is None
            else np.union1d(inner.train_positions, inner.validation_positions)
        )
        rows.append(
            {
                "fold": fold.name,
                "train_max_season": int(train_seasons.max()),
                "validation_min_season": int(validation_seasons.min()),
                "strict_forward_season_order": bool(train_seasons.max() < validation_seasons.min()),
                "train_validation_position_overlap": int(
                    np.intersect1d(fold.train_positions, fold.validation_positions).size
                ),
                "train_validation_row_id_overlap": int(
                    len(set(fold.train_row_ids).intersection(fold.validation_row_ids))
                ),
                "source_position_order_strict": bool(
                    fold.train_positions.max() < fold.validation_positions.min()
                ),
                "inner_defined": inner is not None,
                "inner_plus_stop_equals_outer_train": bool(
                    inner is None or np.array_equal(inner_rows, fold.train_positions)
                ),
                "inner_outer_validation_overlap": int(
                    0
                    if inner is None
                    else np.intersect1d(inner_rows, fold.validation_positions).size
                ),
            }
        )
    return pd.DataFrame(rows)


def _fold_manifest(frame: pd.DataFrame, fold: SeasonFold) -> dict[str, object]:
    inner = fold.inner_stop_split
    result: dict[str, object] = {
        "name": fold.name,
        "role": fold.role,
        "train_seasons": list(fold.train_seasons),
        "validation_season": fold.validation_season,
        "train_rows": fold.train_row_count,
        "validation_rows": fold.validation_row_count,
        "train_row_id_sequence_sha256": _sha256_identifiers(
            frame.iloc[fold.train_positions][ID_COL]
        ),
        "validation_row_id_sequence_sha256": _sha256_identifiers(
            frame.iloc[fold.validation_positions][ID_COL]
        ),
    }
    if inner is None:
        result["inner_stop_split"] = None
    else:
        result["inner_stop_split"] = {
            "train_seasons": list(inner.train_seasons),
            "stop_season": inner.validation_season,
            "train_rows": inner.train_row_count,
            "stop_rows": inner.validation_row_count,
            "train_row_id_sequence_sha256": _sha256_identifiers(
                frame.iloc[inner.train_positions][ID_COL]
            ),
            "stop_row_id_sequence_sha256": _sha256_identifiers(
                frame.iloc[inner.validation_positions][ID_COL]
            ),
        }
    return result


def build_outputs(input_path: Path, output_dir: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("train.csv column order does not match the official training schema")

    plan = build_validation_plan(frame, include_recent_only=True)
    season_summary = _season_summary(frame)
    fold_summary = _fold_summary(frame, plan)
    subgroup_summary = _subgroup_summary(frame, plan)
    integrity_summary = _integrity_summary(frame, plan)

    if int(season_summary["rows"].sum()) != len(frame):
        raise AssertionError("season summary does not reconcile to the input row count")
    if not integrity_summary["strict_forward_season_order"].all():
        raise AssertionError("at least one fold is not strictly forward in time")
    if integrity_summary["train_validation_row_id_overlap"].sum() != 0:
        raise AssertionError("at least one fold has overlapping row IDs")
    if integrity_summary["inner_outer_validation_overlap"].sum() != 0:
        raise AssertionError("outer validation leaked into an inner split")

    manifest = {
        "schema_version": 1,
        "scope": "validation design and split audit only; no model or training executed",
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": _sha256_file(input_path),
        },
        "dataset": {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "seasons": list(EXPECTED_SEASONS),
            "row_id_unique": bool(frame[ID_COL].is_unique),
            "target_values": sorted(int(value) for value in frame[TARGET_COL].unique()),
        },
        "policy": {
            "canonical_outer_folds": [
                "rolling_origin_2022",
                "rolling_origin_2023",
                "rolling_origin_2024",
            ],
            "primary_recentness_anchor": "rolling_origin_2024",
            "anchor_is_untouched": False,
            "diagnostic_only_fold": "recent_only_2023_to_2024",
            "selection_rule": (
                "Select epoch/checkpoint procedure only on each fold's inner train/stop; "
                "outer validation labels are excluded from selection."
            ),
            "preprocessing_rule": (
                "Create a fresh preprocessor and fit every stateful statistic on the "
                "matching inner or outer training rows only."
            ),
            "trackman_rule": (
                "For target season Y, use only TrackMan source seasons strictly earlier than Y "
                "and record source provenance."
            ),
        },
        "folds": [_fold_manifest(frame, fold) for fold in plan.execution_folds()],
        "future_reporting_contract": {
            "primary_metric": "Brier score per outer fold",
            "derived_metrics": ["Brier skill", "official score"],
            "secondary_metrics": ["log loss", "ECE"],
            "required_slices": [
                "game_type F/R",
                "entity both_seen/pitcher_only/batter_only/neither_seen",
                "recent history missing/observed",
                "pitcher and batter cold-start/low-history",
            ],
            "canonical_aggregate": "Report folds separately; any 3-fold mean is secondary.",
        },
        "prohibited_in_this_stage": [
            "model construction",
            "model fitting",
            "prediction generation",
            "prediction metric computation",
            "hyperparameter tuning",
            "calibration or ensembling",
            "inference or submission",
        ],
        "outputs": [
            "season_summary.csv",
            "fold_summary.csv",
            "subgroup_summary.csv",
            "temporal_integrity.csv",
            "manifest.json",
        ],
    }

    _write_csv(season_summary, output_dir / "season_summary.csv")
    _write_csv(fold_summary, output_dir / "fold_summary.csv")
    _write_csv(subgroup_summary, output_dir / "subgroup_summary.csv")
    _write_csv(integrity_summary, output_dir / "temporal_integrity.csv")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_outputs(args.input, args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "scope": manifest["scope"],
                "rows": manifest["dataset"]["rows"],
                "folds": [fold["name"] for fold in manifest["folds"]],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
