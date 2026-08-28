#!/usr/bin/env python3
"""Join nested DL OOF with the frozen public-v7 rolling predictions safely."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPORTS = PROJECT_ROOT.parent / "open (3)" / "LG-aimers" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nested-oof",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_nested_oof" / "nested_oof.csv",
    )
    parser.add_argument(
        "--v7-oof",
        type=Path,
        default=EXTERNAL_REPORTS / "v6_delta_logit_b015_6seed_oof_predictions.parquet",
    )
    parser.add_argument(
        "--v7-2023-stress",
        type=Path,
        default=EXTERNAL_REPORTS / "v6_delta_logit_b015_2023_stress_predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_meta_oof",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_score(y_true: np.ndarray, prediction: np.ndarray) -> float:
    target_rate = float(np.mean(y_true))
    denominator = target_rate * (1.0 - target_rate)
    brier = float(np.mean(np.square(y_true - prediction)))
    return max(0.0, 100000.0 * (1.0 - brier / denominator))


def metric_payload(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(np.mean(np.square(y_true - prediction))),
        "competition_score": official_score(y_true, prediction),
        "prediction_mean": float(np.mean(prediction)),
    }


def canonical_external_game_type(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(int)
    if not set(numeric.unique()).issubset({0, 1}):
        raise ValueError("external game_type must use the frozen 0=F, 1=R mapping")
    return numeric.map({0: "F", 1: "R"})


def load_external(main_path: Path, stress_path: Path, seasons: set[int]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    main_seasons = seasons - {2023}
    if main_seasons:
        main = pd.read_parquet(main_path)
        pieces.append(main.loc[main["season"].isin(main_seasons)].copy())
    if 2023 in seasons:
        stress = pd.read_parquet(stress_path)
        pieces.append(stress.loc[stress["season"].eq(2023)].copy())
    external = pd.concat(pieces, ignore_index=True)
    if external["row_id"].duplicated().any():
        raise ValueError("external v7 OOF row_id must be unique")
    seed_columns = sorted(
        column
        for column in external.columns
        if column.startswith("prediction_candidate_seed_")
    )
    if not seed_columns:
        raise ValueError("external v7 OOF has no seed predictions")
    external["v7_seed_std"] = external[seed_columns].std(axis=1, ddof=0)
    external["external_game_type"] = canonical_external_game_type(
        external["game_type"]
    )
    keep = [
        "row_id",
        "season",
        "control_success",
        "external_game_type",
        "game_month",
        "season_success_delta_k20",
        "prediction_candidate",
        "v7_seed_std",
    ]
    return external[keep].rename(
        columns={
            "control_success": "external_y_true",
            "prediction_candidate": "prediction_v7",
        }
    )


def oracle_convex_weight(
    y_true: np.ndarray, base: np.ndarray, challenger: np.ndarray
) -> dict[str, float]:
    direction = challenger - base
    denominator = float(np.dot(direction, direction))
    challenger_weight = 0.0
    if denominator > 0.0:
        challenger_weight = float(
            np.clip(-np.dot(base - y_true, direction) / denominator, 0.0, 1.0)
        )
    prediction = base + challenger_weight * direction
    return {
        "v7_weight": 1.0 - challenger_weight,
        "dl_equal_weight": challenger_weight,
        **metric_payload(y_true, prediction),
        "diagnostic_only": True,
    }


def main() -> None:
    args = parse_args()
    nested_path = args.nested_oof.resolve()
    main_path = args.v7_oof.resolve()
    stress_path = args.v7_2023_stress.resolve()
    output_dir = args.output_dir.resolve()
    for path in (nested_path, main_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    nested = pd.read_csv(nested_path)
    required = {
        "row_id",
        "season",
        "game_type",
        "y_true",
        "prediction_rate_exclusive",
        "prediction_temporal_te",
        "prediction_equal",
    }
    missing = required - set(nested.columns)
    if missing:
        raise ValueError(f"nested OOF is missing columns: {sorted(missing)}")
    if nested["row_id"].duplicated().any():
        raise ValueError("nested OOF row_id must be unique")
    seasons = set(pd.to_numeric(nested["season"], errors="raise").astype(int))
    if 2023 in seasons and not stress_path.is_file():
        raise FileNotFoundError(stress_path)

    external = load_external(main_path, stress_path, seasons)
    merged = nested.merge(external, on=["row_id", "season"], validate="one_to_one")
    if len(merged) != len(nested):
        raise RuntimeError("external v7 OOF does not cover every nested DL row")
    if not np.array_equal(
        merged["y_true"].to_numpy(dtype=np.int8),
        merged["external_y_true"].to_numpy(dtype=np.int8),
    ):
        raise RuntimeError("external and DL targets differ")
    if not merged["game_type"].astype(str).str.upper().equals(
        merged["external_game_type"]
    ):
        raise RuntimeError("external and DL game_type differ")

    merged = merged.drop(
        columns=[
            "external_y_true",
            "external_game_type",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hybrid_meta_oof.parquet"
    merged.to_parquet(output_path, index=False)

    metrics: dict[str, object] = {}
    for season, frame in merged.groupby("season", sort=True):
        y_true = frame["y_true"].to_numpy(dtype=np.float64)
        v7 = frame["prediction_v7"].to_numpy(dtype=np.float64)
        equal = frame["prediction_equal"].to_numpy(dtype=np.float64)
        metrics[str(int(season))] = {
            "rows": len(frame),
            "target_rate": float(y_true.mean()),
            "v7": metric_payload(y_true, v7),
            "rate_exclusive": metric_payload(
                y_true,
                frame["prediction_rate_exclusive"].to_numpy(dtype=np.float64),
            ),
            "temporal_te": metric_payload(
                y_true,
                frame["prediction_temporal_te"].to_numpy(dtype=np.float64),
            ),
            "dl_equal": metric_payload(y_true, equal),
            "v7_dl_equal_50_50": metric_payload(y_true, 0.5 * (v7 + equal)),
            "oracle_convex": oracle_convex_weight(y_true, v7, equal),
        }
    payload = {
        "nested_oof": str(nested_path),
        "nested_oof_sha256": sha256(nested_path),
        "v7_oof": str(main_path),
        "v7_oof_sha256": sha256(main_path),
        "v7_2023_stress": str(stress_path),
        "rows": len(merged),
        "seasons": sorted(seasons),
        "join_key": ["row_id", "season"],
        "forbidden_game_cluster_in_output": "_game_cluster" not in merged.columns,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"hybrid_meta_oof={output_path} rows={len(merged):,}")
    for season, values in metrics.items():
        print(
            f"season={season} v7={values['v7']['competition_score']:.2f} "
            f"dl={values['dl_equal']['competition_score']:.2f} "
            f"blend={values['v7_dl_equal_50_50']['competition_score']:.2f}"
        )


if __name__ == "__main__":
    main()
