#!/usr/bin/env python3
"""Train leakage-safe constant and random-forest time-split baselines.

Each validation fold uses rows from earlier seasons only.  The script writes one
prediction CSV per fold and a JSON file containing fold and aggregate metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROW_ID_COLUMN = "row_id"
SEASON_COLUMN = "season"
TARGET_COLUMN = "control_success"
DEFAULT_FOLDS = (2022, 2023, 2024)


def positive_int(value: str) -> int:
    """Argparse converter that accepts integers greater than zero."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate constant-prior and Random Forest baselines with expanding "
            "season folds (train: season < fold, validation: season == fold)."
        )
    )
    parser.add_argument(
        "--input",
        "--input-path",
        "--train-path",
        dest="input_path",
        type=Path,
        default=Path("data/train.csv"),
        help="Training CSV path (default: data/train.csv).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(DEFAULT_FOLDS),
        metavar="YEAR",
        help="Validation seasons (default: 2022 2023 2024).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=positive_int,
        default=None,
        help="Deterministically sample at most this many training rows per fold.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=positive_int,
        default=None,
        help="Deterministically sample at most this many validation rows per fold.",
    )
    parser.add_argument(
        "--n-estimators",
        type=positive_int,
        default=100,
        help="Number of Random Forest trees (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline"),
        help="Directory for metrics and fold predictions (default: artifacts/baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by sampling and the model (default: 42).",
    )
    return parser.parse_args(argv)


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the training CSV and enforce the baseline's input contract."""
    if not path.is_file():
        raise FileNotFoundError(f"training CSV does not exist: {path}")

    data = pd.read_csv(path, dtype={ROW_ID_COLUMN: "string"})
    required = {ROW_ID_COLUMN, SEASON_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"training CSV is missing required columns: {missing}")
    if not data.columns.is_unique:
        raise ValueError("training CSV contains duplicate column names")
    if data.empty:
        raise ValueError("training CSV contains no rows")

    row_ids = data[ROW_ID_COLUMN]
    if row_ids.isna().any() or row_ids.str.strip().eq("").any():
        raise ValueError("row_id must not contain missing or blank values")
    if not row_ids.is_unique:
        duplicate = row_ids[row_ids.duplicated()].iloc[0]
        raise ValueError(f"row_id must be unique; duplicate found: {duplicate!r}")

    target = data[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values")
    observed_targets = set(pd.unique(target))
    if observed_targets != {0, 1}:
        raise ValueError(
            f"{TARGET_COLUMN} must contain both binary values 0 and 1; "
            f"found {sorted(observed_targets, key=str)}"
        )
    data[TARGET_COLUMN] = target.astype(np.int8)

    seasons = pd.to_numeric(data[SEASON_COLUMN], errors="coerce")
    if seasons.isna().any() or not np.isfinite(seasons.to_numpy()).all():
        raise ValueError(f"{SEASON_COLUMN} must contain finite numeric values")
    if not np.equal(seasons, np.floor(seasons)).all():
        raise ValueError(f"{SEASON_COLUMN} must contain integer years")
    data[SEASON_COLUMN] = seasons.astype(np.int32)

    feature_columns = [
        column
        for column in data.columns
        if column not in {ROW_ID_COLUMN, TARGET_COLUMN}
    ]
    if not feature_columns:
        raise ValueError("training CSV does not contain any model features")

    numeric = data[feature_columns].select_dtypes(include=[np.number])
    for column in numeric.columns:
        if np.isinf(numeric[column].to_numpy(copy=False)).any():
            raise ValueError(
                f"numeric feature column {column!r} must not contain infinite values"
            )
    return data


def limited_positions(
    positions: np.ndarray,
    limit: int | None,
    *,
    seed: int,
    fold: int,
    stream: int,
) -> np.ndarray:
    """Return a deterministic subset while preserving the source row order."""
    if limit is None or positions.size <= limit:
        return positions
    seed_sequence = np.random.SeedSequence([seed, fold, stream])
    rng = np.random.default_rng(seed_sequence)
    return np.sort(rng.choice(positions, size=limit, replace=False))


def make_model(
    frame: pd.DataFrame, *, n_estimators: int, seed: int
) -> tuple[Pipeline, list[str], list[str]]:
    """Build preprocessing and Random Forest stages from training dtypes."""
    categorical_columns = frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [
        column for column in frame.columns if column not in categorical_columns
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_columns:
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
        transformers.append(("categorical", categorical_encoder, categorical_columns))
    if numeric_columns:
        numeric_imputer = SimpleImputer(
            strategy="median", keep_empty_features=True, copy=False
        )
        transformers.append(("numeric", numeric_imputer, numeric_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=200,
        random_state=seed,
        n_jobs=-1,
    )
    model = Pipeline([("preprocess", preprocessor), ("model", forest)])
    return model, categorical_columns, numeric_columns


def positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Extract P(y=1), including the valid single-class smoke-test case."""
    probabilities = model.predict_proba(features)
    classes = model.named_steps["model"].classes_
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size == 0:
        return np.zeros(len(features), dtype=np.float64)
    return probabilities[:, int(positive_indices[0])].astype(np.float64, copy=False)


def binary_brier(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - y_true)))


def aggregate_metrics(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = (
        "constant_brier",
        "official_reference_brier",
        "rf_brier",
        "rf_brier_skill",
        "rf_logloss",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([fold[name] for fold in folds], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_and_validate(args.input_path)
    fold_years = list(dict.fromkeys(args.folds))
    if not fold_years:
        raise ValueError("at least one validation fold is required")

    available_seasons = set(data[SEASON_COLUMN].unique().tolist())
    invalid_folds = [fold for fold in fold_years if fold not in available_seasons]
    if invalid_folds:
        raise ValueError(
            f"validation folds are absent from {SEASON_COLUMN}: {invalid_folds}"
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in {ROW_ID_COLUMN, TARGET_COLUMN}
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, Any]] = []

    for fold in fold_years:
        train_positions = np.flatnonzero(data[SEASON_COLUMN].to_numpy() < fold)
        val_positions = np.flatnonzero(data[SEASON_COLUMN].to_numpy() == fold)
        if train_positions.size == 0:
            raise ValueError(f"fold {fold} has no rows from an earlier season")
        if val_positions.size == 0:
            raise ValueError(f"fold {fold} has no validation rows")

        train_positions = limited_positions(
            train_positions,
            args.max_train_rows,
            seed=args.seed,
            fold=fold,
            stream=0,
        )
        val_positions = limited_positions(
            val_positions,
            args.max_val_rows,
            seed=args.seed,
            fold=fold,
            stream=1,
        )

        train = data.iloc[train_positions]
        validation = data.iloc[val_positions]
        x_train = train[feature_columns]
        y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
        x_val = validation[feature_columns]
        y_val = validation[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)

        prior = float(y_train.mean())
        constant_prediction = np.full(y_val.shape, prior, dtype=np.float64)
        constant_brier = binary_brier(y_val, constant_prediction)
        validation_rate = float(y_val.mean())
        official_reference_brier = validation_rate * (1.0 - validation_rate)

        model, categorical_columns, numeric_columns = make_model(
            x_train,
            n_estimators=args.n_estimators,
            seed=args.seed,
        )
        print(
            f"fold={fold} train_rows={len(train):,} val_rows={len(validation):,} "
            f"prior={prior:.6f}",
            flush=True,
        )
        model.fit(x_train, y_train)
        prediction = positive_probability(model, x_val)
        rf_brier = binary_brier(y_val, prediction)
        brier_skill = (
            float(1.0 - rf_brier / official_reference_brier)
            if official_reference_brier > 0.0
            else 0.0
        )
        rf_logloss = float(log_loss(y_val, prediction, labels=[0, 1]))

        prediction_path = args.output_dir / f"fold_{fold}_predictions.csv"
        pd.DataFrame(
            {
                ROW_ID_COLUMN: validation[ROW_ID_COLUMN].to_numpy(),
                "y_true": y_val,
                "pred": prediction,
            }
        ).to_csv(prediction_path, index=False)

        result = {
            "fold": int(fold),
            "train_seasons": sorted(
                int(year) for year in train[SEASON_COLUMN].unique().tolist()
            ),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_positive_rate": prior,
            "validation_positive_rate": validation_rate,
            "constant_brier": constant_brier,
            "official_reference_brier": official_reference_brier,
            "rf_brier": rf_brier,
            "rf_brier_skill": brier_skill,
            "rf_logloss": rf_logloss,
            "prediction_file": prediction_path.name,
            "categorical_feature_count": len(categorical_columns),
            "numeric_feature_count": len(numeric_columns),
        }
        fold_results.append(result)
        print(
            f"fold={fold} constant_brier={constant_brier:.8f} "
            f"rf_brier={rf_brier:.8f} rf_brier_skill={brier_skill:.8f} "
            f"rf_logloss={rf_logloss:.8f}",
            flush=True,
        )

        del model, train, validation, x_train, x_val
        gc.collect()

    metrics: dict[str, Any] = {
        "input_path": str(args.input_path),
        "dataset_rows": int(len(data)),
        "feature_count": len(feature_columns),
        "folds": fold_years,
        "config": {
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "n_estimators": args.n_estimators,
            "max_depth": 10,
            "min_samples_leaf": 200,
            "seed": args.seed,
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "fold_metrics": fold_results,
        "aggregate": aggregate_metrics(fold_results),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved metrics: {metrics_path}", flush=True)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
