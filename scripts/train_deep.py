#!/usr/bin/env python3
"""Train the row-independent embedding Residual MLP on forward season folds."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_control.artifacts import save_model_bundle
from lg_control.constants import ID_COL, TARGET_COL
from lg_control.features import FeaturePreprocessor
from lg_control.metrics import metric_dict
from lg_control.model import TabularResidualModel
from lg_control.schema import validate_input_frame
from lg_control.training import TrainConfig, train_model
from lg_control.trackman import JOINED_TRACKMAN_FEATURE_COLUMNS, join_trackman_context


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/deep_cv"))
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=4096)
    parser.add_argument("--eval-batch-size", type=positive_int, default=16384)
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument("--num-blocks", type=positive_int, default=3)
    parser.add_argument("--dropout", type=probability, default=0.15)
    parser.add_argument("--embedding-dropout", type=probability, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    parser.add_argument("--max-train-rows", type=positive_int, default=None)
    parser.add_argument("--max-val-rows", type=positive_int, default=None)
    parser.add_argument("--min-category-frequency", type=positive_int, default=2)
    parser.add_argument("--smoothing-alpha", type=float, default=50.0)
    parser.add_argument(
        "--trackman-lookup",
        type=Path,
        default=None,
        help="Optional prior-season context lookup built by build_trackman_context.py.",
    )
    parser.add_argument(
        "--recency-decay",
        type=float,
        default=1.0,
        help="Per-year multiplier in (0,1]; 1 disables recency weighting.",
    )
    return parser.parse_args()


def deterministic_limit(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.copy()
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=limit, replace=False))
    return frame.iloc[positions].copy()


def subgroup_metrics(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    masks = {
        "all": np.ones(len(val_frame), dtype=bool),
        "game_type_F": val_frame["game_type"].astype(str).eq("F").to_numpy(),
        "game_type_R": val_frame["game_type"].astype(str).eq("R").to_numpy(),
        "new_pitcher": ~val_frame["pitcher_id"].isin(set(train_frame["pitcher_id"])).to_numpy(),
        "new_batter": ~val_frame["batter_id"].isin(set(train_frame["batter_id"])).to_numpy(),
        "low_pitcher_history": (val_frame["asof_pitcher_n"].to_numpy() < 50),
    }
    target = val_frame[TARGET_COL].to_numpy(dtype=np.float64)
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        result[name] = {"rows": int(mask.sum()), **metric_dict(target[mask], probabilities[mask])}
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0.0 < args.recency_decay <= 1.0:
        raise ValueError("recency_decay must be in (0, 1]")
    data = pd.read_csv(args.input, low_memory=False)
    validate_input_frame(data, require_target=True, allow_target=True)
    extra_numeric_columns: list[str] = []
    if args.trackman_lookup is not None:
        data = join_trackman_context(data, args.trackman_lookup)
        extra_numeric_columns = list(JOINED_TRACKMAN_FEATURE_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict[str, object]] = []
    oof_pieces: list[pd.DataFrame] = []
    for fold_index, val_year in enumerate(dict.fromkeys(args.folds)):
        train_frame = data.loc[data["season"] < val_year]
        val_frame = data.loc[data["season"] == val_year]
        if train_frame.empty or val_frame.empty:
            raise ValueError(f"fold {val_year} has empty train or validation data")
        train_frame = deterministic_limit(
            train_frame, args.max_train_rows, args.seed + 1000 * fold_index
        )
        val_frame = deterministic_limit(
            val_frame, args.max_val_rows, args.seed + 1000 * fold_index + 1
        )
        print(
            f"fold={val_year} train_rows={len(train_frame):,} val_rows={len(val_frame):,}",
            flush=True,
        )

        preprocessor = FeaturePreprocessor(
            min_category_frequency=args.min_category_frequency,
            smoothing_alpha=args.smoothing_alpha,
            extra_numeric_columns=extra_numeric_columns,
        ).fit(train_frame)
        train_arrays = preprocessor.transform(train_frame)
        val_arrays = preprocessor.transform(val_frame)
        model = TabularResidualModel(
            category_cardinalities=preprocessor.category_cardinalities,
            num_numerical=len(preprocessor.numerical_columns),
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            embedding_dropout=args.embedding_dropout,
        )
        weights = np.power(
            args.recency_decay,
            (val_year - 1) - train_frame["season"].to_numpy(dtype=np.float32),
        ).astype(np.float32)
        weights /= float(weights.mean())
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            brier_weight=args.brier_weight,
            patience=args.patience,
            seed=args.seed + fold_index,
            num_workers=args.num_workers,
            use_amp=True,
        )
        result = train_model(
            model,
            train_arrays["categorical"],
            train_arrays["numerical"],
            train_frame[TARGET_COL].to_numpy(dtype=np.float32),
            val_arrays["categorical"],
            val_arrays["numerical"],
            val_frame[TARGET_COL].to_numpy(dtype=np.float32),
            config=config,
            device=args.device,
            sample_weights=weights,
        )

        fold_dir = args.output_dir / f"fold_{val_year}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        prediction_frame = pd.DataFrame(
            {
                ID_COL: val_frame[ID_COL].to_numpy(),
                "season": val_frame["season"].to_numpy(),
                "y_true": val_frame[TARGET_COL].to_numpy(dtype=np.int8),
                "prediction": result.probabilities,
            }
        )
        prediction_path = fold_dir / "predictions.csv"
        prediction_frame.to_csv(prediction_path, index=False)
        oof_pieces.append(prediction_frame)
        summary: dict[str, object] = {
            "fold": val_year,
            "train_rows": len(train_frame),
            "validation_rows": len(val_frame),
            "train_seasons": sorted(int(value) for value in train_frame["season"].unique()),
            "best_epoch": result.best_epoch,
            "metrics": result.best_metrics,
            "subgroups": subgroup_metrics(train_frame, val_frame, result.probabilities),
            "history": result.history,
            "model_config": model.get_config(),
            "train_config": config.to_dict(),
            "recency_decay": args.recency_decay,
            "trackman_lookup": str(args.trackman_lookup) if args.trackman_lookup else None,
        }
        save_model_bundle(
            fold_dir / "bundle",
            preprocessor=preprocessor,
            models=[model],
            metadata={"validation_year": val_year, "best_epoch": result.best_epoch},
        )
        (fold_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_summaries.append(summary)
        print(
            f"fold={val_year} best_epoch={result.best_epoch} "
            f"brier={result.best_metrics['brier']:.8f} "
            f"skill={result.best_metrics['brier_skill']:.6f}",
            flush=True,
        )

        del train_arrays, val_arrays, model, preprocessor, result, train_frame, val_frame
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof = pd.concat(oof_pieces, ignore_index=True)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    payload: dict[str, object] = {
        "input": str(args.input),
        "folds": list(dict.fromkeys(args.folds)),
        "fold_summaries": fold_summaries,
        "torch_version": torch.__version__,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "trackman_lookup": str(args.trackman_lookup) if args.trackman_lookup else None,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    run(parse_args())
