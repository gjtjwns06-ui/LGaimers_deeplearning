#!/usr/bin/env python3
"""Train one or more final models on every labelled season and save a bundle."""

from __future__ import annotations

import argparse
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
from lg_control.calibration import AffineLogitCalibrator
from lg_control.constants import TARGET_COL
from lg_control.features import FeaturePreprocessor
from lg_control.model import TabularResidualModel
from lg_control.schema import validate_input_frame
from lg_control.training import TrainConfig, train_fixed_epochs
from lg_control.trackman import JOINED_TRACKMAN_FEATURE_COLUMNS, join_trackman_context


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_bundle"))
    parser.add_argument("--calibrator", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--epochs", type=positive_int, default=8)
    parser.add_argument("--batch-size", type=positive_int, default=4096)
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument("--num-blocks", type=positive_int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--embedding-dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--recency-decay", type=float, default=1.0)
    parser.add_argument("--min-category-frequency", type=positive_int, default=2)
    parser.add_argument("--smoothing-alpha", type=float, default=50.0)
    parser.add_argument("--trackman-lookup", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=positive_int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0.0 < args.recency_decay <= 1.0:
        raise ValueError("recency_decay must be in (0, 1]")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.embedding_dropout < 1.0:
        raise ValueError("dropout values must be in [0, 1)")
    data = pd.read_csv(args.input, low_memory=False)
    validate_input_frame(data, require_target=True, allow_target=True)
    extra_numeric_columns: list[str] = []
    if args.trackman_lookup is not None:
        data = join_trackman_context(data, args.trackman_lookup)
        extra_numeric_columns = list(JOINED_TRACKMAN_FEATURE_COLUMNS)
    if args.max_rows is not None and len(data) > args.max_rows:
        rng = np.random.default_rng(args.seeds[0])
        positions = np.sort(rng.choice(len(data), args.max_rows, replace=False))
        data = data.iloc[positions].copy()

    preprocessor = FeaturePreprocessor(
        min_category_frequency=args.min_category_frequency,
        smoothing_alpha=args.smoothing_alpha,
        extra_numeric_columns=extra_numeric_columns,
    ).fit(data)
    arrays = preprocessor.transform(data)
    max_season = int(data["season"].max())
    weights = np.power(
        args.recency_decay,
        max_season - data["season"].to_numpy(dtype=np.float32),
    ).astype(np.float32)
    weights /= float(weights.mean())

    models: list[TabularResidualModel] = []
    histories: dict[str, object] = {}
    for seed in dict.fromkeys(args.seeds):
        print(f"training final seed={seed} rows={len(data):,}", flush=True)
        model = TabularResidualModel(
            category_cardinalities=preprocessor.category_cardinalities,
            num_numerical=len(preprocessor.numerical_columns),
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            embedding_dropout=args.embedding_dropout,
        )
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            brier_weight=args.brier_weight,
            seed=seed,
            num_workers=args.num_workers,
        )
        histories[str(seed)] = train_fixed_epochs(
            model,
            arrays["categorical"],
            arrays["numerical"],
            data[TARGET_COL].to_numpy(dtype=np.float32),
            config=config,
            device=args.device,
            sample_weights=weights,
        )
        models.append(model.cpu())

    calibrator = AffineLogitCalibrator.load(args.calibrator) if args.calibrator else None
    metadata: dict[str, object] = {
        "rows": len(data),
        "seasons": sorted(int(value) for value in data["season"].unique()),
        "seeds": list(dict.fromkeys(args.seeds)),
        "epochs": args.epochs,
        "recency_decay": args.recency_decay,
        "histories": histories,
        "torch_version": torch.__version__,
        "trackman_lookup": str(args.trackman_lookup) if args.trackman_lookup else None,
    }
    save_model_bundle(
        args.output_dir,
        preprocessor=preprocessor,
        models=models,
        calibrator=calibrator,
        metadata=metadata,
    )
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved final bundle: {args.output_dir}", flush=True)
    return metadata


if __name__ == "__main__":
    run(parse_args())
