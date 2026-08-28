#!/usr/bin/env python3
"""One fixed low-LR polish epoch using only post-2022 rows.

The base artifact must itself prove zero exposure to pre-2023 F.  The fitted
preprocessor is retained so that historical-R entity coverage is not lost,
while every gradient in this stage comes from the new regime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import ID_COL, TARGET_COL
from lg_control.exclusive_outcome import (
    ExclusiveOutcomeGatedCrossModel,
    ExclusiveOutcomeLossConfig,
    recover_fold_train_exclusive_targets,
    train_exclusive_outcome_model,
)
from lg_control.rate_basis_features import RateBasisAugmentedPreprocessor
from lg_control.schema import validate_input_frame
from lg_control.training import TrainConfig

from train_regime_cross import predict_head_probabilities, save_state_npz, sliced_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-year", type=int, required=True)
    parser.add_argument("--inner-validation-start-month", type=int, default=None)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--scheduler-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--brier-weight", type=float, default=4.0)
    parser.add_argument("--ball-strike-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-mode", choices=("best", "last"), default="last"
    )
    return parser.parse_args()


def _load_base(
    directory: Path,
) -> tuple[RateBasisAugmentedPreprocessor, ExclusiveOutcomeGatedCrossModel, dict]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    audit = metrics.get("recent_regime_audit")
    if not isinstance(audit, dict) or audit.get("obsolete_pre_regime_f_rows") != 0:
        raise ValueError("base artifact does not prove zero obsolete-F exposure")
    if metrics.get("feature_set") != "rate_basis" or not metrics.get("exclusive_outcome"):
        raise ValueError("base artifact is not the fixed rate-basis exclusive model")
    preprocessor_state = json.loads(
        (directory / "preprocessor.json").read_text(encoding="utf-8")
    )
    preprocessor = RateBasisAugmentedPreprocessor.from_state_dict(preprocessor_state)
    model_config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
    model = ExclusiveOutcomeGatedCrossModel.from_config(model_config)
    with np.load(directory / "model_state.npz", allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    model.load_state_dict(state, strict=True)
    return preprocessor, model, metrics


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.epochs <= 4:
        raise ValueError("recent polish epochs must be in [1, 4]")
    if args.validation_year < args.regime_start_year:
        raise ValueError("validation precedes the new regime")
    torch.set_num_threads(args.cpu_threads)
    data = pd.read_csv(args.input, low_memory=False)
    validate_input_frame(data, require_target=True, allow_target=True)
    if args.inner_validation_start_month is None:
        train = data.loc[
            (data["season"] >= args.regime_start_year)
            & (data["season"] < args.validation_year)
        ].copy()
        validation = data.loc[data["season"].eq(args.validation_year)].copy()
    else:
        if args.validation_year != args.regime_start_year:
            raise ValueError("inner polish must use the first new-regime year")
        train = data.loc[
            data["season"].eq(args.validation_year)
            & data["game_month"].lt(args.inner_validation_start_month)
        ].copy()
        validation = data.loc[
            data["season"].eq(args.validation_year)
            & data["game_month"].ge(args.inner_validation_start_month)
        ].copy()
    if train.empty or validation.empty:
        raise ValueError("recent polish produced an empty partition")
    if (train["season"] < args.regime_start_year).any():
        raise RuntimeError("old rows entered the recent-only polish")

    preprocessor, model, base_metrics = _load_base(args.base_dir)
    train_arrays = preprocessor.transform(train)
    validation_arrays = preprocessor.transform(validation)
    targets = recover_fold_train_exclusive_targets(train, partition="fold_train")
    config = TrainConfig(
        epochs=args.epochs,
        scheduler_epochs=args.scheduler_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        brier_weight=args.brier_weight,
        patience=args.epochs,
        seed=args.seed,
        num_workers=0,
        use_amp=True,
        verbose=True,
    )
    result = train_exclusive_outcome_model(
        model,
        train_arrays["categorical"],
        train_arrays["numerical"],
        targets,
        validation_arrays["categorical"],
        validation_arrays["numerical"],
        validation[TARGET_COL].to_numpy(dtype=np.float32),
        config=config,
        loss_config=ExclusiveOutcomeLossConfig(
            ball_strike_weight=args.ball_strike_weight
        ),
        device=args.device,
        sample_weights=np.ones(len(train), dtype=np.float32),
        checkpoint_mode=args.checkpoint_mode,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            ID_COL: validation[ID_COL].to_numpy(copy=True),
            "season": validation["season"].to_numpy(copy=True),
            "game_type": validation["game_type"].to_numpy(copy=True),
            "y_true": validation[TARGET_COL].to_numpy(dtype=np.int8),
            "prediction": result.probabilities,
        }
    )
    for name, values in predict_head_probabilities(
        model,
        validation_arrays["categorical"],
        validation_arrays["numerical"],
        batch_size=args.eval_batch_size,
        device=args.device,
    ).items():
        predictions[name] = values
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "preprocessor.json").write_text(
        json.dumps(preprocessor.to_state_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "model_config.json").write_text(
        json.dumps(model.get_config(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_state_npz(model, args.output_dir / "model_state.npz")
    payload = {
        "stage": "recent_only_polish",
        "base_dir": str(args.base_dir),
        "base_metrics": base_metrics["metrics"],
        "validation_year": args.validation_year,
        "inner_validation_start_month": args.inner_validation_start_month,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_seasons": sorted(int(value) for value in train["season"].unique()),
        "obsolete_pre_regime_f_rows": int(
            ((train["season"] < args.regime_start_year) & train["game_type"].eq("F")).sum()
        ),
        "exclusive_valid_rows": int(targets.outcome_mask.sum()),
        "metrics": result.best_metrics,
        "slices": sliced_metrics(validation, result.probabilities),
        "history": result.history,
        "selected_epoch": result.best_epoch,
        "checkpoint_mode": args.checkpoint_mode,
        "train_config": config.to_dict(),
        "model_config": model.get_config(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"brier={result.best_metrics['brier']:.8f} "
        f"score={result.best_metrics['competition_score']:.2f}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run(parse_args())
