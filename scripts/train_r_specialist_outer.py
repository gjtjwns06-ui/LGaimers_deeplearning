#!/usr/bin/env python3
"""Train a distinct R-only TabM specialist with rolling-origin confirmation."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_control.exclusive_outcome import recover_fold_train_exclusive_targets  # noqa: E402
from lg_control.rate_basis_features import RateBasisAugmentedPreprocessor  # noqa: E402
from lg_control.r_specialist import (  # noqa: E402
    RSpecialistConfig,
    RSpecialistTabM,
    RSpecialistTrainConfig,
    initialize_priors,
    predict_r_specialist,
    save_model_npz,
    train_r_specialist_fixed_epochs,
)


OUTPUT = ROOT / "artifacts/r_specialist_outer_v1"
MODEL_CONFIG = RSpecialistConfig(
    embedding_dim_cap=10,
    hidden_dim=64,
    ensemble_size=4,
    hidden_layers=2,
    dropout=0.10,
)
EPOCHS = 3


def metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    y = frame.control_success.to_numpy(float)
    p = np.asarray(probability, dtype=np.float64)
    rate = float(y.mean())
    brier = float(np.mean(np.square(y - p)))
    return {
        "rows": int(len(frame)),
        "rate": rate,
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "brier": brier,
        "score": max(0.0, 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def weights(frame: pd.DataFrame) -> np.ndarray:
    early = frame.balls_before.le(1) & frame.strikes_before.le(1)
    close_rate = frame.asof_pitcher_success_rate.between(0.45, 0.55, inclusive="both")
    result = np.ones(len(frame), dtype=np.float32)
    result[early.to_numpy()] *= 1.75
    result[close_rate.fillna(False).to_numpy()] *= 1.20
    return result


def fit_fold(
    data: pd.DataFrame,
    *,
    train_years: tuple[int, ...],
    validation_year: int,
    seed: int,
    save_directory: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit = data[data.season.isin(train_years)].reset_index(drop=True)
    validation = data[data.season.eq(validation_year)].reset_index(drop=True)
    print(
        f"fold train={train_years} rows={len(fit):,} validation={validation_year} "
        f"rows={len(validation):,}",
        flush=True,
    )
    started = time.perf_counter()
    preprocessor = RateBasisAugmentedPreprocessor(min_category_frequency=2).fit(fit)
    fit_arrays = preprocessor.transform(fit)
    validation_arrays = preprocessor.transform(validation)
    targets = recover_fold_train_exclusive_targets(fit, partition="fold_train")
    sample_weight = weights(fit)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RSpecialistTabM(preprocessor.feature_layout(), MODEL_CONFIG)
    initialize_priors(model, targets, sample_weight)
    config = RSpecialistTrainConfig(
        epochs=EPOCHS,
        batch_size=4096,
        eval_batch_size=16384,
        learning_rate=8e-4,
        weight_decay=2e-4,
        brier_weight=2.0,
        auxiliary_weight=0.10,
        seed=seed,
        verbose=True,
    )
    history = train_r_specialist_fixed_epochs(
        model,
        fit_arrays["categorical"],
        fit_arrays["numerical"],
        targets,
        sample_weight,
        config=config,
        device="cpu",
    )
    output = predict_r_specialist(
        model,
        validation_arrays["categorical"],
        validation_arrays["numerical"],
        batch_size=16384,
        device="cpu",
    )
    prediction = output["outcome_control_success"]
    result = validation[["row_id", "season", "game_month", "game_type", "balls_before", "strikes_before", "control_success"]].copy()
    result["prediction"] = prediction
    for name, values in output.items():
        result[f"head_{name}"] = values
    early = validation.balls_before.le(1) & validation.strikes_before.le(1)
    payload = {
        "train_years": list(train_years),
        "validation_year": validation_year,
        "seed": seed,
        "train_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "exclusive_valid_rows": int(targets.outcome_mask.sum()),
        "exclusive_valid_fraction": float(targets.outcome_mask.mean()),
        "metrics": metric(validation, prediction),
        "early_count_metrics": metric(validation.loc[early], prediction[early.to_numpy()]),
        "history": history,
        "seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(payload["metrics"], ensure_ascii=False), flush=True)
    if save_directory is not None:
        save_directory.mkdir(parents=True)
        (save_directory / "preprocessor.json").write_text(
            json.dumps(preprocessor.to_state_dict(), ensure_ascii=False), encoding="utf-8"
        )
        (save_directory / "model_config.json").write_text(
            json.dumps(model.to_config_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_model_npz(model, save_directory / f"model_seed_{seed}.npz")
        (save_directory / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": "r_specialist_tabm_ensemble_v1",
                    "model_files": [f"model_seed_{seed}.npz"],
                    "training_years": list(train_years),
                    "training_rows": int(len(fit)),
                    "seed": seed,
                    "epochs": EPOCHS,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    del fit_arrays, validation_arrays, targets, model, preprocessor, fit, validation
    gc.collect()
    return result, payload


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    data = raw[
        raw.game_type.astype(str).str.upper().eq("R")
        & raw.season.isin([2021, 2022, 2023, 2024])
    ].reset_index(drop=True)
    del raw
    gc.collect()

    inner_predictions, inner = fit_fold(
        data,
        train_years=(2021, 2022),
        validation_year=2023,
        seed=42,
        save_directory=None,
    )
    inner_predictions.to_parquet(OUTPUT / "inner_2023_predictions.parquet", index=False)
    outer_predictions, outer = fit_fold(
        data,
        train_years=(2022, 2023),
        validation_year=2024,
        seed=42,
        save_directory=OUTPUT / "outer_bundle",
    )
    outer_predictions.to_parquet(OUTPUT / "outer_2024_predictions.parquet", index=False)
    payload = {
        "architecture": "R specialist TabM-style shared-trunk member ensemble",
        "model_config": asdict(MODEL_CONFIG),
        "epochs": EPOCHS,
        "inner": inner,
        "outer": outer,
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={OUTPUT}", flush=True)


if __name__ == "__main__":
    from dataclasses import asdict

    main()
