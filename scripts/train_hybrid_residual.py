#!/usr/bin/env python3
"""Develop and freeze a bounded v7-logit residual network on 2021->2022 OOF."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.hybrid_residual import (
    CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    HybridResidualPreprocessor,
    HybridResidualTrainConfig,
    predict_bounded_residual,
    save_model_npz,
    train_bounded_residual,
    train_bounded_residual_fixed_epochs,
)
from lg_control.metrics import metric_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-oof",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_meta_oof" / "hybrid_meta_oof.parquet",
    )
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_residual_dev",
    )
    parser.add_argument("--caps", type=float, nargs="+", default=[0.03, 0.05, 0.08])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_joined(meta_path: Path, train_path: Path) -> pd.DataFrame:
    meta = pd.read_parquet(meta_path)
    seasons = set(pd.to_numeric(meta["season"], errors="raise").astype(int))
    if not {2021, 2022}.issubset(seasons):
        raise ValueError("residual development requires nested 2021 and 2022 OOF")
    raw_columns = [
        "row_id",
        "season",
        "control_success",
        *RAW_NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
    ]
    raw_columns = list(dict.fromkeys(raw_columns))
    raw = pd.read_csv(train_path, usecols=raw_columns, low_memory=False)
    raw = raw.loc[raw["season"].isin(seasons)].copy()
    duplicates = [column for column in CATEGORICAL_COLUMNS if column in meta.columns]
    raw = raw.drop(columns=["season", *duplicates])
    raw = raw.rename(columns={"control_success": "raw_y_true"})
    joined = meta.merge(raw, on="row_id", how="left", validate="one_to_one")
    if len(joined) != len(meta) or joined["raw_y_true"].isna().any():
        raise RuntimeError("raw official features do not cover the meta OOF")
    if not np.array_equal(
        joined["y_true"].to_numpy(dtype=np.int8),
        joined["raw_y_true"].to_numpy(dtype=np.int8),
    ):
        raise RuntimeError("raw and meta targets differ")
    return joined.drop(columns="raw_y_true")


def _candidate_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    corrections: np.ndarray,
    *,
    cap: float,
) -> dict[str, object]:
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    result: dict[str, object] = {
        **metric_dict(y_true, probabilities),
        "mean_correction": float(np.mean(corrections)),
        "mean_abs_correction": float(np.mean(np.abs(corrections))),
        "cap_hit_rate": float(np.mean(np.abs(corrections) >= cap * 0.95)),
    }
    slices: dict[str, object] = {}
    for game_type in ("F", "R"):
        mask = frame["game_type"].astype(str).str.upper().eq(game_type).to_numpy()
        slices[game_type] = {
            "rows": int(mask.sum()),
            **metric_dict(y_true[mask], probabilities[mask]),
            "mean_correction": float(np.mean(corrections[mask])),
        }
    result["slices"] = slices
    return result


def _base_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return metric_dict(
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["prediction_v7"].to_numpy(dtype=np.float64),
    )


def _new_model(input_dim: int, cap: float, seed: int) -> BoundedLogitResidualNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return BoundedLogitResidualNet(
            HybridResidualModelConfig(
                input_dim=input_dim,
                correction_cap=cap,
                hidden_dim=32,
                hidden_layers=2,
                dropout=0.10,
            )
        )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    meta_path = args.meta_oof.resolve()
    train_path = args.train_csv.resolve()
    output_dir = args.output_dir.resolve()
    for path in (meta_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"residual output directory is not empty: {output_dir}")
    if any(not 0.0 < cap <= 0.5 for cap in args.caps):
        raise ValueError("all correction caps must be in (0, 0.5]")

    frame = _load_joined(meta_path, train_path)
    train_frame = frame.loc[frame["season"].eq(2021)].copy()
    validation_frame = frame.loc[frame["season"].eq(2022)].copy()
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError("2021->2022 residual split is empty")
    preprocessor = HybridResidualPreprocessor().fit(train_frame)
    train_features = preprocessor.transform(train_frame)
    validation_features = preprocessor.transform(validation_frame)
    train_y = train_frame["y_true"].to_numpy(dtype=np.float32)
    train_base = train_frame["prediction_v7"].to_numpy(dtype=np.float32)
    validation_y = validation_frame["y_true"].to_numpy(dtype=np.float32)
    validation_base = validation_frame["prediction_v7"].to_numpy(dtype=np.float32)
    base_metrics = _base_metrics(validation_frame)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "development_preprocessor.json", preprocessor.to_state_dict())

    candidate_payloads: list[dict[str, object]] = []
    candidate_predictions = validation_frame[["row_id", "season", "game_type", "y_true"]].copy()
    selected_models: dict[tuple[float, int], BoundedLogitResidualNet] = {}
    selected_epochs: dict[tuple[float, int], int] = {}
    for cap in args.caps:
        seed_probabilities: list[np.ndarray] = []
        seed_corrections: list[np.ndarray] = []
        seed_payloads: list[dict[str, object]] = []
        for seed in args.seeds:
            print(f"candidate cap={cap:.3f} seed={seed}", flush=True)
            model = _new_model(train_features.shape[1], cap, seed)
            config = HybridResidualTrainConfig(
                epochs=args.epochs,
                scheduler_epochs=args.epochs,
                patience=args.patience,
                seed=seed,
                verbose=True,
            )
            result = train_bounded_residual(
                model,
                train_features,
                train_base,
                train_y,
                validation_features,
                validation_base,
                validation_y,
                config=config,
                device=args.device,
            )
            seed_probabilities.append(result.probabilities)
            seed_corrections.append(result.corrections)
            selected_models[(cap, seed)] = model
            selected_epochs[(cap, seed)] = result.best_epoch
            candidate_predictions[f"cap_{cap:.3f}_seed_{seed}"] = result.probabilities
            seed_metrics = _candidate_metrics(
                validation_frame,
                result.probabilities,
                result.corrections,
                cap=cap,
            )
            seed_payloads.append(
                {
                    "seed": seed,
                    "selected_epoch": result.best_epoch,
                    "metrics": seed_metrics,
                    "delta_brier_vs_v7": float(seed_metrics["brier"] - base_metrics["brier"]),
                    "history": result.history,
                    "train_config": config.to_dict(),
                }
            )
            seed_dir = output_dir / "candidates" / f"cap_{cap:.3f}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            save_model_npz(model, seed_dir / "model_state.npz")
            _write_json(seed_dir / "metrics.json", seed_payloads[-1])

        ensemble_probability = np.mean(seed_probabilities, axis=0)
        ensemble_correction = np.mean(seed_corrections, axis=0)
        ensemble_metrics = _candidate_metrics(
            validation_frame,
            ensemble_probability,
            ensemble_correction,
            cap=cap,
        )
        candidate_predictions[f"cap_{cap:.3f}_ensemble"] = ensemble_probability
        candidate_payloads.append(
            {
                "cap": cap,
                "ensemble_metrics": ensemble_metrics,
                "delta_brier_vs_v7": float(ensemble_metrics["brier"] - base_metrics["brier"]),
                "all_seeds_improve_v7": all(
                    item["delta_brier_vs_v7"] < 0.0 for item in seed_payloads
                ),
                "seeds": seed_payloads,
            }
        )

    candidate_predictions.to_parquet(output_dir / "development_predictions.parquet", index=False)
    selected = min(candidate_payloads, key=lambda item: item["ensemble_metrics"]["brier"])
    selected_cap = float(selected["cap"])
    selected_metrics = selected["ensemble_metrics"]
    pass_gate = bool(
        selected["delta_brier_vs_v7"] < -1e-6
        and selected["all_seeds_improve_v7"]
        and selected_metrics["cap_hit_rate"] < 0.10
        and abs(selected_metrics["mean_correction"]) < selected_cap * 0.5
    )
    selection = {
        "development_train_season": 2021,
        "development_validation_season": 2022,
        "base": "public_verified_open1_v7",
        "base_metrics": base_metrics,
        "selected_cap": selected_cap,
        "selected_seed_epochs": {
            str(seed): selected_epochs[(selected_cap, seed)] for seed in args.seeds
        },
        "selected_metrics": selected_metrics,
        "selected_delta_brier_vs_v7": selected["delta_brier_vs_v7"],
        "decision": "PASS" if pass_gate else "FAIL",
        "gate": {
            "ensemble_brier_improvement_min": 1e-6,
            "all_seeds_must_improve": True,
            "cap_hit_rate_max": 0.10,
            "absolute_mean_correction_max_fraction_of_cap": 0.5,
        },
        "candidates": candidate_payloads,
        "feature_count": train_features.shape[1],
        "feature_names": preprocessor.feature_names_,
    }
    _write_json(output_dir / "selection.json", selection)

    if pass_gate:
        print("development_gate=PASS; fitting frozen 2021+2022 future bundle", flush=True)
        future_preprocessor = HybridResidualPreprocessor().fit(frame)
        future_features = future_preprocessor.transform(frame)
        future_y = frame["y_true"].to_numpy(dtype=np.float32)
        future_base = frame["prediction_v7"].to_numpy(dtype=np.float32)
        bundle_dir = output_dir / "future_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        _write_json(bundle_dir / "preprocessor.json", future_preprocessor.to_state_dict())
        model_files: dict[str, str] = {}
        histories: dict[str, object] = {}
        for seed in args.seeds:
            selected_epoch = selected_epochs[(selected_cap, seed)]
            model = _new_model(future_features.shape[1], selected_cap, seed)
            config = HybridResidualTrainConfig(
                epochs=selected_epoch,
                scheduler_epochs=args.epochs,
                patience=args.patience,
                seed=seed,
                verbose=True,
            )
            histories[str(seed)] = train_bounded_residual_fixed_epochs(
                model,
                future_features,
                future_base,
                future_y,
                config=config,
                device=args.device,
            )
            model_file = f"model_seed_{seed}.npz"
            save_model_npz(model, bundle_dir / model_file)
            model_files[str(seed)] = model_file
        _write_json(
            bundle_dir / "manifest.json",
            {
                "training_seasons": [2021, 2022],
                "correction_cap": selected_cap,
                "model_config": HybridResidualModelConfig(
                    input_dim=future_features.shape[1],
                    correction_cap=selected_cap,
                    hidden_dim=32,
                    hidden_layers=2,
                    dropout=0.10,
                ).to_dict(),
                "selected_epochs": selection["selected_seed_epochs"],
                "scheduler_epochs": args.epochs,
                "model_files": model_files,
                "histories": histories,
                "selection_file": "../selection.json",
            },
        )
    else:
        print("development_gate=FAIL; no future residual bundle was fitted", flush=True)

    report = [
        "# Hybrid residual development",
        "",
        f"- base v7 Brier: `{base_metrics['brier']:.10f}`",
        f"- selected cap: `{selected_cap:.3f}`",
        f"- selected ensemble Brier: `{selected_metrics['brier']:.10f}`",
        f"- delta Brier: `{selected['delta_brier_vs_v7']:+.10f}`",
        f"- decision: **{'PASS' if pass_gate else 'FAIL'}**",
        "",
        "2022 is a development season. 2023 and 2024 were not read by this run.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"selected_cap={selected_cap:.3f} "
        f"delta_brier={selected['delta_brier_vs_v7']:+.10f} "
        f"decision={'PASS' if pass_gate else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
