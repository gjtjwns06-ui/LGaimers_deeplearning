#!/usr/bin/env python3
"""Evaluate a frozen bounded-residual bundle on one untouched forward season."""

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

from lg_control.hybrid_residual import (
    CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    HybridResidualPreprocessor,
    load_model_npz,
    predict_bounded_residual,
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
        "--bundle-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_residual_dev" / "future_bundle",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_residual_evaluation",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def _load_frame(meta_path: Path, train_path: Path, season: int) -> pd.DataFrame:
    meta = pd.read_parquet(meta_path)
    meta = meta.loc[meta["season"].eq(season)].copy()
    if meta.empty:
        raise ValueError(f"meta OOF has no rows for season {season}")
    raw_columns = list(
        dict.fromkeys(
            [
                "row_id",
                "season",
                "control_success",
                *RAW_NUMERIC_COLUMNS,
                *CATEGORICAL_COLUMNS,
            ]
        )
    )
    raw = pd.read_csv(train_path, usecols=raw_columns, low_memory=False)
    raw = raw.loc[raw["season"].eq(season)].copy()
    duplicate_columns = [column for column in CATEGORICAL_COLUMNS if column in meta.columns]
    raw = raw.drop(columns=["season", *duplicate_columns]).rename(
        columns={"control_success": "raw_y_true"}
    )
    frame = meta.merge(raw, on="row_id", how="left", validate="one_to_one")
    if len(frame) != len(meta) or frame["raw_y_true"].isna().any():
        raise RuntimeError("official raw features do not cover evaluation OOF")
    if not np.array_equal(
        frame["y_true"].to_numpy(dtype=np.int8),
        frame["raw_y_true"].to_numpy(dtype=np.int8),
    ):
        raise RuntimeError("official raw target differs from meta OOF")
    return frame.drop(columns="raw_y_true")


def _slice_metrics(
    frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray
) -> dict[str, object]:
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    result: dict[str, object] = {}
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "F": frame["game_type"].astype(str).str.upper().eq("F").to_numpy(),
        "R": frame["game_type"].astype(str).str.upper().eq("R").to_numpy(),
    }
    for name, mask in masks.items():
        base_metrics = metric_dict(y_true[mask], base[mask])
        candidate_metrics = metric_dict(y_true[mask], candidate[mask])
        result[name] = {
            "rows": int(mask.sum()),
            "base": base_metrics,
            "candidate": candidate_metrics,
            "delta_brier": float(candidate_metrics["brier"] - base_metrics["brier"]),
            "delta_score": float(
                candidate_metrics["competition_score"]
                - base_metrics["competition_score"]
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    meta_path = args.meta_oof.resolve()
    train_path = args.train_csv.resolve()
    bundle_dir = args.bundle_dir.resolve()
    output_dir = args.output_dir.resolve() / f"season_{args.season}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"evaluation output directory is not empty: {output_dir}")
    manifest_path = bundle_dir / "manifest.json"
    preprocessor_path = bundle_dir / "preprocessor.json"
    for path in (meta_path, train_path, manifest_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocessor = HybridResidualPreprocessor.from_state_dict(
        json.loads(preprocessor_path.read_text(encoding="utf-8"))
    )
    frame = _load_frame(meta_path, train_path, args.season)
    features = preprocessor.transform(frame)
    model_config = HybridResidualModelConfig(**manifest["model_config"])
    if model_config.input_dim != features.shape[1]:
        raise RuntimeError("bundle input dimension differs from evaluation features")
    base = frame["prediction_v7"].to_numpy(dtype=np.float32)
    seed_probabilities: list[np.ndarray] = []
    seed_corrections: list[np.ndarray] = []
    seed_metrics: dict[str, object] = {}
    for seed, model_file in manifest["model_files"].items():
        model = BoundedLogitResidualNet(model_config)
        load_model_npz(model, bundle_dir / model_file)
        probability, correction = predict_bounded_residual(
            model,
            features,
            base,
            batch_size=32768,
            device=args.device,
        )
        seed_probabilities.append(probability)
        seed_corrections.append(correction)
        seed_metrics[seed] = _slice_metrics(frame, base, probability)

    ensemble_probability = np.mean(seed_probabilities, axis=0)
    ensemble_correction = np.mean(seed_corrections, axis=0)
    slices = _slice_metrics(frame, base, ensemble_probability)
    cap = model_config.correction_cap
    correction_metrics = {
        "mean": float(np.mean(ensemble_correction)),
        "mean_abs": float(np.mean(np.abs(ensemble_correction))),
        "std": float(np.std(ensemble_correction)),
        "cap_hit_rate": float(np.mean(np.abs(ensemble_correction) >= cap * 0.95)),
    }
    if args.season == 2023:
        pass_gate = bool(
            slices["all"]["delta_brier"] <= 0.0005
            and slices["R"]["delta_brier"] <= 0.0001
            and correction_metrics["cap_hit_rate"] < 0.10
            and abs(correction_metrics["mean"]) < cap * 0.5
        )
        gate = {
            "purpose": "regime_shift_stress",
            "overall_delta_brier_max": 0.0005,
            "R_delta_brier_max": 0.0001,
            "cap_hit_rate_max": 0.10,
            "absolute_mean_correction_max_fraction_of_cap": 0.5,
        }
    elif args.season == 2024:
        all_seed_improve = all(
            metrics["all"]["delta_brier"] < 0.0 for metrics in seed_metrics.values()
        )
        pass_gate = bool(
            slices["all"]["delta_brier"] < -1e-6
            and all_seed_improve
            and correction_metrics["cap_hit_rate"] < 0.10
            and abs(correction_metrics["mean"]) < cap * 0.5
        )
        gate = {
            "purpose": "forward_confirmation",
            "ensemble_brier_improvement_min": 1e-6,
            "all_seeds_must_improve": True,
            "cap_hit_rate_max": 0.10,
            "absolute_mean_correction_max_fraction_of_cap": 0.5,
        }
    else:
        pass_gate = False
        gate = {"purpose": "diagnostic_only", "automatic_pass": False}

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = frame[["row_id", "season", "game_type", "y_true"]].copy()
    predictions["prediction_v7"] = base
    for (seed, _), probability, correction in zip(
        manifest["model_files"].items(), seed_probabilities, seed_corrections
    ):
        predictions[f"prediction_residual_seed_{seed}"] = probability
        predictions[f"correction_seed_{seed}"] = correction
    predictions["prediction_residual_ensemble"] = ensemble_probability
    predictions["correction_ensemble"] = ensemble_correction
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    payload = {
        "season": args.season,
        "bundle": str(bundle_dir),
        "bundle_training_seasons": manifest["training_seasons"],
        "correction_cap": cap,
        "rows": len(frame),
        "slices": slices,
        "seed_metrics": seed_metrics,
        "correction_metrics": correction_metrics,
        "gate": gate,
        "decision": "PASS" if pass_gate else "FAIL",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"season={args.season} delta_brier={slices['all']['delta_brier']:+.10f} "
        f"delta_score={slices['all']['delta_score']:+.2f} "
        f"decision={'PASS' if pass_gate else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
