#!/usr/bin/env python3
"""Fit a two-parameter Brier calibrator using forward-fold OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_control.calibration import AffineLogitCalibrator
from lg_control.metrics import metric_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("artifacts/deep_cv/oof_predictions.csv")],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/deep_cv/calibrator.json")
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Fit production parameters only on OOF rows at or after this season.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    required = {"season", "y_true", "prediction"}
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"invalid OOF file; missing={missing}, rows={len(frame)}")
    raw = frame["prediction"].to_numpy(dtype=float)
    target = frame["y_true"].to_numpy(dtype=float)
    fit_mask = (
        np.ones(len(frame), dtype=bool)
        if args.min_season is None
        else frame["season"].to_numpy() >= args.min_season
    )
    if not fit_mask.any():
        raise ValueError("min-season removed every OOF row")
    calibrator = AffineLogitCalibrator().fit(raw[fit_mask], target[fit_mask])
    calibrated = calibrator.transform(raw)

    per_season: dict[str, object] = {}
    for season in sorted(int(value) for value in frame["season"].unique()):
        mask = frame["season"].to_numpy() == season
        per_season[str(season)] = {
            "rows": int(mask.sum()),
            "raw": metric_dict(target[mask], raw[mask]),
            "calibrated_in_sample": metric_dict(target[mask], calibrated[mask]),
        }

    forward_audit: dict[str, object] = {}
    seasons = sorted(int(value) for value in frame["season"].unique())
    for season in seasons[1:]:
        fit_mask = frame["season"].to_numpy() < season
        eval_mask = frame["season"].to_numpy() == season
        forward = AffineLogitCalibrator().fit(raw[fit_mask], target[fit_mask])
        forward_prediction = forward.transform(raw[eval_mask])
        forward_audit[str(season)] = {
            "fit_seasons": [value for value in seasons if value < season],
            "scale": forward.scale,
            "bias": forward.bias,
            "raw": metric_dict(target[eval_mask], raw[eval_mask]),
            "calibrated": metric_dict(target[eval_mask], forward_prediction),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.output)
    payload: dict[str, object] = {
        "rows": len(frame),
        "fit_rows": int(fit_mask.sum()),
        "min_season": args.min_season,
        "scale": calibrator.scale,
        "bias": calibrator.bias,
        "raw": metric_dict(target, raw),
        "calibrated_in_sample": metric_dict(target, calibrated),
        "per_season": per_season,
        "forward_audit": forward_audit,
    }
    metrics_path = args.output.with_name(args.output.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"saved={args.output} scale={calibrator.scale:.6f} bias={calibrator.bias:.6f} "
        f"raw_brier={payload['raw']['brier']:.8f} "
        f"calibrated_brier={payload['calibrated_in_sample']['brier']:.8f}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run(parse_args())
