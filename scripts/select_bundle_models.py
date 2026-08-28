#!/usr/bin/env python3
"""Create a clean deployable bundle containing selected ensemble members."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_control.artifacts import load_model_bundle, save_model_bundle
from lg_control.calibration import AffineLogitCalibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--model-indices", nargs="+", type=int, required=True)
    parser.add_argument("--calibrator", type=Path, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    preprocessor, models, old_calibrator, metadata = load_model_bundle(args.input_bundle)
    indices = list(dict.fromkeys(args.model_indices))
    if not indices or min(indices) < 0 or max(indices) >= len(models):
        raise ValueError(f"model indices out of range for {len(models)} models: {indices}")
    selected = [models[index] for index in indices]
    calibrator = (
        AffineLogitCalibrator.load(args.calibrator)
        if args.calibrator is not None
        else old_calibrator
    )
    new_metadata = dict(metadata)
    source_seeds = list(metadata.get("seeds", []))
    new_metadata["selected_model_indices"] = indices
    if len(source_seeds) == len(models):
        new_metadata["selected_seeds"] = [source_seeds[index] for index in indices]
    save_model_bundle(
        args.output_bundle,
        preprocessor=preprocessor,
        models=selected,
        calibrator=calibrator,
        metadata=new_metadata,
    )
    print(
        f"saved={args.output_bundle} models={len(selected)} indices={indices}",
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())

