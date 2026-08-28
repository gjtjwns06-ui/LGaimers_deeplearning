#!/usr/bin/env python3
"""Train the fixed post-2022 regime ensemble on all allowed training rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.recent_regime_final import train_final_recent_regime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/final_recent_regime")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    artifact = train_final_recent_regime(
        args.input,
        args.output_dir,
        seeds=args.seeds,
        device=args.device,
        cpu_threads=args.cpu_threads,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "directory": artifact.directory,
                "model_files": artifact.manifest["model_files"],
                "fit_rows": artifact.training_record["fit_rows"],
                "recent_polish_rows": artifact.training_record[
                    "recent_polish_rows"
                ],
                "obsolete_pre_2023_f_training_rows": artifact.manifest[
                    "obsolete_pre_2023_f_training_rows"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
