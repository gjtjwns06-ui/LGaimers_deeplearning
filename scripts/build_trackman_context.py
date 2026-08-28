#!/usr/bin/env python3
"""Build leakage-safe prior-season Trackman context lookup CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_control.trackman import HISTORY_REQUIRED_COLUMNS, build_trackman_lookup


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/trackman_history.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/trackman_context.csv"))
    parser.add_argument(
        "--target-seasons",
        nargs="+",
        type=int,
        default=list(range(2019, 2026)),
        help="Target seasons; each lookup uses only source seasons strictly before it.",
    )
    parser.add_argument("--min-group-count", type=positive_int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header = pd.read_csv(args.input, nrows=0)
    missing = sorted(set(HISTORY_REQUIRED_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    history = pd.read_csv(args.input, usecols=list(HISTORY_REQUIRED_COLUMNS))
    lookup = build_trackman_lookup(
        history,
        target_seasons=args.target_seasons,
        min_group_count=args.min_group_count,
    )
    report = dict(lookup.attrs.get("build_report", {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(args.output, index=False, float_format="%.10g")
    report["output"] = str(args.output)
    report["output_columns"] = int(len(lookup.columns))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
