#!/usr/bin/env python3
"""Select an equal-weight seed ensemble using aligned OOF predictions."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_control.metrics import metric_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Repeat for every aligned seed prediction file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/ensemble_selection")
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    sources: dict[str, pd.DataFrame] = {}
    for item in args.prediction:
        if "=" not in item:
            raise ValueError(f"prediction must use NAME=CSV syntax: {item}")
        name, raw_path = item.split("=", 1)
        if not name or name in sources:
            raise ValueError(f"empty or duplicate model name: {name!r}")
        frame = pd.read_csv(Path(raw_path))
        required = {"row_id", "season", "y_true", "prediction"}
        missing = sorted(required.difference(frame.columns))
        if missing or frame.empty:
            raise ValueError(f"invalid prediction file {raw_path}; missing={missing}")
        sources[name] = frame

    names = list(sources)
    reference = sources[names[0]]
    for name, frame in sources.items():
        if not frame["row_id"].equals(reference["row_id"]):
            raise ValueError(f"row_id order differs for {name}")
        if not frame["y_true"].equals(reference["y_true"]):
            raise ValueError(f"targets differ for {name}")
    y = reference["y_true"].to_numpy(dtype=float)

    candidates: list[dict[str, object]] = []
    predictions: dict[tuple[str, ...], np.ndarray] = {}
    for size in range(1, len(names) + 1):
        for members in itertools.combinations(names, size):
            probability = np.mean(
                [sources[name]["prediction"].to_numpy(dtype=float) for name in members],
                axis=0,
            )
            predictions[members] = probability
            candidates.append({"members": list(members), **metric_dict(y, probability)})
    candidates.sort(key=lambda item: float(item["brier"]))
    selected = tuple(candidates[0]["members"])
    selected_probability = predictions[selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_frame = reference[["row_id", "season", "y_true"]].copy()
    selected_frame["prediction"] = selected_probability
    selected_frame.to_csv(args.output_dir / "selected_oof.csv", index=False)
    payload: dict[str, object] = {
        "selected": list(selected),
        "selection_metric": "brier",
        "candidates": candidates,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"selected={'+'.join(selected)} brier={candidates[0]['brier']:.8f} "
        f"candidates={len(candidates)}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run(parse_args())

