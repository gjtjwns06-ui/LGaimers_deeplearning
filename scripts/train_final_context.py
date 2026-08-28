#!/usr/bin/env python3
"""Train or strictly verify the confirmed single C2 final artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.final_context_training import train_final_context


DEFAULT_TRAIN_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_ANCHOR_DIRECTORY = PROJECT_ROOT / "artifacts" / "anchor_confirmation"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "final_context_c2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument(
        "--anchor-directory", type=Path, default=DEFAULT_ANCHOR_DIRECTORY
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and return an existing immutable final artifact; never retrain.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    loaded = train_final_context(
        train_input=args.train_input,
        anchor_directory=args.anchor_directory,
        output_directory=args.output_directory,
        resume=args.resume,
        verbose=not args.quiet,
    )
    summary = {
        "status": "ok",
        "output_directory": loaded.directory,
        "artifact_manifest_sha256": loaded.manifest["artifact_manifest_sha256"],
        "selected_candidate_id": loaded.manifest["selected_candidate_id"],
        "selected_epoch": loaded.manifest["selected_epoch"],
        "model_count": loaded.manifest["model_count"],
        "source_input_sha256": loaded.manifest["source_input_sha256"],
        "anchor_confirmation_sha256": loaded.manifest[
            "anchor_confirmation_sha256"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
