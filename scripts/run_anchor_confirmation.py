"""Run one explicit phase of the one-shot 2024 anchor confirmation."""

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

from lg_control.anchor_confirmation import run_anchor_confirmation


DEFAULT_DEVELOPMENT_INPUT = PROJECT_ROOT / "data" / "development_2019_2023.csv"
DEFAULT_REGIME_DIRECTORY = PROJECT_ROOT / "artifacts" / "regime_tuning"
DEFAULT_FULL_TRAIN_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "anchor_confirmation"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-input", type=Path, default=DEFAULT_DEVELOPMENT_INPUT
    )
    parser.add_argument(
        "--regime-directory", type=Path, default=DEFAULT_REGIME_DIRECTORY
    )
    parser.add_argument("--full-train-input", type=Path, default=DEFAULT_FULL_TRAIN_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop at the hash-sealed anchor_ready pre-access checkpoint.",
    )
    phase.add_argument(
        "--confirm-only",
        action="store_true",
        help="Require an existing anchor_ready seal and claim one-shot access.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Strictly verify and resume pre-access preparation artifacts.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and not args.prepare_only:
        parser.error("--resume is permitted only with --prepare-only")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    phase = "prepare" if args.prepare_only else "confirm"
    result = run_anchor_confirmation(
        development_input=args.development_input,
        regime_directory=args.regime_directory,
        full_train_input=args.full_train_input,
        output_directory=args.output_directory,
        phase=phase,
        resume=args.resume,
        verbose=not args.quiet,
    )
    summary: dict[str, object] = {
        "status": "ok",
        "phase": result["phase"],
        "output_directory": str(args.output_directory.resolve()),
    }
    if "anchor_ready" in result:
        summary["anchor_ready_sha256"] = result["anchor_ready"][
            "anchor_ready_sha256"
        ]
        summary["anchor_2024_accessed_before_confirmation"] = False
    if "confirmation" in result:
        summary["confirmation_sha256"] = result["confirmation"][
            "confirmation_sha256"
        ]
        summary["confirmation_status"] = result["confirmation"]["status"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
