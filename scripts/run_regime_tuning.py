"""Run the fixed CPU-only C1/C2/C3 development regime ablations.

The input must be the anchor-free 2019--2023 development snapshot.  A CSV
containing any 2024 row is rejected before the remaining columns, baseline
artifacts, or output directory are touched.
"""

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

from lg_control.evaluation import EvaluationConfig
from lg_control.learning import OptimizationProcedure
from lg_control.regime_tuning import BASE_SEED, run_regime_tuning


DEFAULT_INPUT = PROJECT_ROOT / "data" / "development_2019_2023.csv"
DEFAULT_BASELINE_LEARNING = PROJECT_ROOT / "artifacts" / "learning"
DEFAULT_BASELINE_EVALUATION = PROJECT_ROOT / "artifacts" / "validation_evaluation"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "regime_tuning"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--baseline-learning-dir", type=Path, default=DEFAULT_BASELINE_LEARNING
    )
    parser.add_argument(
        "--baseline-evaluation-dir", type=Path, default=DEFAULT_BASELINE_EVALUATION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-selection-epochs", type=int, default=8)
    parser.add_argument("--scheduler-horizon-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Hash-verify and resume immutable candidate staging artifacts.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    optimization = OptimizationProcedure(
        max_selection_epochs=args.max_selection_epochs,
        scheduler_horizon_epochs=args.scheduler_horizon_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        brier_weight=args.brier_weight,
        patience=args.patience,
        min_delta=args.min_delta,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=0,
        use_amp=False,
        cpu_threads=args.cpu_threads,
    )
    result = run_regime_tuning(
        input_path=args.input,
        baseline_learning_dir=args.baseline_learning_dir,
        baseline_evaluation_dir=args.baseline_evaluation_dir,
        output_dir=args.output_dir,
        optimization=optimization,
        evaluation_config=EvaluationConfig(
            eval_batch_size=args.eval_batch_size,
            num_workers=0,
            cpu_threads=args.cpu_threads,
        ),
        base_seed=args.base_seed,
        resume=args.resume,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "plan_sha256": result["plan"]["plan_sha256"],
                "selection_sha256": result["selection"]["selection_sha256"],
                "selected_candidate_id": result["selection"]["decision"][
                    "selected_candidate_id"
                ],
                "anchor_2024_accessed": False,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
