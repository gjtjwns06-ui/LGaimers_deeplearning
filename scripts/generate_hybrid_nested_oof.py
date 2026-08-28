#!/usr/bin/env python3
"""Generate leakage-safe nested OOF predictions for the hybrid residual stage.

For an outer season ``t``, the epoch is selected on ``t-1`` using only seasons
before ``t-1``.  The model is then rebuilt on all seasons before ``t`` for the
preselected number of epochs.  The outer target is scored but never selects a
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_regime_cross.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "hybrid_nested_oof"
MODEL_NAMES = ("rate_exclusive", "temporal_te")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_set: str
    mode_flag: str


MODEL_SPECS = {
    "rate_exclusive": ModelSpec(
        name="rate_exclusive",
        feature_set="rate_basis",
        mode_flag="--exclusive-outcome",
    ),
    "temporal_te": ModelSpec(
        name="temporal_te",
        feature_set="temporal_te",
        mode_flag="--multitask",
    ),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024],
    )
    parser.add_argument(
        "--models",
        choices=MODEL_NAMES,
        nargs="+",
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=positive_int, default=8)
    parser.add_argument("--max-train-rows", type=positive_int, default=None)
    parser.add_argument("--max-val-rows", type=positive_int, default=None)
    return parser.parse_args()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _base_command(args: argparse.Namespace, spec: ModelSpec) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--input",
        str(args.input.resolve()),
        "--feature-set",
        spec.feature_set,
        spec.mode_flag,
        "--epochs",
        "4",
        "--patience",
        "2",
        "--batch-size",
        "8192",
        "--eval-batch-size",
        "32768",
        "--learning-rate",
        "0.0008",
        "--weight-decay",
        "0.00002",
        "--brier-weight",
        "1.0",
        "--seed",
        "42",
        "--cpu-threads",
        str(args.cpu_threads),
        "--embedding-dim-cap",
        "16",
        "--interaction-dim",
        "64",
        "--cross-layers",
        "3",
        "--num-experts",
        "4",
        "--low-rank",
        "16",
        "--deep-hidden-dim",
        "96",
        "--deep-layers",
        "2",
        "--dropout",
        "0.1",
        "--r-decay",
        "0.6",
        "--f-decay",
        "0.08",
        "--device",
        args.device,
    ]
    if spec.name == "rate_exclusive":
        command.extend(["--ball-strike-weight", "0.1"])
    else:
        command.extend(
            ["--auxiliary-weight", "0.15", "--consistency-weight", "0.02"]
        )
    if args.max_train_rows is not None:
        command.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.max_val_rows is not None:
        command.extend(["--max-val-rows", str(args.max_val_rows)])
    return command


def _run_or_reuse(command: list[str], output_dir: Path) -> dict[str, object]:
    invocation_path = output_dir / "invocation.json"
    metrics_path = output_dir / "metrics.json"
    prediction_path = output_dir / "predictions.csv"
    invocation = {
        "command": command,
        "command_sha256": _json_hash(command),
    }
    if invocation_path.is_file() or metrics_path.is_file() or prediction_path.is_file():
        if not (invocation_path.is_file() and metrics_path.is_file() and prediction_path.is_file()):
            raise RuntimeError(f"incomplete existing run cannot be reused: {output_dir}")
        observed = json.loads(invocation_path.read_text(encoding="utf-8"))
        if observed != invocation:
            raise RuntimeError(f"existing run has a different invocation: {output_dir}")
        print(f"reuse={output_dir}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    print("run=" + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not (metrics_path.is_file() and prediction_path.is_file()):
        raise RuntimeError(f"training run did not produce required artifacts: {output_dir}")
    _write_json(invocation_path, invocation)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _replace_option(command: list[str], name: str, value: str) -> list[str]:
    result = list(command)
    position = result.index(name)
    result[position + 1] = value
    return result


def _run_fold(
    args: argparse.Namespace,
    spec: ModelSpec,
    outer_year: int,
) -> dict[str, object]:
    inner_year = outer_year - 1
    fold_root = args.output_dir.resolve() / spec.name / f"fold_{outer_year}"
    base = _base_command(args, spec)

    inner_dir = fold_root / f"inner_select_{inner_year}"
    inner_command = base + [
        "--validation-year",
        str(inner_year),
        "--output-dir",
        str(inner_dir),
        "--checkpoint-mode",
        "best",
    ]
    inner_metrics = _run_or_reuse(inner_command, inner_dir)
    selected_epoch = int(inner_metrics["best_epoch"])
    if selected_epoch < 1 or selected_epoch > 4:
        raise RuntimeError(f"invalid selected epoch for {spec.name}/{outer_year}")

    outer_dir = fold_root / "outer_fixed"
    outer_command = _replace_option(base, "--epochs", str(selected_epoch))
    outer_command.extend(
        [
            "--scheduler-epochs",
            "4",
            "--validation-year",
            str(outer_year),
            "--output-dir",
            str(outer_dir),
            "--checkpoint-mode",
            "last",
            "--save-head-probabilities",
        ]
    )
    outer_metrics = _run_or_reuse(outer_command, outer_dir)
    if outer_metrics.get("checkpoint_mode") != "last":
        raise RuntimeError("outer fold did not use its fixed final checkpoint")
    if int(outer_metrics["best_epoch"]) != selected_epoch:
        raise RuntimeError("outer fold epoch differs from the inner selection")

    fold_manifest = {
        "model": spec.name,
        "outer_year": outer_year,
        "inner_selection_year": inner_year,
        "selected_epoch": selected_epoch,
        "scheduler_epochs": 4,
        "outer_target_used_for_checkpoint_selection": False,
        "inner_metrics": inner_metrics["metrics"],
        "outer_metrics": outer_metrics["metrics"],
    }
    _write_json(fold_root / "fold_manifest.json", fold_manifest)
    return fold_manifest


def _concatenate_model_predictions(
    output_dir: Path,
    model_name: str,
    years: Iterable[int],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for year in years:
        path = output_dir / model_name / f"fold_{year}" / "outer_fixed" / "predictions.csv"
        frame = pd.read_csv(path)
        if set(frame["season"].astype(int)) != {int(year)}:
            raise RuntimeError(f"prediction season mismatch: {path}")
        if frame["row_id"].duplicated().any():
            raise RuntimeError(f"duplicate row_id in {path}")
        rename = {"prediction": f"prediction_{model_name}"}
        rename.update(
            {
                column: f"{model_name}_{column}"
                for column in frame.columns
                if column.startswith("head_")
            }
        )
        pieces.append(frame.rename(columns=rename))
    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(output_dir / model_name / "nested_oof.csv", index=False)
    return result


def _merge_models(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(frames)
    merged = frames[names[0]].copy()
    for name in names[1:]:
        other = frames[name]
        duplicate_contract = ["season", "game_type", "y_true"]
        merged = merged.merge(
            other.drop(columns=duplicate_contract),
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
        reference = frames[names[0]].set_index("row_id")
        observed = other.set_index("row_id")
        if not reference.index.equals(observed.index):
            observed = observed.reindex(reference.index)
        for column in duplicate_contract:
            if not reference[column].equals(observed[column]):
                raise RuntimeError(f"model OOF contract differs for {name}/{column}")
    if len(names) == 2:
        merged["prediction_equal"] = 0.5 * (
            merged[f"prediction_{names[0]}"] + merged[f"prediction_{names[1]}"]
        )
    return merged


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    years = sorted(set(args.years))
    if years[0] < 2021 or years[-1] > 2024:
        raise ValueError("nested OOF years must stay inside 2021..2024")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    manifests = []
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        for year in years:
            manifests.append(_run_fold(args, spec, year))

    model_frames = {
        name: _concatenate_model_predictions(args.output_dir, name, years)
        for name in args.models
    }
    merged = _merge_models(model_frames)
    merged.to_csv(args.output_dir / "nested_oof.csv", index=False)
    payload = {
        "input": str(args.input),
        "models": args.models,
        "years": years,
        "rows": len(merged),
        "row_id_unique": bool(not merged["row_id"].duplicated().any()),
        "outer_target_used_for_checkpoint_selection": False,
        "folds": manifests,
    }
    _write_json(args.output_dir / "manifest.json", payload)
    print(f"nested_oof={args.output_dir / 'nested_oof.csv'} rows={len(merged):,}")


if __name__ == "__main__":
    main()
