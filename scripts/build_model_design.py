"""Build and audit the neural model architecture without training it.

The script fits only the already-approved target-independent preprocessor on
the latest development fold's training rows.  It initializes the architecture
and runs shape/row-independence smoke checks, but creates no optimizer, fitted
weights, validation metric, prediction file, calibration, or submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import ID_COL, TARGET_COL, TRAIN_COLUMNS
from lg_control.model import (
    ContextGatedModelConfig,
    build_context_gated_model,
    validate_transformed_arrays,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.validation import build_validation_plan, fit_preprocessor_on_fold_train


DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "model_design"
INITIALIZATION_SEED = 2026
PARAMETER_BUDGET = 1_000_000


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def _candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": "tabular_residual_baseline",
                "status": "retain_as_baseline",
                "strength": "simple row-local residual MLP with numeric wide path",
                "risk": "mixes entity, history, and context before fallback is established",
            },
            {
                "candidate": "context_gated_residual_v1",
                "status": "implemented_primary",
                "strength": "name-resolved three towers, OOV fallback, gated ID corrections",
                "risk": "extra structure must be justified only after future training/validation",
            },
            {
                "candidate": "low_rank_dcn_v2",
                "status": "defer_to_tuning_stage",
                "strength": "low-cost explicit feature crosses",
                "risk": "additional design choice before the primary is measured",
            },
            {
                "candidate": "feature_token_transformer",
                "status": "not_selected_for_v1",
                "strength": "flexible within-row feature interaction",
                "risk": "higher activation cost and tuning/overconfidence risk for Brier score",
            },
        ]
    )


def _parameter_table(report: dict[str, object]) -> pd.DataFrame:
    groups = report["by_top_level_module"]
    assert isinstance(groups, dict)
    total = int(report["total_parameters"])
    rows = [
        {
            "component": str(component),
            "parameters": int(count),
            "share": int(count) / total,
        }
        for component, count in groups.items()
    ]
    rows.append({"component": "TOTAL", "parameters": total, "share": 1.0})
    return pd.DataFrame(rows)


def build_outputs(input_path: Path, output_dir: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, low_memory=False)
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("train.csv column order does not match the official training schema")

    plan = build_validation_plan(frame, include_recent_only=True)
    development_folds = plan.development_folds
    if [fold.validation_season for fold in development_folds] != [2022, 2023]:
        raise AssertionError("model-development fold contract changed unexpectedly")
    reference_fold = development_folds[-1]
    if reference_fold.validation_season >= plan.anchor_fold.validation_season:
        raise AssertionError("model design reference must precede the anchor")

    fitted = fit_preprocessor_on_fold_train(
        lambda: DeepTabularPreprocessor(min_category_frequency=1),
        frame,
        reference_fold,
    )
    fitted.audit.assert_train_only()
    preprocessor = fitted.preprocessor
    if not isinstance(preprocessor, DeepTabularPreprocessor):
        raise TypeError("unexpected preprocessor type returned by fit helper")

    config = ContextGatedModelConfig()
    model = build_context_gated_model(
        preprocessor,
        config=config,
        initialization_seed=INITIALIZATION_SEED,
    ).eval()
    parameter_report = model.parameter_report()
    if int(parameter_report["total_parameters"]) >= PARAMETER_BUDGET:
        raise AssertionError("primary model exceeds the fixed V1 parameter budget")
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise AssertionError("row-independent architecture must not contain BatchNorm")

    validation_inputs = frame.iloc[reference_fold.validation_positions].drop(
        columns=[TARGET_COL]
    )
    transformed = preprocessor.transform(validation_inputs)
    array_audit = validate_transformed_arrays(
        model.feature_layout,
        transformed,
        expected_row_ids=validation_inputs[ID_COL].to_numpy(),
    )

    categorical = transformed["categorical"]
    numerical = transformed["numerical"]
    layout_records = model.feature_layout_records()
    for record in layout_records:
        if record["kind"] != "categorical":
            record["missing_rate_on_reference_validation"] = None
            record["oov_rate_on_reference_validation"] = None
            continue
        position = int(record["position"])
        record["missing_rate_on_reference_validation"] = float(
            np.mean(categorical[:, position] == 0)
        )
        record["oov_rate_on_reference_validation"] = float(
            np.mean(categorical[:, position] == 1)
        )
    feature_layout = pd.DataFrame(layout_records)

    smoke_rows = min(16, len(validation_inputs))
    smoke_cat = torch.from_numpy(categorical[:smoke_rows])
    smoke_num = torch.from_numpy(numerical[:smoke_rows])
    with torch.inference_mode():
        batched = model(smoke_cat, smoke_num)
        singles = torch.cat(
            [
                model(smoke_cat[index : index + 1], smoke_num[index : index + 1])
                for index in range(smoke_rows)
            ]
        )
        reversed_logits = model(
            torch.flip(smoke_cat, dims=(0,)),
            torch.flip(smoke_num, dims=(0,)),
        ).flip(0)
    single_max_difference = float(torch.max(torch.abs(batched - singles)).item())
    permutation_max_difference = float(
        torch.max(torch.abs(batched - reversed_logits)).item()
    )
    if not torch.isfinite(batched).all().item():
        raise AssertionError("initialized architecture produced non-finite smoke logits")
    if single_max_difference > 1e-6 or permutation_max_difference > 1e-6:
        raise AssertionError("row-independence smoke check failed")

    season_position = model.feature_layout.categorical_index("season")
    reference_season_oov_rate = float(np.mean(categorical[:, season_position] == 1))
    if reference_season_oov_rate != 1.0:
        raise AssertionError("next-season categorical token should be entirely OOV")

    candidate_table = _candidate_table()
    parameter_table = _parameter_table(parameter_report)
    _write_csv(candidate_table, output_dir / "candidate_comparison.csv")
    _write_csv(feature_layout, output_dir / "feature_layout.csv")
    _write_csv(parameter_table, output_dir / "parameter_summary.csv")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "scope": "model design and initialized-forward audit only; no training or performance validation",
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": _sha256_file(input_path),
            "rows": int(len(frame)),
        },
        "selection_boundary": {
            "development_folds": [fold.name for fold in development_folds],
            "reference_fold": reference_fold.name,
            "reference_train_seasons": list(reference_fold.train_seasons),
            "reference_validation_season": reference_fold.validation_season,
            "anchor_fold": plan.anchor_fold.name,
            "anchor_predictions_evaluated_in_this_run": False,
            "anchor_labels_previously_observed_in_project": True,
            "anchor_rows_used_for_architecture_config_in_this_run": False,
            "recent_only_diagnostic_accessed_for_model_design": False,
        },
        "canonical_preprocessor": {
            "class": "DeepTabularPreprocessor",
            "target_independent_fit": True,
            "fit_rows": reference_fold.train_row_count,
            "fit_seasons": list(reference_fold.train_seasons),
            "min_category_frequency": preprocessor.min_category_frequency,
            "categorical_features": len(preprocessor.categorical_columns),
            "numerical_features": len(preprocessor.numerical_columns),
            "state_sha256": model.feature_layout.preprocessor_fingerprint,
            "array_contract": array_audit,
        },
        "architecture": {
            "name": model.architecture_name,
            "config": model.get_config(),
            "initialization_seed": INITIALIZATION_SEED,
            "initial_probability": 0.5,
            "initial_probability_source": "fixed neutral architecture constant; no target statistic",
            "parameter_budget_exclusive": PARAMETER_BUDGET,
            "parameter_report": parameter_report,
            "pitcher_id_index_resolved_by_name": model.pitcher_id_index,
            "batter_id_index_resolved_by_name": model.batter_id_index,
            "reserved_missing_index": 0,
            "reserved_oov_index": 1,
            "reserved_embeddings_zeroed_in_forward": True,
            "output_contract": "one unbounded logit per row; sigmoid remains outside the model",
        },
        "design_reasons": [
            "Separate context, pitcher-history, and batter-history towers preserve feature semantics.",
            "Player ID embeddings are gated residual corrections, so OOV and cold-start rows retain history/context fallback.",
            "Future-season and other OOV categorical embeddings are zeroed instead of using untrained random vectors.",
            "Pitcher/batter/context products add bounded within-row interactions without cross-row attention.",
            "A direct numerical wide head supports smooth probability structure for the Brier objective.",
            "LayerNorm and per-row dropout avoid evaluation-batch statistics.",
        ],
        "reference_validation_input_audit": {
            "rows": int(len(validation_inputs)),
            "season_oov_rate": reference_season_oov_rate,
            "pitcher_id_oov_rate": float(
                np.mean(categorical[:, model.pitcher_id_index] == 1)
            ),
            "batter_id_oov_rate": float(
                np.mean(categorical[:, model.batter_id_index] == 1)
            ),
        },
        "initialized_forward_smoke": {
            "rows": smoke_rows,
            "finite_logits": True,
            "output_shape": list(batched.shape),
            "single_vs_batch_max_abs_difference": single_max_difference,
            "permutation_max_abs_difference": permutation_max_difference,
            "weights_saved": False,
            "predictions_saved": False,
        },
        "deferred_to_later_stages": [
            "loss and optimizer execution",
            "epoch selection or checkpointing",
            "Brier/log-loss/ECE computation on model outputs",
            "architecture comparison using validation predictions",
            "hyperparameter tuning",
            "calibration or ensembling",
            "inference packaging or submission",
        ],
        "outputs": [
            "candidate_comparison.csv",
            "feature_layout.csv",
            "parameter_summary.csv",
            "manifest.json",
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_outputs(args.input, args.output_dir)
    architecture = manifest["architecture"]
    assert isinstance(architecture, dict)
    report = architecture["parameter_report"]
    assert isinstance(report, dict)
    print(
        json.dumps(
            {
                "status": "ok",
                "scope": manifest["scope"],
                "architecture": architecture["name"],
                "parameters": report["total_parameters"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
