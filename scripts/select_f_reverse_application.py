#!/usr/bin/env python3
"""Select a narrow row-local application scope for the F reverse residual."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_f_regime_reverse_residual as training  # noqa: E402
from lg_control.f_regime_residual import FRegimeResidualPreprocessor  # noqa: E402
from lg_control.hybrid_residual import predict_bounded_residual, train_bounded_residual  # noqa: E402


ARTIFACT = ROOT / "artifacts/f_regime_reverse_residual_v1"


def mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    up = frame["logit_correction"].to_numpy(float) >= 0.0
    uncertainty = np.abs(frame["prediction_v20_base"].to_numpy(float) - 0.5)
    reverse = frame["predicted_reverse"].to_numpy(float)
    definitions = {
        "all": np.ones(len(frame), dtype=bool),
        "correction_up": up,
        "uncertainty_ge_025": uncertainty >= 0.025,
        "uncertainty_ge_050": uncertainty >= 0.050,
        "up_uncertainty_ge_025": up & (uncertainty >= 0.025),
        "up_uncertainty_ge_050": up & (uncertainty >= 0.050),
        "reverse_below_028": reverse < 0.28,
        "reverse_at_least_028": reverse >= 0.28,
        "up_reverse_below_028": up & (reverse < 0.28),
        "up_reverse_at_least_028": up & (reverse >= 0.28),
    }
    return definitions[scope]


def evaluate(frame: pd.DataFrame, scope: str) -> dict[str, float | int | str]:
    selected = mask(frame, scope)
    base = frame.prediction_v20_base.to_numpy(float)
    raw = frame.prediction_raw_residual.to_numpy(float)
    probability = base.copy()
    probability[selected] = raw[selected]
    y = frame.control_success.to_numpy(float)
    base_brier = float(np.mean(np.square(y - base)))
    brier = float(np.mean(np.square(y - probability)))
    return {
        "scope": scope,
        "rows": int(selected.sum()),
        "coverage": float(selected.mean()),
        "base_brier": base_brier,
        "brier": brier,
        "delta_brier": brier - base_brier,
    }


def main() -> None:
    torch.set_num_threads(8)
    selection = json.loads((ARTIFACT / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected"]
    frame = training.load_frame()
    development_train = frame[frame.game_month.le(5)].reset_index(drop=True)
    validation = frame[frame.game_month.between(6, 7)].reset_index(drop=True)
    train_base, validation_base = training.split_base(development_train, validation)
    preprocessor = FRegimeResidualPreprocessor(include_reverse_heads=True).fit(
        development_train, train_base, training.head_mapping(development_train)
    )
    train_x = preprocessor.transform(
        development_train, train_base, training.head_mapping(development_train)
    )
    validation_x = preprocessor.transform(
        validation, validation_base, training.head_mapping(validation)
    )
    net = training.model(train_x.shape[1], float(selected["cap"]), 42)
    result = train_bounded_residual(
        net,
        train_x,
        train_base.astype(np.float32),
        development_train.control_success.to_numpy(np.float32),
        validation_x,
        validation_base.astype(np.float32),
        validation.control_success.to_numpy(np.float32),
        config=training.train_config(42),
        device="cpu",
    )
    if result.best_epoch != int(selected["selected_epoch"]):
        raise RuntimeError("reproduced development epoch differs")
    validation_table = validation[["row_id", "control_success"]].copy()
    validation_table["prediction_v20_base"] = validation_base
    validation_table["prediction_raw_residual"] = result.probabilities
    validation_table["logit_correction"] = result.corrections
    validation_table["predicted_reverse"] = (
        validation.dlh_outcome_reverse_only.to_numpy(float)
        + validation.dlh_outcome_reverse_and_middle.to_numpy(float)
    )

    audit_file = pd.read_parquet(ARTIFACT / "audit_predictions.parquet")
    audit_rows = frame[frame.game_month.ge(8)][
        ["row_id", "dlh_outcome_reverse_only", "dlh_outcome_reverse_and_middle"]
    ]
    audit_table = audit_file.merge(audit_rows, on="row_id", validate="one_to_one")
    audit_table = audit_table.rename(
        columns={"prediction_f_reverse_residual": "prediction_raw_residual"}
    )
    audit_table["predicted_reverse"] = (
        audit_table.dlh_outcome_reverse_only + audit_table.dlh_outcome_reverse_and_middle
    )

    scopes = (
        "all",
        "correction_up",
        "uncertainty_ge_025",
        "uncertainty_ge_050",
        "up_uncertainty_ge_025",
        "up_uncertainty_ge_050",
        "reverse_below_028",
        "reverse_at_least_028",
        "up_reverse_below_028",
        "up_reverse_at_least_028",
    )
    validation_results = [evaluate(validation_table, scope) for scope in scopes]
    chosen = min(validation_results, key=lambda item: item["brier"])
    audit_results = [evaluate(audit_table, scope) for scope in scopes]
    audit_chosen = next(item for item in audit_results if item["scope"] == chosen["scope"])
    passed = bool(chosen["delta_brier"] < -1e-6 and audit_chosen["delta_brier"] < -1e-6)
    payload = {
        "decision": "PASS" if passed else "FAIL",
        "kind": "f_regime_reverse_application_v1",
        "chosen": chosen,
        "validation_candidates": validation_results,
        "audit_candidates_diagnostic": audit_results,
        "audit": audit_chosen,
        "selection_rule": "minimum validation Brier; audit used only as pass/fail",
    }
    (ARTIFACT / "application.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_table.to_parquet(ARTIFACT / "validation_application_predictions.parquet", index=False)
    print(pd.DataFrame(validation_results).to_string(index=False))
    print("\nAUDIT")
    print(pd.DataFrame(audit_results).to_string(index=False))
    print(
        f"decision={payload['decision']} chosen={chosen['scope']} "
        f"validation_delta={chosen['delta_brier']:+.10f} "
        f"audit_delta={audit_chosen['delta_brier']:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
