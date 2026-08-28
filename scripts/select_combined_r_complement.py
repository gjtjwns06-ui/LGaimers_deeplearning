#!/usr/bin/env python3
"""Select a bounded gate from v21 toward the combined-signal DCNv2 R expert."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_uncertain_residual as residual  # noqa: E402
from select_r_specialist_shape import _phase_v21  # noqa: E402
from select_trackman_r_complement import _brier, _fit_gate, _predict_gate  # noqa: E402


OUTPUT = ROOT / "artifacts/combined_r_complement_v1"
SPECIALIST = ROOT / "artifacts/target1200_combined_mtl_v1/predictions.csv"
UNCERTAIN_SELECTION = ROOT / "artifacts/uncertain_residual_v1/selection.json"
AUDIT_REFERENCE = ROOT / "artifacts/uncertain_application_v1/audit_predictions.parquet"


def _context(frame: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["count"] = output.balls_before.astype(str) + "-" + output.strikes_before.astype(str)
    output["direction"] = np.where(
        output.prediction_trackman.to_numpy(float) >= base, "up", "down"
    )
    output["uncertainty"] = pd.cut(
        np.abs(np.asarray(base, dtype=float) - 0.5),
        [-np.inf, 0.025, 0.050, np.inf],
        labels=["u025", "u050", "far"],
    ).astype(str)
    return output


def _phase(
    base_frame: pd.DataFrame,
    fit_mask: pd.Series,
    evaluation_mask: pd.Series,
    selected: dict[str, object],
) -> tuple[pd.Series, pd.Series]:
    fit = base_frame.loc[fit_mask].reset_index(drop=True)
    evaluation = base_frame.loc[evaluation_mask].reset_index(drop=True)
    fit_probability, evaluation_probability = _phase_v21(fit, evaluation, selected)
    return (
        pd.Series(fit_probability, index=fit.row_id),
        pd.Series(evaluation_probability, index=evaluation.row_id),
    )


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads(UNCERTAIN_SELECTION.read_text(encoding="utf-8"))["selected"]
    args = SimpleNamespace(
        train_csv=ROOT / "data/train.csv",
        research_root=ROOT.parent / "LG-aimers-v11-independent-research",
        dl_oof=ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    base_frame = residual._load_frame(args)
    specialist = pd.read_csv(SPECIALIST, usecols=["row_id", "prediction"]).rename(
        columns={"prediction": "prediction_trackman"}
    )
    frame = base_frame.merge(specialist, on="row_id", how="inner", validate="one_to_one")
    frame = frame.loc[frame.game_type.astype(str).str.upper().eq("R")].reset_index(drop=True)

    dev = frame.loc[frame.game_month.le(5)].reset_index(drop=True)
    validation = frame.loc[frame.game_month.between(6, 7)].reset_index(drop=True)
    dev_lookup, validation_lookup = _phase(
        base_frame,
        base_frame.game_month.le(5),
        base_frame.game_month.between(6, 7),
        selected,
    )
    dev_base = dev.row_id.map(dev_lookup).to_numpy(float)
    validation_base = validation.row_id.map(validation_lookup).to_numpy(float)
    dev = _context(dev, dev_base)
    validation = _context(validation, validation_base)

    keysets = (
        (),
        ("count",),
        ("uncertainty",),
        ("direction",),
        ("count", "direction"),
        ("uncertainty", "direction"),
        ("count", "uncertainty"),
    )
    rows: list[dict[str, object]] = []
    for columns in keysets:
        for shrink_rows in (250, 1000, 5000):
            for cap in (0.10, 0.20, 0.30):
                gate = _fit_gate(
                    dev,
                    dev_base,
                    columns=columns,
                    shrink_rows=shrink_rows,
                    cap=cap,
                )
                probability, weights = _predict_gate(validation, validation_base, gate)
                rows.append(
                    {
                        "columns": "+".join(columns) if columns else "global",
                        "shrink_rows": shrink_rows,
                        "cap": cap,
                        "validation_brier": _brier(validation, probability),
                        "validation_delta": _brier(validation, probability)
                        - _brier(validation, validation_base),
                        "validation_mean_weight": float(weights.mean()),
                        "development_global_weight": float(gate["global_weight"]),
                    }
                )
    candidates = pd.DataFrame(rows).sort_values(["validation_brier", "columns"])
    chosen = candidates.iloc[0].to_dict()
    print(candidates.head(20).to_string(index=False), flush=True)

    audit_fit = frame.loc[frame.game_month.le(7)].reset_index(drop=True)
    audit = frame.loc[frame.game_month.ge(8)].reset_index(drop=True)
    audit_fit_lookup, audit_lookup = _phase(
        base_frame,
        base_frame.game_month.le(7),
        base_frame.game_month.ge(8),
        selected,
    )
    audit_fit_base = audit_fit.row_id.map(audit_fit_lookup).to_numpy(float)
    audit_base = audit.row_id.map(audit_lookup).to_numpy(float)
    audit_fit = _context(audit_fit, audit_fit_base)
    audit = _context(audit, audit_base)
    reference = pd.read_parquet(AUDIT_REFERENCE)[["row_id", "prediction_selected_application"]]
    reference = audit[["row_id"]].merge(reference, on="row_id", validate="one_to_one")
    max_difference = float(
        np.max(np.abs(audit_base - reference.prediction_selected_application.to_numpy(float)))
    )
    if max_difference > 1e-8:
        raise RuntimeError(f"reconstructed v21 audit base differs by {max_difference}")
    columns = () if chosen["columns"] == "global" else tuple(str(chosen["columns"]).split("+"))
    gate = _fit_gate(
        audit_fit,
        audit_fit_base,
        columns=columns,
        shrink_rows=int(chosen["shrink_rows"]),
        cap=float(chosen["cap"]),
    )
    probability, weights = _predict_gate(audit, audit_base, gate)
    audit_delta = _brier(audit, probability) - _brier(audit, audit_base)
    months = []
    for month, positions in audit.groupby("game_month").groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        months.append(
            {
                "month": int(month),
                "rows": int(len(index)),
                "delta": _brier(audit.iloc[index], probability[index])
                - _brier(audit.iloc[index], audit_base[index]),
                "mean_weight": float(weights[index].mean()),
            }
        )
    passed = bool(chosen["validation_delta"] < -1e-6 and audit_delta < -1e-6)
    output = audit[["row_id", "game_month", "control_success"]].copy()
    output["prediction_v21"] = audit_base
    output["prediction_specialist"] = audit.prediction_trackman.to_numpy(float)
    output["gate_weight"] = weights
    output["prediction_candidate"] = probability
    output.to_parquet(OUTPUT / "audit_predictions.parquet", index=False)
    candidates.to_csv(OUTPUT / "candidates.csv", index=False)
    payload = {
        "decision": "PASS" if passed else "FAIL",
        "chosen": chosen,
        "audit_delta": audit_delta,
        "audit_months": months,
        "audit_mean_weight": float(weights.mean()),
        "audit_gate": gate,
        "base_reconstruction_max_abs": max_difference,
    }
    (OUTPUT / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
