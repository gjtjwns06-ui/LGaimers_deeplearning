#!/usr/bin/env python3
"""Select a row-local TrackMan-DL complement gate and forward-audit it."""

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
from lg_control.strict_trackman import StrictTrackmanFeatureBuilder  # noqa: E402


OUTPUT = ROOT / "artifacts/trackman_r_complement_v1"
TRACKMAN_PREDICTIONS = ROOT / "artifacts/target1200_trackman_te_v1/predictions.csv"
TRACKMAN_STATE = ROOT / "artifacts/strict_trackman_fold2024.json"
UNCERTAIN_SELECTION = ROOT / "artifacts/uncertain_residual_v1/selection.json"
AUDIT_REFERENCE = ROOT / "artifacts/uncertain_application_v1/audit_predictions.parquet"


def _brier(frame: pd.DataFrame, probability: np.ndarray) -> float:
    target = frame.control_success.to_numpy(np.float64)
    return float(np.mean(np.square(target - np.asarray(probability, dtype=np.float64))))


def _add_context(frame: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["count"] = (
        output.balls_before.astype(str) + "-" + output.strikes_before.astype(str)
    )
    output["direction"] = np.where(output.prediction_trackman.to_numpy(float) >= base, "up", "down")
    distance = np.abs(np.asarray(base, dtype=np.float64) - 0.5)
    output["uncertainty"] = pd.cut(
        distance,
        bins=[-np.inf, 0.025, 0.050, np.inf],
        labels=["u025", "u050", "far"],
    ).astype(str)
    pitcher = output.tm_pitcher_id_mapped.gt(0.5)
    batter = output.tm_batter_id_mapped.gt(0.5)
    output["mapping"] = np.select(
        [pitcher & batter, pitcher, batter],
        ["both", "pitcher", "batter"],
        default="none",
    )
    return output


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if not columns:
        return pd.Series("__global__", index=frame.index, dtype="string")
    result = frame[columns[0]].astype("string")
    for column in columns[1:]:
        result = result.str.cat(frame[column].astype("string"), sep="|")
    return result


def _fit_gate(
    frame: pd.DataFrame,
    base: np.ndarray,
    *,
    columns: tuple[str, ...],
    shrink_rows: int,
    cap: float,
) -> dict[str, object]:
    target = frame.control_success.to_numpy(np.float64)
    disagreement = frame.prediction_trackman.to_numpy(np.float64) - base
    numerator = disagreement * (target - base)
    denominator = np.square(disagreement)
    global_denominator = float(denominator.sum())
    global_weight = (
        float(numerator.sum()) / global_denominator if global_denominator > 1e-12 else 0.0
    )
    global_weight = float(np.clip(global_weight, 0.0, cap))
    mean_denominator = max(float(denominator.mean()), 1e-12)
    keys = _key(frame, columns)
    statistics = pd.DataFrame(
        {"key": keys.to_numpy(), "numerator": numerator, "denominator": denominator}
    ).groupby("key", sort=True, observed=True)[["numerator", "denominator"]].sum()
    groups: dict[str, float] = {}
    for value, row in statistics.iterrows():
        prior_denominator = shrink_rows * mean_denominator
        raw = (
            float(row.numerator) + prior_denominator * global_weight
        ) / (float(row.denominator) + prior_denominator)
        groups[str(value)] = float(np.clip(raw, 0.0, cap))
    return {
        "columns": list(columns),
        "shrink_rows": int(shrink_rows),
        "cap": float(cap),
        "global_weight": global_weight,
        "groups": groups,
    }


def _predict_gate(frame: pd.DataFrame, base: np.ndarray, gate: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    columns = tuple(str(value) for value in gate["columns"])
    keys = _key(frame, columns)
    weights = keys.map(gate["groups"]).fillna(float(gate["global_weight"])).to_numpy(np.float64)
    trackman = frame.prediction_trackman.to_numpy(np.float64)
    probability = np.clip(base + weights * (trackman - base), 1e-6, 1.0 - 1e-6)
    return probability, weights


def _phase_base(
    base_frame: pd.DataFrame,
    fit_months: pd.Series,
    evaluation_months: pd.Series,
    selected_residual: dict[str, object],
) -> tuple[pd.Series, pd.Series]:
    fit_all = base_frame.loc[fit_months].reset_index(drop=True)
    evaluation_all = base_frame.loc[evaluation_months].reset_index(drop=True)
    fit_probability, evaluation_probability = _phase_v21(
        fit_all, evaluation_all, selected_residual
    )
    return (
        pd.Series(fit_probability, index=fit_all.row_id),
        pd.Series(evaluation_probability, index=evaluation_all.row_id),
    )


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_residual = json.loads(UNCERTAIN_SELECTION.read_text(encoding="utf-8"))["selected"]
    args = SimpleNamespace(
        train_csv=ROOT / "data/train.csv",
        research_root=ROOT.parent / "LG-aimers-v11-independent-research",
        dl_oof=ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    base_frame = residual._load_frame(args)
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    raw_2024 = raw.loc[raw.season.eq(2024)].reset_index(drop=True)
    trackman_features = StrictTrackmanFeatureBuilder.load(TRACKMAN_STATE).transform(raw_2024)
    context = raw_2024[["row_id"]].copy()
    context["tm_pitcher_id_mapped"] = trackman_features.tm_pitcher_id_mapped.to_numpy(float)
    context["tm_batter_id_mapped"] = trackman_features.tm_batter_id_mapped.to_numpy(float)
    trackman_prediction = pd.read_csv(TRACKMAN_PREDICTIONS, usecols=["row_id", "prediction"]).rename(
        columns={"prediction": "prediction_trackman"}
    )
    frame = (
        base_frame.merge(context, on="row_id", how="inner", validate="one_to_one")
        .merge(trackman_prediction, on="row_id", how="inner", validate="one_to_one")
    )
    frame = frame.loc[frame.game_type.astype(str).str.upper().eq("R")].reset_index(drop=True)
    del raw, raw_2024, trackman_features

    dev_fit = frame.loc[frame.game_month.le(5)].reset_index(drop=True)
    validation = frame.loc[frame.game_month.between(6, 7)].reset_index(drop=True)
    fit_lookup, validation_lookup = _phase_base(
        base_frame,
        base_frame.game_month.le(5),
        base_frame.game_month.between(6, 7),
        selected_residual,
    )
    dev_base = dev_fit.row_id.map(fit_lookup).to_numpy(float)
    validation_base = validation.row_id.map(validation_lookup).to_numpy(float)
    dev_fit = _add_context(dev_fit, dev_base)
    validation = _add_context(validation, validation_base)

    key_sets = (
        (),
        ("count",),
        ("uncertainty",),
        ("mapping",),
        ("count", "direction"),
        ("uncertainty", "direction"),
        ("mapping", "direction"),
        ("count", "uncertainty"),
    )
    rows: list[dict[str, object]] = []
    gates: dict[tuple[object, ...], dict[str, object]] = {}
    for columns in key_sets:
        for shrink_rows in (250, 1000, 5000):
            for cap in (0.10, 0.15, 0.20):
                gate = _fit_gate(
                    dev_fit,
                    dev_base,
                    columns=columns,
                    shrink_rows=shrink_rows,
                    cap=cap,
                )
                probability, weights = _predict_gate(validation, validation_base, gate)
                row = {
                    "columns": "+".join(columns) if columns else "global",
                    "shrink_rows": shrink_rows,
                    "cap": cap,
                    "validation_brier": _brier(validation, probability),
                    "validation_delta": _brier(validation, probability) - _brier(validation, validation_base),
                    "validation_mean_weight": float(weights.mean()),
                    "development_global_weight": float(gate["global_weight"]),
                }
                rows.append(row)
                gates[(row["columns"], shrink_rows, cap)] = gate
    candidates = pd.DataFrame(rows).sort_values(["validation_brier", "columns"])
    chosen = candidates.iloc[0].to_dict()
    print(candidates.head(20).to_string(index=False), flush=True)

    audit_fit = frame.loc[frame.game_month.le(7)].reset_index(drop=True)
    audit = frame.loc[frame.game_month.ge(8)].reset_index(drop=True)
    audit_fit_lookup, audit_lookup = _phase_base(
        base_frame,
        base_frame.game_month.le(7),
        base_frame.game_month.ge(8),
        selected_residual,
    )
    audit_fit_base = audit_fit.row_id.map(audit_fit_lookup).to_numpy(float)
    audit_base = audit.row_id.map(audit_lookup).to_numpy(float)
    audit_fit = _add_context(audit_fit, audit_fit_base)
    audit = _add_context(audit, audit_base)
    reference = pd.read_parquet(AUDIT_REFERENCE)[["row_id", "prediction_selected_application"]]
    reference = audit[["row_id"]].merge(reference, on="row_id", validate="one_to_one")
    max_difference = float(
        np.max(np.abs(audit_base - reference.prediction_selected_application.to_numpy(float)))
    )
    if max_difference > 1e-8:
        raise RuntimeError(f"reconstructed v21 audit base differs by {max_difference}")
    chosen_columns = () if chosen["columns"] == "global" else tuple(str(chosen["columns"]).split("+"))
    final_gate = _fit_gate(
        audit_fit,
        audit_fit_base,
        columns=chosen_columns,
        shrink_rows=int(chosen["shrink_rows"]),
        cap=float(chosen["cap"]),
    )
    audit_probability, audit_weights = _predict_gate(audit, audit_base, final_gate)
    audit_delta = _brier(audit, audit_probability) - _brier(audit, audit_base)
    months = []
    for month, positions in audit.groupby("game_month").groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        months.append(
            {
                "month": int(month),
                "rows": int(len(index)),
                "delta": _brier(audit.iloc[index], audit_probability[index])
                - _brier(audit.iloc[index], audit_base[index]),
                "mean_weight": float(audit_weights[index].mean()),
            }
        )
    passed = bool(chosen["validation_delta"] < -1e-6 and audit_delta < -1e-6)
    output = audit[["row_id", "game_month", "control_success"]].copy()
    output["prediction_v21"] = audit_base
    output["prediction_trackman"] = audit.prediction_trackman.to_numpy(float)
    output["gate_weight"] = audit_weights
    output["prediction_candidate"] = audit_probability
    output.to_parquet(OUTPUT / "audit_predictions.parquet", index=False)
    candidates.to_csv(OUTPUT / "candidates.csv", index=False)
    payload = {
        "decision": "PASS" if passed else "FAIL",
        "chosen": chosen,
        "audit_delta": audit_delta,
        "audit_months": months,
        "audit_mean_weight": float(audit_weights.mean()),
        "audit_gate": final_gate,
        "base_reconstruction_max_abs": max_difference,
    }
    (OUTPUT / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
