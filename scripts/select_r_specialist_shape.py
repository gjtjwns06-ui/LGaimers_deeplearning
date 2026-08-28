#!/usr/bin/env python3
"""Select and forward-audit a level-free correction from the new R DL heads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_uncertain_residual as residual  # noqa: E402
from lg_control.hybrid_residual import predict_bounded_residual  # noqa: E402


OUTPUT = ROOT / "artifacts/r_specialist_shape_v1"
SPECIALIST = ROOT / "artifacts/r_specialist_outer_v1/outer_2024_predictions.parquet"
UNCERTAIN_SELECTION = ROOT / "artifacts/uncertain_residual_v1/selection.json"
AUDIT_REFERENCE = ROOT / "artifacts/uncertain_application_v1/audit_predictions.parquet"
HEAD_NAMES = (
    "outcome_control_success",
    "outcome_reverse_only",
    "outcome_middle_only",
    "outcome_reverse_and_middle",
    "outcome_other_failure",
    "aux_ball",
    "aux_strike",
)


def _phase_v21(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor, models, eval_focus_probability, _, _ = residual._fit_fixed_bundle(
        fit, evaluation, selected, "cpu"
    )
    fit_base, _ = residual._conditional_prediction(fit, fit)
    eval_base, _ = residual._conditional_prediction(fit, evaluation)

    fit_focus = fit.near_half.to_numpy(bool)
    if fit_focus.any():
        subset = fit.loc[fit_focus].reset_index(drop=True)
        features = preprocessor.transform(
            subset,
            fit.loc[fit_focus, "prediction_v17"].to_numpy(float),
            fit.loc[fit_focus, "prediction_dl"].to_numpy(float),
        )
        probabilities = [
            predict_bounded_residual(
                model,
                features,
                fit.loc[fit_focus, "prediction_v17"].to_numpy(float),
                batch_size=32768,
                device="cpu",
            )[0]
            for model in models
        ]
        fit_residual = np.mean(probabilities, axis=0)
    else:
        fit_residual = np.empty(0)
    fit_apply = fit_focus & fit.game_type.astype(str).str.upper().eq("R").to_numpy()
    fit_focus_positions = np.flatnonzero(fit_focus)
    fit_apply_positions = np.flatnonzero(fit_apply)
    fit_lookup = {row: position for position, row in enumerate(fit_focus_positions)}
    fit_base[fit_apply_positions] = fit_residual[
        [fit_lookup[row] for row in fit_apply_positions]
    ]

    eval_focus = evaluation.near_half.to_numpy(bool)
    eval_apply = eval_focus & evaluation.game_type.astype(str).str.upper().eq("R").to_numpy()
    eval_focus_positions = np.flatnonzero(eval_focus)
    eval_apply_positions = np.flatnonzero(eval_apply)
    eval_lookup = {row: position for position, row in enumerate(eval_focus_positions)}
    eval_base[eval_apply_positions] = eval_focus_probability[
        [eval_lookup[row] for row in eval_apply_positions]
    ]
    return fit_base, eval_base


def _features(frame: pd.DataFrame, base: np.ndarray, mode: str) -> np.ndarray:
    reverse = frame.head_outcome_reverse_only + frame.head_outcome_reverse_and_middle
    middle = frame.head_outcome_middle_only + frame.head_outcome_reverse_and_middle
    values = [
        reverse.to_numpy(float),
        middle.to_numpy(float),
        reverse.to_numpy(float) - frame.asof_pitcher_reverse_rate.fillna(reverse.mean()).to_numpy(float),
        middle.to_numpy(float) - frame.asof_pitcher_middle_rate.fillna(middle.mean()).to_numpy(float),
        *(frame[f"head_{name}"].to_numpy(float) for name in HEAD_NAMES),
    ]
    if mode == "heads_context":
        values.extend(
            [
                frame.prediction_specialist.to_numpy(float) - base,
                np.abs(frame.prediction_specialist.to_numpy(float) - base),
                np.abs(base - 0.5),
                frame.balls_before.to_numpy(float),
                frame.strikes_before.to_numpy(float),
                frame.asof_pitcher_success_rate.fillna(0.5).to_numpy(float),
            ]
        )
    return np.column_stack(values)


def _fit_predict(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    fit_base: np.ndarray,
    evaluation_base: np.ndarray,
    *,
    mode: str,
    alpha: float,
    cap: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, object]]:
    fit_x = _features(fit, fit_base, mode)
    eval_x = _features(evaluation, evaluation_base, mode)
    mean = fit_x.mean(axis=0)
    scale = fit_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    fit_x = (fit_x - mean) / scale
    eval_x = (eval_x - mean) / scale
    target = fit.control_success.to_numpy(float) - fit_base
    removed_level = float(target.mean())
    model = Ridge(alpha=alpha, fit_intercept=False).fit(fit_x, target - removed_level)
    raw = model.predict(eval_x)
    correction = strength * np.clip(raw, -cap, cap)
    probability = np.clip(evaluation_base + correction, 1e-6, 1.0 - 1e-6)
    return probability, {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": model.coef_.tolist(),
        "removed_level": removed_level,
        "mean_correction": float(correction.mean()),
        "cap_hit_rate": float((np.abs(raw) >= cap).mean()),
    }


def _brier(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(np.mean(np.square(frame.control_success.to_numpy(float) - probability)))


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    selected_residual = json.loads(UNCERTAIN_SELECTION.read_text(encoding="utf-8"))["selected"]
    args = SimpleNamespace(
        train_csv=ROOT / "data/train.csv",
        research_root=ROOT.parent / "LG-aimers-v11-independent-research",
        dl_oof=ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    base_frame = residual._load_frame(args)
    specialist = pd.read_parquet(SPECIALIST).rename(columns={"prediction": "prediction_specialist"})
    keep = ["row_id", "prediction_specialist", *(f"head_{name}" for name in HEAD_NAMES)]
    frame = base_frame.merge(specialist[keep], on="row_id", how="inner", validate="one_to_one")
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "row_id",
            "balls_before",
            "strikes_before",
            "asof_pitcher_success_rate",
            "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate",
        ],
        low_memory=False,
    )
    duplicate = [column for column in raw if column in frame and column != "row_id"]
    frame = frame.merge(raw.drop(columns=duplicate), on="row_id", validate="one_to_one")
    frame = frame[frame.game_type.astype(str).str.upper().eq("R")].reset_index(drop=True)

    dev_fit = frame[frame.game_month.le(5)].reset_index(drop=True)
    validation = frame[frame.game_month.between(6, 7)].reset_index(drop=True)
    audit_fit = frame[frame.game_month.le(7)].reset_index(drop=True)
    audit = frame[frame.game_month.ge(8)].reset_index(drop=True)
    dev_fit_all = base_frame[base_frame.game_month.le(5)].reset_index(drop=True)
    validation_all = base_frame[base_frame.game_month.between(6, 7)].reset_index(drop=True)
    dev_fit_base_all, validation_base_all = _phase_v21(
        dev_fit_all, validation_all, selected_residual
    )
    dev_fit_lookup = pd.Series(dev_fit_base_all, index=dev_fit_all.row_id)
    validation_lookup = pd.Series(validation_base_all, index=validation_all.row_id)
    dev_fit_base = dev_fit.row_id.map(dev_fit_lookup).to_numpy(float)
    validation_base = validation.row_id.map(validation_lookup).to_numpy(float)
    rows: list[dict[str, object]] = []
    for mode in ("heads", "heads_context"):
        for alpha in (100.0, 1000.0, 10000.0, 100000.0):
            for cap in (0.005, 0.010, 0.020):
                for strength in (0.25, 0.50, 1.00):
                    probability, _ = _fit_predict(
                        dev_fit,
                        validation,
                        dev_fit_base,
                        validation_base,
                        mode=mode,
                        alpha=alpha,
                        cap=cap,
                        strength=strength,
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "alpha": alpha,
                            "cap": cap,
                            "strength": strength,
                            "validation_brier": _brier(validation, probability),
                            "validation_delta": _brier(validation, probability) - _brier(validation, validation_base),
                        }
                    )
    candidates = pd.DataFrame(rows).sort_values("validation_brier")
    chosen = candidates.iloc[0].to_dict()
    print(candidates.head(20).to_string(index=False), flush=True)

    audit_fit_all = base_frame[base_frame.game_month.le(7)].reset_index(drop=True)
    audit_all = base_frame[base_frame.game_month.ge(8)].reset_index(drop=True)
    audit_fit_base_all, reconstructed_audit_base_all = _phase_v21(
        audit_fit_all, audit_all, selected_residual
    )
    audit_fit_lookup = pd.Series(audit_fit_base_all, index=audit_fit_all.row_id)
    audit_lookup = pd.Series(reconstructed_audit_base_all, index=audit_all.row_id)
    audit_fit_base = audit_fit.row_id.map(audit_fit_lookup).to_numpy(float)
    reconstructed_audit_base = audit.row_id.map(audit_lookup).to_numpy(float)
    reference = pd.read_parquet(AUDIT_REFERENCE)[["row_id", "prediction_selected_application"]]
    reference = audit[["row_id"]].merge(reference, on="row_id", validate="one_to_one")
    max_difference = float(
        np.max(np.abs(reconstructed_audit_base - reference.prediction_selected_application.to_numpy(float)))
    )
    if max_difference > 1e-8:
        raise RuntimeError(f"reconstructed v21 audit base differs by {max_difference}")
    audit_probability, audit_state = _fit_predict(
        audit_fit,
        audit,
        audit_fit_base,
        reconstructed_audit_base,
        mode=str(chosen["mode"]),
        alpha=float(chosen["alpha"]),
        cap=float(chosen["cap"]),
        strength=float(chosen["strength"]),
    )
    audit_delta = _brier(audit, audit_probability) - _brier(audit, reconstructed_audit_base)
    month_rows = []
    for month, positions in audit.groupby("game_month").groups.items():
        index = np.asarray(list(positions), dtype=int)
        month_rows.append(
            {
                "month": int(month),
                "rows": int(len(index)),
                "delta": _brier(audit.iloc[index], audit_probability[index])
                - _brier(audit.iloc[index], reconstructed_audit_base[index]),
            }
        )
    passed = bool(chosen["validation_delta"] < -1e-6 and audit_delta < -1e-6)
    output = audit[["row_id", "game_month", "control_success"]].copy()
    output["prediction_v21"] = reconstructed_audit_base
    output["prediction_candidate"] = audit_probability
    output.to_parquet(OUTPUT / "audit_predictions.parquet", index=False)
    candidates.to_csv(OUTPUT / "candidates.csv", index=False)
    payload = {
        "decision": "PASS" if passed else "FAIL",
        "chosen": chosen,
        "audit_delta": audit_delta,
        "audit_months": month_rows,
        "audit_state": audit_state,
        "base_reconstruction_max_abs": max_difference,
    }
    (OUTPUT / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
