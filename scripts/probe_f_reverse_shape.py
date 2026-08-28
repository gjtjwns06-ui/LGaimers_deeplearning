#!/usr/bin/env python3
"""Probe level-free linear corrections from the deep reverse-state heads."""

from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_f_regime_reverse_residual as training  # noqa: E402


def features(frame: pd.DataFrame, base: np.ndarray, mode: str) -> np.ndarray:
    reverse = (
        frame.dlh_outcome_reverse_only.to_numpy(float)
        + frame.dlh_outcome_reverse_and_middle.to_numpy(float)
    )
    middle = (
        frame.dlh_outcome_middle_only.to_numpy(float)
        + frame.dlh_outcome_reverse_and_middle.to_numpy(float)
    )
    common = [
        reverse,
        middle,
        reverse - frame.asof_pitcher_reverse_rate.fillna(reverse.mean()).to_numpy(float),
        middle - frame.asof_pitcher_middle_rate.fillna(middle.mean()).to_numpy(float),
    ]
    if mode == "reverse":
        return np.column_stack(common)
    if mode == "exclusive":
        return np.column_stack(
            [
                *common,
                *(frame[f"dlh_{name}"].to_numpy(float) for name in training.HEAD_COLUMNS),
            ]
        )
    if mode == "exclusive_context":
        return np.column_stack(
            [
                *common,
                *(frame[f"dlh_{name}"].to_numpy(float) for name in training.HEAD_COLUMNS),
                frame.prediction_dl.to_numpy(float) - base,
                frame.asof_pitcher_success_rate.fillna(0.5).to_numpy(float),
                frame.asof_pitcher_ball_rate.fillna(0.38).to_numpy(float),
                frame.asof_pitcher_strike_rate.fillna(0.43).to_numpy(float),
                frame.balls_before.to_numpy(float),
                frame.strikes_before.to_numpy(float),
            ]
        )
    raise ValueError(mode)


def fit_predict(
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
    x_fit = features(fit, fit_base, mode)
    x_eval = features(evaluation, evaluation_base, mode)
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_fit = (x_fit - mean) / scale
    x_eval = (x_eval - mean) / scale
    residual = fit.control_success.to_numpy(float) - fit_base
    residual = residual - residual.mean()
    model = Ridge(alpha=alpha, fit_intercept=False).fit(x_fit, residual)
    raw = model.predict(x_eval)
    correction = strength * np.clip(raw, -cap, cap)
    probability = np.clip(evaluation_base + correction, 1e-6, 1.0 - 1e-6)
    return probability, {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": model.coef_.tolist(),
        "mean_correction": float(correction.mean()),
        "cap_hit_rate": float((np.abs(raw) >= cap).mean()),
    }


def delta(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray) -> float:
    y = frame.control_success.to_numpy(float)
    return float(np.mean(np.square(y - candidate)) - np.mean(np.square(y - base)))


def main() -> None:
    full = training.load_frame(all_game_types=True)
    dev_train_all = full[full.game_month.le(5)].reset_index(drop=True)
    validation_all = full[full.game_month.between(6, 7)].reset_index(drop=True)
    audit_train_all = full[full.game_month.le(7)].reset_index(drop=True)
    audit_all = full[full.game_month.ge(8)].reset_index(drop=True)
    train_base_all, validation_base_all = training.split_base(
        dev_train_all, validation_all
    )
    audit_train_base_all, audit_base_all = training.split_base(
        audit_train_all, audit_all
    )
    train_mask = dev_train_all.game_type.astype(str).str.upper().eq("F").to_numpy()
    validation_mask = validation_all.game_type.astype(str).str.upper().eq("F").to_numpy()
    audit_train_mask = audit_train_all.game_type.astype(str).str.upper().eq("F").to_numpy()
    audit_mask = audit_all.game_type.astype(str).str.upper().eq("F").to_numpy()
    dev_train = dev_train_all.loc[train_mask].reset_index(drop=True)
    validation = validation_all.loc[validation_mask].reset_index(drop=True)
    audit_train = audit_train_all.loc[audit_train_mask].reset_index(drop=True)
    audit = audit_all.loc[audit_mask].reset_index(drop=True)
    train_base = train_base_all[train_mask]
    validation_base = validation_base_all[validation_mask]
    audit_train_base = audit_train_base_all[audit_train_mask]
    audit_base = audit_base_all[audit_mask]
    rows: list[dict[str, object]] = []
    for mode in ("reverse", "exclusive", "exclusive_context"):
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            for cap in (0.005, 0.010, 0.020):
                for strength in (0.25, 0.50, 1.00):
                    probability, _ = fit_predict(
                        dev_train,
                        validation,
                        train_base,
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
                            "validation_delta": delta(validation, validation_base, probability),
                        }
                    )
    table = pd.DataFrame(rows).sort_values("validation_delta")
    selected = table.iloc[0].to_dict()
    audit_probability, state = fit_predict(
        audit_train,
        audit,
        audit_train_base,
        audit_base,
        mode=str(selected["mode"]),
        alpha=float(selected["alpha"]),
        cap=float(selected["cap"]),
        strength=float(selected["strength"]),
    )
    selected["audit_delta"] = delta(audit, audit_base, audit_probability)
    print(table.head(30).to_string(index=False))
    print("\nSELECTED", selected)
    print("STATE", state)
    for month, month_frame in audit.assign(p=audit_probability, b=audit_base).groupby("game_month"):
        print("MONTH", month, delta(month_frame, month_frame.b.to_numpy(), month_frame.p.to_numpy()))
    table.to_csv(ROOT / "reports/f_regime_shift/reverse_shape_candidates.csv", index=False)
    audit_output = audit[["row_id", "game_month", "control_success"]].copy()
    audit_output["prediction_v20_base"] = audit_base
    audit_output["prediction_f_deep_shape"] = audit_probability
    audit_output.to_parquet(
        ROOT / "reports/f_regime_shift/reverse_shape_audit_predictions.parquet",
        index=False,
    )
    (ROOT / "reports/f_regime_shift/reverse_shape_selection.json").write_text(
        json.dumps(
            {"selected": selected, "audit_fit_state": state},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
