#!/usr/bin/env python3
"""Select where and how strongly the uncertain residual should be applied."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_uncertain_residual as residual  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "artifacts/uncertain_application_v1"
SELECTION_PATH = PROJECT_ROOT / "artifacts/uncertain_residual_v1/selection.json"
AUDIT_PATH = PROJECT_ROOT / "artifacts/uncertain_residual_v1/audit_predictions.parquet"


def _brier(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            np.square(frame["control_success"].to_numpy(float) - prediction)
        )
    )


def _mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    focus = frame["near_half"].to_numpy(bool)
    if scope == "all":
        return focus
    if scope == "R":
        return focus & frame["game_type"].astype(str).str.upper().eq("R").to_numpy()
    if scope == "F":
        return focus & frame["game_type"].astype(str).str.upper().eq("F").to_numpy()
    raise ValueError(f"unknown residual scope: {scope}")


def _apply(
    frame: pd.DataFrame,
    base: np.ndarray,
    residual_probability: np.ndarray,
    *,
    scope: str,
    strength: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    apply_mask = _mask(frame, scope)
    v17 = frame["prediction_v17"].to_numpy(float)
    blended = v17 + strength * (residual_probability - v17)
    output[apply_mask] = blended[apply_mask]
    return np.clip(output, 1e-6, 1.0 - 1e-6)


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selected"]
    args = SimpleNamespace(
        train_csv=PROJECT_ROOT / "data/train.csv",
        research_root=PROJECT_ROOT.parent / "LG-aimers-v11-independent-research",
        dl_oof=PROJECT_ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    frame = residual._load_frame(args)
    train = frame.loc[frame["game_month"].le(5)].copy()
    validation = frame.loc[frame["game_month"].between(6, 7)].copy()
    _, _, focus_probability, _, histories = residual._fit_fixed_bundle(
        train, validation, selected, "cpu"
    )
    validation_base, _ = residual._conditional_prediction(train, validation)
    validation_residual = validation["prediction_v17"].to_numpy(float).copy()
    validation_residual[validation["near_half"].to_numpy()] = focus_probability

    candidates: list[dict[str, object]] = []
    for scope in ("all", "R", "F"):
        for strength in (0.25, 0.50, 0.75, 1.00):
            probability = _apply(
                validation,
                validation_base,
                validation_residual,
                scope=scope,
                strength=strength,
            )
            candidates.append(
                {
                    "scope": scope,
                    "strength": strength,
                    "brier": _brier(validation, probability),
                    "delta_vs_conditional": _brier(validation, probability)
                    - _brier(validation, validation_base),
                }
            )
    chosen = min(candidates, key=lambda item: item["brier"])

    audit_file = pd.read_parquet(AUDIT_PATH)
    audit = frame.loc[frame["game_month"].ge(8)].copy()
    audit_file = audit_file.set_index("row_id").loc[audit["row_id"]].reset_index()
    audit_base = audit_file["prediction_conditional_base"].to_numpy(float)
    audit_residual = audit_file["prediction_uncertain_residual"].to_numpy(float)
    audit_probability = _apply(
        audit,
        audit_base,
        audit_residual,
        scope=str(chosen["scope"]),
        strength=float(chosen["strength"]),
    )
    audit_delta = _brier(audit, audit_probability) - _brier(audit, audit_base)
    decision = bool(chosen["delta_vs_conditional"] < -1e-6 and audit_delta < -1e-6)
    payload = {
        "decision": "PASS" if decision else "FAIL",
        "kind": "uncertain_residual_application_v1",
        "chosen": chosen,
        "candidates": candidates,
        "development_base_brier": _brier(validation, validation_base),
        "audit": {
            "rows": int(len(audit)),
            "base_brier": _brier(audit, audit_base),
            "candidate_brier": _brier(audit, audit_probability),
            "delta_vs_conditional": audit_delta,
        },
        "development_training_histories": histories,
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output = audit_file.copy()
    output["prediction_selected_application"] = audit_probability
    output.to_parquet(OUTPUT_DIR / "audit_predictions.parquet", index=False)
    print(
        f"decision={payload['decision']} scope={chosen['scope']} "
        f"strength={chosen['strength']:.2f} "
        f"dev_delta={chosen['delta_vs_conditional']:+.10f} "
        f"audit_delta={audit_delta:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
