#!/usr/bin/env python3
"""Refit only the two forward-validated R complement cells on all 2024 OOF."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_uncertain_residual as residual  # noqa: E402
from select_combined_r_complement import _context  # noqa: E402
from select_trackman_r_complement import _brier, _fit_gate, _predict_gate  # noqa: E402
from lg_control.complementary_gate import (  # noqa: E402
    load_complementary_gate,
    predict_complementary_hybrid,
)


OUTPUT = ROOT / "artifacts/final_combined_r_gate"
SPECIALIST = ROOT / "artifacts/target1200_combined_mtl_v1/predictions.csv"
ALLOWED_GROUPS = {"far|up", "u050|down"}


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        train_csv=ROOT / "data/train.csv",
        research_root=ROOT.parent / "LG-aimers-v11-independent-research",
        dl_oof=ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    frame = residual._load_frame(args)
    primary_gate = load_complementary_gate(ROOT / "artifacts/complementary_hybrid_v1/gate")
    base, _ = predict_complementary_hybrid(
        frame,
        frame.prediction_v17.to_numpy(float),
        frame.prediction_dl.to_numpy(float),
        primary_gate,
    )
    specialist = pd.read_csv(SPECIALIST, usecols=["row_id", "prediction"]).rename(
        columns={"prediction": "prediction_trackman"}
    )
    frame = frame.merge(specialist, on="row_id", validate="one_to_one")
    r_mask = frame.game_type.astype(str).str.upper().eq("R").to_numpy()
    r_frame = frame.loc[r_mask].reset_index(drop=True)
    r_base = base[r_mask]
    r_frame = _context(r_frame, r_base)
    gate = _fit_gate(
        r_frame,
        r_base,
        columns=("uncertainty", "direction"),
        shrink_rows=250,
        cap=0.30,
    )
    gate["global_weight"] = 0.0
    gate["groups"] = {
        name: float(weight) if name in ALLOWED_GROUPS else 0.0
        for name, weight in gate["groups"].items()
    }
    probability, weights = _predict_gate(r_frame, r_base, gate)
    group_counts = (
        (r_frame.uncertainty.astype(str) + "|" + r_frame.direction.astype(str))
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    payload = {
        "kind": "combined_signal_r_complement_gate_v1",
        "decision": "PASS",
        "scope": "R",
        "formula": "p_v21 + weight(uncertainty,direction) * (p_combined_dl - p_v21)",
        "selection_source": "months<=5 -> months6-7, then months<=7 -> months>=8",
        "allowed_groups": sorted(ALLOWED_GROUPS),
        "gate": gate,
        "fit_rows": int(len(r_frame)),
        "group_rows": group_counts,
        "mean_weight": float(weights.mean()),
        "fit_brier_base": _brier(r_frame, r_base),
        "fit_brier_candidate": _brier(r_frame, probability),
        "fit_delta": _brier(r_frame, probability) - _brier(r_frame, r_base),
    }
    (OUTPUT / "artifact.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
