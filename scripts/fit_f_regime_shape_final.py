#!/usr/bin/env python3
"""Fit the frozen level-free F deep-shape correction on full 2024 OOF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_f_regime_reverse_residual as training  # noqa: E402
from lg_control.complementary_gate import (  # noqa: E402
    load_complementary_gate,
    predict_complementary_hybrid,
)
from lg_control.f_regime_shape import FEATURE_NAMES, build_f_regime_shape_features  # noqa: E402


OUTPUT = ROOT / "artifacts/f_regime_deep_shape_v1"
ALPHA = 10000.0
CAP = 0.020
STRENGTH = 0.50


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    frame = training.load_frame()
    gate = load_complementary_gate(ROOT / "artifacts/complementary_hybrid_v1/gate")
    base, _ = predict_complementary_hybrid(
        frame,
        frame.prediction_v17.to_numpy(float),
        frame.prediction_dl.to_numpy(float),
        gate,
    )
    raw = build_f_regime_shape_features(frame, training.head_mapping(frame))
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (raw - mean) / scale
    residual = frame.control_success.to_numpy(float) - base
    residual_mean_removed = float(residual.mean())
    residual = residual - residual_mean_removed
    model = Ridge(alpha=ALPHA, fit_intercept=False).fit(standardized, residual)
    raw_correction = standardized @ model.coef_
    correction = STRENGTH * np.clip(raw_correction, -CAP, CAP)
    probability = np.clip(base + correction, 1e-6, 1.0 - 1e-6)
    y = frame.control_success.to_numpy(float)
    diagnostic_delta = float(
        np.mean(np.square(y - probability)) - np.mean(np.square(y - base))
    )
    payload = {
        "kind": "f_regime_deep_shape_v1",
        "scope": "F",
        "cause": "post-2022 reverse-label/coordinate regime unification",
        "training_source": "leakage-safe 2024 F OOF",
        "training_rows": int(len(frame)),
        "uses_privileged_labels_at_inference": False,
        "level_free": True,
        "removed_training_residual_mean": residual_mean_removed,
        "alpha": ALPHA,
        "cap": CAP,
        "strength": STRENGTH,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": model.coef_.tolist(),
        "in_sample_diagnostic_delta_brier": diagnostic_delta,
        "mean_probability_correction": float(correction.mean()),
        "cap_hit_rate": float((np.abs(raw_correction) >= CAP).mean()),
        "selection_contract": {
            "development_fit": "2024 months 3-5",
            "selection": "2024 months 6-7",
            "forward_audit": "2024 months 8-10",
            "validation_delta_brier": -8.468624521695833e-06,
            "audit_delta_brier": -3.46226045551401e-05,
            "audit_months_all_improved": True,
        },
    }
    (OUTPUT / "artifact.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"saved={OUTPUT} rows={len(frame)} in_sample_delta={diagnostic_delta:+.10f} "
        f"mean_correction={correction.mean():+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
