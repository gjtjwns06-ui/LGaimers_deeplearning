from __future__ import annotations

import numpy as np
import pandas as pd

from lg_control.f_regime_residual import HEAD_COLUMNS
from lg_control.f_regime_shape import (
    FEATURE_NAMES,
    FRegimeShapeArtifact,
    build_f_regime_shape_features,
    predict_f_regime_shape,
)


def _heads(rows: int) -> dict[str, np.ndarray]:
    outcome = {
        "outcome_control_success": 0.46,
        "outcome_reverse_only": 0.27,
        "outcome_middle_only": 0.09,
        "outcome_reverse_and_middle": 0.04,
        "outcome_other_failure": 0.14,
    }
    return {
        **{name: np.full(rows, value) for name, value in outcome.items()},
        "aux_ball": np.full(rows, 0.38),
        "aux_strike": np.full(rows, 0.43),
    }


def test_shape_features_expose_reverse_gap() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_reverse_rate": [0.20, np.nan],
            "asof_pitcher_middle_rate": [0.10, np.nan],
        }
    )
    values = build_f_regime_shape_features(frame, _heads(2))
    assert values.shape == (2, len(FEATURE_NAMES))
    assert np.isclose(values[0, 0], 0.31)
    assert np.isclose(values[0, 2], 0.11)
    assert np.isclose(values[1, 2], 0.0)


def test_shape_correction_is_f_only_and_row_independent() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["F", "R", "F"],
            "asof_pitcher_reverse_rate": [0.20, 0.25, 0.30],
            "asof_pitcher_middle_rate": [0.10, 0.11, 0.12],
        }
    )
    artifact = FRegimeShapeArtifact(
        feature_mean=np.zeros(len(FEATURE_NAMES)),
        feature_scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.full(len(FEATURE_NAMES), 0.001),
        cap=0.02,
        strength=0.5,
        manifest={},
    )
    base = np.array([0.45, 0.55, 0.50])
    heads = _heads(3)
    full, _, mask = predict_f_regime_shape(artifact, frame, base, heads)
    assert mask.tolist() == [True, False, True]
    assert full[1] == base[1]
    for index in (0, 2):
        one_frame = frame.iloc[[index]].reset_index(drop=True)
        one_heads = {name: values[[index]] for name, values in heads.items()}
        one, _, _ = predict_f_regime_shape(
            artifact, one_frame, base[[index]], one_heads
        )
        assert np.isclose(one[0], full[index])


def test_head_contract_is_complete() -> None:
    assert len(HEAD_COLUMNS) == 7
