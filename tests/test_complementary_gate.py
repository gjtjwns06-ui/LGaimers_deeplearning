from __future__ import annotations

import numpy as np
import pandas as pd

from lg_control.complementary_gate import (
    ComplementaryGateArtifact,
    predict_complementary_hybrid,
)


def _artifact() -> ComplementaryGateArtifact:
    return ComplementaryGateArtifact(
        table=pd.DataFrame(
            {
                "game_type": ["F", "R"],
                "count": ["0-0", "3-2"],
                "direction": ["up", "down"],
                "weight": [0.15, 0.0],
            }
        ),
        fallback_weight=0.05,
        maximum_weight=0.15,
    )


def test_gate_applies_state_weight_and_fallback() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["F", "R", "R"],
            "balls_before": [0, 3, 1],
            "strikes_before": [0, 2, 1],
        }
    )
    v17 = np.array([0.40, 0.60, 0.50])
    dl = np.array([0.50, 0.40, 0.60])
    probability, weight = predict_complementary_hybrid(
        frame, v17, dl, _artifact()
    )
    np.testing.assert_allclose(weight, [0.15, 0.0, 0.05])
    np.testing.assert_allclose(probability, [0.415, 0.60, 0.505])


def test_gate_is_row_independent_under_shuffle() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["F", "R", "R", "F"],
            "balls_before": [0, 3, 1, 2],
            "strikes_before": [0, 2, 1, 2],
        }
    )
    v17 = np.array([0.40, 0.60, 0.50, 0.52])
    dl = np.array([0.50, 0.40, 0.60, 0.48])
    expected, expected_weight = predict_complementary_hybrid(
        frame, v17, dl, _artifact()
    )
    order = np.array([2, 0, 3, 1])
    observed, observed_weight = predict_complementary_hybrid(
        frame.iloc[order].reset_index(drop=True), v17[order], dl[order], _artifact()
    )
    inverse = np.argsort(order)
    np.testing.assert_allclose(observed[inverse], expected)
    np.testing.assert_allclose(observed_weight[inverse], expected_weight)
