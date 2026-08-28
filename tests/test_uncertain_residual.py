from __future__ import annotations

import numpy as np
import pandas as pd

from lg_control.uncertain_residual import (
    BASE_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    UncertainResidualPreprocessor,
)


def _frame(rows: int = 8) -> pd.DataFrame:
    payload: dict[str, object] = {}
    for position, column in enumerate(RAW_NUMERIC_COLUMNS):
        payload[column] = np.linspace(position, position + 1, rows)
    payload.update(
        {
            "game_type": ["F", "R"] * (rows // 2),
            "top_bottom": [0, 1] * (rows // 2),
            "outs_before": np.arange(rows) % 3,
            "base_state": np.arange(rows) % 8,
            "pitcher_hand": np.arange(rows) % 2,
            "batter_hand": (np.arange(rows) + 1) % 2,
            "balls_before": np.arange(rows) % 4,
            "strikes_before": np.arange(rows) % 3,
        }
    )
    payload["asof_pitcher_prev1_game_success_rate"] = [np.nan] + [0.5] * (rows - 1)
    return pd.DataFrame(payload)


def test_uncertain_preprocessor_round_trip() -> None:
    frame = _frame()
    v17 = np.linspace(0.48, 0.52, len(frame))
    dl = np.linspace(0.50, 0.46, len(frame))
    fitted = UncertainResidualPreprocessor(include_count_context=True).fit(
        frame, v17, dl
    )
    expected = fitted.transform(frame, v17, dl)
    restored = UncertainResidualPreprocessor.from_state_dict(fitted.to_state_dict())
    observed = restored.transform(frame, v17, dl)
    np.testing.assert_array_equal(expected, observed)
    assert np.isfinite(observed).all()
    assert any(name.startswith("count_state==") for name in restored.feature_names_)


def test_count_context_adds_explicit_count_interactions() -> None:
    frame = _frame()
    v17 = np.full(len(frame), 0.5)
    dl = np.linspace(0.45, 0.55, len(frame))
    basic = UncertainResidualPreprocessor(include_count_context=False).fit(
        frame, v17, dl
    )
    count = UncertainResidualPreprocessor(include_count_context=True).fit(
        frame, v17, dl
    )
    assert set(BASE_CATEGORICAL_COLUMNS).issubset(basic.category_values_)
    assert len(count.feature_names_) > len(basic.feature_names_)
