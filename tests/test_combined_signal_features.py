from __future__ import annotations

import json

import numpy as np

from lg_control.combined_signal_features import (
    COMBINED_SIGNAL_FEATURE_COLUMNS,
    CombinedSignalAugmentedPreprocessor,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.rate_basis_features import RATE_BASIS_FEATURE_COLUMNS
from lg_control.temporal_target_features import TEMPORAL_TARGET_FEATURE_COLUMNS
from tests.test_preprocessing import make_frame


def test_combined_contract_and_serialization_are_exact() -> None:
    frame = make_frame()
    fitted = CombinedSignalAugmentedPreprocessor(min_category_frequency=1).fit(frame)
    expected = fitted.transform(frame)
    layout = fitted.feature_layout()
    assert COMBINED_SIGNAL_FEATURE_COLUMNS == (
        *RATE_BASIS_FEATURE_COLUMNS,
        *TEMPORAL_TARGET_FEATURE_COLUMNS,
    )
    assert expected["numerical"].shape[1] == (
        len(DeepTabularPreprocessor.numerical_columns)
        + len(COMBINED_SIGNAL_FEATURE_COLUMNS)
    )
    assert expected["numerical"].shape[1] == len(layout.numerical_names)
    assert np.isfinite(expected["numerical"]).all()

    restored = CombinedSignalAugmentedPreprocessor.from_state_dict(
        json.loads(json.dumps(fitted.to_state_dict()))
    )
    actual = restored.transform(frame)
    for name in ("categorical", "numerical", "row_id", "target"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert restored.feature_layout() == layout


def test_combined_transform_is_row_and_target_independent() -> None:
    frame = make_frame()
    fitted = CombinedSignalAugmentedPreprocessor(min_category_frequency=1).fit(frame)
    expected = fitted.transform(frame)
    changed = frame.iloc[::-1].copy()
    changed["control_success"] = 1 - changed["control_success"]
    actual = fitted.transform(changed)
    np.testing.assert_array_equal(actual["categorical"][::-1], expected["categorical"])
    np.testing.assert_array_equal(actual["numerical"][::-1], expected["numerical"])
    for position in range(len(frame)):
        singleton = fitted.transform(frame.iloc[[position]])
        np.testing.assert_array_equal(
            singleton["numerical"], expected["numerical"][[position]]
        )
