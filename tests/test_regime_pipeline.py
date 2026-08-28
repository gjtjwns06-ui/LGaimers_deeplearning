from __future__ import annotations

import json
import unittest

import numpy as np

from lg_control.regime_pipeline import RegimeAugmentedPreprocessor
from tests.test_preprocessing import make_frame


class RegimeAugmentedPreprocessorTests(unittest.TestCase):
    def test_fit_transform_layout_and_round_trip(self) -> None:
        frame = make_frame()
        preprocessor = RegimeAugmentedPreprocessor(min_category_frequency=1).fit(frame)
        expected = preprocessor.transform(frame)
        layout = preprocessor.feature_layout()
        self.assertEqual(expected["numerical"].shape[1], len(layout.numerical_names))
        self.assertEqual(expected["categorical"].shape[1], len(layout.categorical_names))
        self.assertTrue(np.isfinite(expected["numerical"]).all())

        payload = json.loads(json.dumps(preprocessor.to_state_dict()))
        restored = RegimeAugmentedPreprocessor.from_state_dict(payload)
        actual = restored.transform(frame)
        np.testing.assert_array_equal(actual["categorical"], expected["categorical"])
        np.testing.assert_allclose(actual["numerical"], expected["numerical"], rtol=0, atol=0)
        self.assertEqual(restored.feature_layout(), layout)

    def test_transform_is_row_order_and_target_invariant(self) -> None:
        frame = make_frame()
        preprocessor = RegimeAugmentedPreprocessor(min_category_frequency=1).fit(frame)
        batch = preprocessor.transform(frame)
        reversed_frame = frame.iloc[::-1].copy()
        reversed_frame["control_success"] = 1 - reversed_frame["control_success"]
        reversed_arrays = preprocessor.transform(reversed_frame)
        np.testing.assert_allclose(
            reversed_arrays["numerical"][::-1], batch["numerical"], rtol=0, atol=0
        )
        np.testing.assert_array_equal(
            reversed_arrays["categorical"][::-1], batch["categorical"]
        )

    def test_unfitted_transform_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            RegimeAugmentedPreprocessor().transform(make_frame())


if __name__ == "__main__":
    unittest.main()
