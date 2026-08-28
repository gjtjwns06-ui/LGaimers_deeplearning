from __future__ import annotations

import json
import unittest

import numpy as np

from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.rate_basis_features import (
    CompactRateBasisBuilder,
    RATE_BASIS_FEATURE_COLUMNS,
    RateBasisAugmentedPreprocessor,
)
from tests.test_preprocessing import make_frame


class CompactRateBasisBuilderTests(unittest.TestCase):
    def test_feature_contract_and_finite_missing_handling(self) -> None:
        frame = make_frame()
        features = CompactRateBasisBuilder().transform(frame)
        self.assertEqual(features.shape, (len(frame), 60))
        self.assertEqual(tuple(features.columns), RATE_BASIS_FEATURE_COLUMNS)
        self.assertEqual(len(set(features.columns)), 60)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertFalse(any("target" in name for name in features.columns))
        self.assertFalse(any("row_id" in name for name in features.columns))
        self.assertFalse(any(name.endswith("_id") for name in features.columns))
        self.assertEqual(features.loc[0, "rb_pitcher_success_logit"], 0.0)
        self.assertEqual(features.loc[0, "rb_pitcher_success_uncertainty"], 0.0)

    def test_builder_is_row_order_batch_and_target_invariant(self) -> None:
        frame = make_frame()
        builder = CompactRateBasisBuilder()
        expected = builder.transform(frame)

        reversed_frame = frame.iloc[::-1].copy()
        reversed_frame["control_success"] = 1 - reversed_frame["control_success"]
        reversed_features = builder.transform(reversed_frame).iloc[::-1]
        reversed_features.index = expected.index
        np.testing.assert_allclose(reversed_features, expected, rtol=0, atol=0)

        for position in range(len(frame)):
            singleton = builder.transform(frame.iloc[[position]])
            np.testing.assert_allclose(
                singleton.to_numpy(), expected.iloc[[position]].to_numpy(), rtol=0, atol=0
            )


class RateBasisAugmentedPreprocessorTests(unittest.TestCase):
    def test_fit_transform_layout_and_json_round_trip_are_exact(self) -> None:
        frame = make_frame()
        preprocessor = RateBasisAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(frame)
        expected = preprocessor.transform(frame)
        layout = preprocessor.feature_layout()

        self.assertEqual(
            expected["numerical"].shape[1],
            len(DeepTabularPreprocessor.numerical_columns) + 60,
        )
        self.assertEqual(expected["numerical"].shape[1], len(layout.numerical_names))
        self.assertEqual(expected["categorical"].shape[1], len(layout.categorical_names))
        self.assertTrue(np.isfinite(expected["numerical"]).all())
        self.assertTrue(set(RATE_BASIS_FEATURE_COLUMNS).issubset(layout.numerical_names))
        self.assertIn("rb_pitcher_success_logit", layout.pitcher_numerical_names)
        self.assertIn("rb_batter_success_logit", layout.batter_numerical_names)
        self.assertNotIn("rb_compat_success_product", layout.pitcher_numerical_names)
        self.assertNotIn("rb_compat_success_product", layout.batter_numerical_names)

        payload = json.loads(json.dumps(preprocessor.to_state_dict()))
        restored = RateBasisAugmentedPreprocessor.from_state_dict(payload)
        actual = restored.transform(frame)
        np.testing.assert_array_equal(actual["categorical"], expected["categorical"])
        np.testing.assert_array_equal(actual["numerical"], expected["numerical"])
        np.testing.assert_array_equal(actual["row_id"], expected["row_id"])
        np.testing.assert_array_equal(actual["target"], expected["target"])
        self.assertEqual(restored.feature_layout(), layout)

    def test_fit_and_transform_ignore_target_and_transform_batch_context(self) -> None:
        frame = make_frame()
        changed = frame.copy()
        changed["control_success"] = 1 - changed["control_success"]
        original_preprocessor = RateBasisAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(frame)
        changed_preprocessor = RateBasisAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(changed)

        original = original_preprocessor.transform(frame)
        target_changed = changed_preprocessor.transform(changed)
        np.testing.assert_array_equal(original["categorical"], target_changed["categorical"])
        np.testing.assert_array_equal(original["numerical"], target_changed["numerical"])

        reversed_arrays = original_preprocessor.transform(changed.iloc[::-1])
        np.testing.assert_array_equal(
            reversed_arrays["categorical"][::-1], original["categorical"]
        )
        np.testing.assert_array_equal(
            reversed_arrays["numerical"][::-1], original["numerical"]
        )
        for position in range(len(frame)):
            singleton = original_preprocessor.transform(frame.iloc[[position]])
            np.testing.assert_array_equal(
                singleton["categorical"], original["categorical"][[position]]
            )
            np.testing.assert_array_equal(
                singleton["numerical"], original["numerical"][[position]]
            )

    def test_unfitted_and_corrupt_state_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            RateBasisAugmentedPreprocessor().transform(make_frame())

        fitted = RateBasisAugmentedPreprocessor(min_category_frequency=1).fit(make_frame())
        payload = fitted.to_state_dict()
        payload["feature_columns"] = payload["feature_columns"][:-1]
        with self.assertRaisesRegex(ValueError, "feature order"):
            RateBasisAugmentedPreprocessor.from_state_dict(payload)


if __name__ == "__main__":
    unittest.main()
