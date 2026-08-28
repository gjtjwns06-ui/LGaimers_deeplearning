from __future__ import annotations

import json
import unittest

import numpy as np

from lg_control.temporal_target_features import (
    TEMPORAL_TARGET_FEATURE_COLUMNS,
    PriorSeasonTargetEncoder,
    TemporalTargetAugmentedPreprocessor,
)
from tests.test_preprocessing import make_frame


class PriorSeasonTargetEncoderTests(unittest.TestCase):
    def test_transform_is_finite_row_local_and_uses_only_prior_seasons(self) -> None:
        frame = make_frame()
        encoder = PriorSeasonTargetEncoder().fit(frame)
        expected = encoder.transform(frame)
        self.assertEqual(tuple(expected.columns), TEMPORAL_TARGET_FEATURE_COLUMNS)
        self.assertTrue(np.isfinite(expected.to_numpy()).all())

        changed = frame.copy()
        changed.loc[changed["season"] >= 2020, "control_success"] = 1
        changed_encoder = PriorSeasonTargetEncoder().fit(changed)
        actual_2020 = changed_encoder.transform(frame.loc[frame["season"] == 2020])
        np.testing.assert_allclose(
            actual_2020.to_numpy(),
            expected.loc[frame["season"] == 2020].to_numpy(),
            rtol=0,
            atol=0,
        )

        reversed_frame = frame.iloc[::-1]
        reversed_values = encoder.transform(reversed_frame)
        np.testing.assert_allclose(
            reversed_values.iloc[::-1].to_numpy(), expected.to_numpy(), rtol=0, atol=0
        )

    def test_augmented_preprocessor_matches_layout(self) -> None:
        frame = make_frame()
        preprocessor = TemporalTargetAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(frame)
        arrays = preprocessor.transform(frame)
        layout = preprocessor.feature_layout()
        self.assertEqual(arrays["numerical"].shape[1], len(layout.numerical_names))
        self.assertEqual(arrays["categorical"].shape[1], len(layout.categorical_names))
        self.assertTrue(np.isfinite(arrays["numerical"]).all())

    def test_augmented_preprocessor_json_round_trip_is_exact(self) -> None:
        frame = make_frame()
        fitted = TemporalTargetAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(frame)
        expected = fitted.transform(frame)
        payload = json.loads(json.dumps(fitted.to_state_dict()))
        restored = TemporalTargetAugmentedPreprocessor.from_state_dict(payload)
        actual = restored.transform(frame)
        np.testing.assert_array_equal(actual["categorical"], expected["categorical"])
        np.testing.assert_array_equal(actual["numerical"], expected["numerical"])
        np.testing.assert_array_equal(actual["row_id"], expected["row_id"])
        np.testing.assert_array_equal(actual["target"], expected["target"])
        self.assertEqual(restored.feature_layout(), fitted.feature_layout())

    def test_unfitted_use_is_rejected(self) -> None:
        frame = make_frame()
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            PriorSeasonTargetEncoder().transform(frame)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            TemporalTargetAugmentedPreprocessor().transform(frame)


if __name__ == "__main__":
    unittest.main()
