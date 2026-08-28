from __future__ import annotations

import unittest

import numpy as np

from lg_control.strict_trackman import ROLE_FEATURE_COLUMNS, StrictTrackmanFeatureBuilder
from lg_control.trackman_pipeline import StrictTrackmanAugmentedPreprocessor
from tests.test_preprocessing import make_frame


class StrictTrackmanAugmentedPreprocessorTests(unittest.TestCase):
    def test_prefitted_builder_contract_and_layout(self) -> None:
        frame = make_frame()
        builder = StrictTrackmanFeatureBuilder()
        builder.target_seasons_ = tuple(sorted(int(value) for value in frame["season"].unique()))
        builder.role_lookups_ = {
            "pitcher": __import__("pandas").DataFrame(
                columns=["target_season", "main_id", *ROLE_FEATURE_COLUMNS["pitcher"]]
            ),
            "batter": __import__("pandas").DataFrame(
                columns=["target_season", "main_id", *ROLE_FEATURE_COLUMNS["batter"]]
            ),
        }
        builder.alignment_audit_ = {"strict_matched_games": 0}
        builder.is_fitted_ = True
        preprocessor = StrictTrackmanAugmentedPreprocessor(
            builder, min_category_frequency=1
        ).fit(frame)
        arrays = preprocessor.transform(frame)
        layout = preprocessor.feature_layout()
        self.assertEqual(arrays["numerical"].shape[1], len(layout.numerical_names))
        self.assertTrue(np.isfinite(arrays["numerical"]).all())
        self.assertTrue(set(ROLE_FEATURE_COLUMNS["pitcher"]).issubset(layout.pitcher_numerical_names))

    def test_unfitted_builder_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            StrictTrackmanAugmentedPreprocessor(StrictTrackmanFeatureBuilder())


if __name__ == "__main__":
    unittest.main()
