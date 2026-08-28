from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from lg_control.auxiliary_targets import FOLD_TRAIN_PARTITION
from lg_control.preprocessing import NUMERICAL_FEATURE_COLUMNS
from lg_control.targeted_prior_features import (
    TARGETED_LOOKUP_FEATURE_COLUMNS,
    TARGETED_LOOKUP_SPECS,
    TARGETED_PRIOR_FEATURE_COLUMNS,
    TARGETED_ROW_LOCAL_FEATURE_COLUMNS,
    TargetedPriorAugmentedPreprocessor,
    TargetedPriorSeasonEncoder,
)
from tests.test_preprocessing import make_frame


def make_auxiliary_history() -> pd.DataFrame:
    frame = make_frame().iloc[:4].copy().reset_index(drop=True)
    frame["row_id"] = [f"ROW_{index:03d}" for index in range(4)]
    frame["season"] = 2023
    frame["game_type"] = "F"
    frame["pitcher_id"] = 101
    frame["batter_id"] = 201
    frame["pitcher_team_id"] = 10
    frame["batter_team_id"] = 20
    frame["pitcher_hand"] = 1
    frame["batter_hand"] = 2
    frame["asof_pitcher_n"] = [0, 1, 2, 3]
    frame["asof_pitcher_pitchmix_n"] = [0, 1, 2, 3]
    frame["asof_batter_n"] = [0, 1, 2, 3]
    frame["control_success"] = [1, 0, 0, 1]

    frame["asof_pitcher_success_rate"] = [np.nan, 1.0, 0.5, 1.0 / 3.0]
    frame["asof_pitcher_reverse_rate"] = [np.nan, 0.0, 0.5, 1.0 / 3.0]
    frame["asof_pitcher_middle_rate"] = [np.nan, 0.0, 0.0, 1.0 / 3.0]
    frame["asof_pitcher_ball_rate"] = [np.nan, 0.0, 0.5, 1.0 / 3.0]
    frame["asof_pitcher_strike_rate"] = [np.nan, 1.0, 0.5, 1.0 / 3.0]
    frame["asof_batter_success_rate"] = [np.nan, 1.0, 0.5, 1.0 / 3.0]
    frame["asof_batter_middle_rate"] = [np.nan, 0.0, 0.0, 1.0 / 3.0]
    return frame


class TargetedPriorSeasonEncoderTests(unittest.TestCase):
    def test_feature_budget_names_and_fixed_strengths(self) -> None:
        self.assertEqual(len(TARGETED_LOOKUP_FEATURE_COLUMNS), 21)
        self.assertEqual(len(TARGETED_ROW_LOCAL_FEATURE_COLUMNS), 9)
        self.assertEqual(len(TARGETED_PRIOR_FEATURE_COLUMNS), 30)
        self.assertEqual(len(set(TARGETED_PRIOR_FEATURE_COLUMNS)), 30)
        self.assertTrue(all("prev1" in name for name in TARGETED_LOOKUP_FEATURE_COLUMNS))
        self.assertFalse(any("prev2" in name for name in TARGETED_PRIOR_FEATURE_COLUMNS))
        for spec in TARGETED_LOOKUP_SPECS:
            self.assertEqual(spec.strength, 100.0 if spec.label == "control_success" else 50.0)

    def test_fit_scope_is_explicit_and_auxiliary_lookup_is_reconstructed(self) -> None:
        frame = make_auxiliary_history()
        with self.assertRaisesRegex(ValueError, "fold_train"):
            TargetedPriorSeasonEncoder().fit(frame, partition="validation")

        encoder = TargetedPriorSeasonEncoder().fit(
            frame, partition=FOLD_TRAIN_PARTITION
        )
        query = frame.iloc[[1]].copy()
        query["season"] = 2024
        features = encoder.transform(query)
        reverse_name = "aux_reverse_pitcher_batter_hand_prev1_rate"
        expected = (1.0 + 50.0 * (1.0 / 3.0)) / 53.0
        # Three reconstructed pitcher rows share this pitcher/hand key.
        self.assertAlmostEqual(features.iloc[0][reverse_name], expected)
        self.assertAlmostEqual(
            features.iloc[0]["aux_reverse_pitcher_batter_hand_prev1_reliability"],
            3.0 / 53.0,
        )

    def test_future_and_transform_target_mutation_cannot_change_prior_features(self) -> None:
        history_2023 = make_auxiliary_history()
        history_2024 = make_auxiliary_history()
        history_2024["season"] = 2024
        history_2024["row_id"] = [f"ROW_{index + 100:03d}" for index in range(4)]
        combined = pd.concat([history_2023, history_2024], ignore_index=True)

        first = TargetedPriorSeasonEncoder().fit(
            combined, partition=FOLD_TRAIN_PARTITION
        )
        changed = combined.copy()
        # This is terminal for both entities, so auxiliary reconstruction
        # remains valid while the future-season target statistic changes.
        changed.loc[changed["row_id"] == "ROW_103", "control_success"] = 0
        second = TargetedPriorSeasonEncoder().fit(
            changed, partition=FOLD_TRAIN_PARTITION
        )
        query = history_2023.iloc[[1]].copy()
        query["season"] = 2024
        expected = first.transform(query)
        actual = second.transform(query)
        np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())

        query_changed = query.copy()
        query_changed["control_success"] = 1 - query_changed["control_success"]
        np.testing.assert_array_equal(
            first.transform(query_changed).to_numpy(), expected.to_numpy()
        )

    def test_fit_order_and_transform_batch_order_are_invariant(self) -> None:
        frame = make_auxiliary_history()
        ordered = TargetedPriorSeasonEncoder().fit(
            frame, partition=FOLD_TRAIN_PARTITION
        )
        shuffled_frame = frame.sample(frac=1.0, random_state=7)
        shuffled = TargetedPriorSeasonEncoder().fit(
            shuffled_frame, partition=FOLD_TRAIN_PARTITION
        )
        self.assertEqual(ordered.to_state_dict(), shuffled.to_state_dict())

        query = frame.copy()
        query["season"] = 2024
        expected = ordered.transform(query)
        reversed_values = ordered.transform(query.iloc[::-1])
        np.testing.assert_array_equal(
            reversed_values.iloc[::-1].to_numpy(), expected.to_numpy()
        )
        for position in range(len(query)):
            np.testing.assert_array_equal(
                ordered.transform(query.iloc[[position]]).iloc[0].to_numpy(),
                expected.iloc[position].to_numpy(),
            )

    def test_singleton_auxiliary_history_falls_back_to_game_type_neutral(self) -> None:
        singleton = make_auxiliary_history().iloc[[0]].copy()
        encoder = TargetedPriorSeasonEncoder().fit(
            singleton, partition=FOLD_TRAIN_PARTITION
        )
        query = singleton.copy()
        query["season"] = 2024
        values = encoder.transform(query)
        self.assertTrue(np.isfinite(values.to_numpy()).all())
        self.assertAlmostEqual(
            values.iloc[0]["aux_reverse_pitcher_batter_hand_prev1_rate"], 0.5
        )
        self.assertEqual(
            values.iloc[0]["aux_reverse_pitcher_batter_hand_prev1_reliability"], 0.0
        )


class TargetedPriorAugmentedPreprocessorTests(unittest.TestCase):
    def test_layout_and_json_round_trip_are_complete_and_exact(self) -> None:
        frame = make_auxiliary_history()
        fitted = TargetedPriorAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(frame, partition=FOLD_TRAIN_PARTITION)
        arrays = fitted.transform(frame)
        layout = fitted.feature_layout()
        self.assertEqual(
            arrays["numerical"].shape[1],
            len(NUMERICAL_FEATURE_COLUMNS) + len(TARGETED_PRIOR_FEATURE_COLUMNS),
        )
        self.assertEqual(tuple(layout.numerical_names), fitted.numerical_columns)
        self.assertTrue(np.isfinite(arrays["numerical"]).all())

        payload = json.loads(json.dumps(fitted.to_state_dict()))
        restored = TargetedPriorAugmentedPreprocessor.from_state_dict(payload)
        actual = restored.transform(frame)
        for key in ("categorical", "numerical", "row_id", "target"):
            np.testing.assert_array_equal(actual[key], arrays[key])
        self.assertEqual(restored.feature_layout(), layout)

    def test_preprocessor_transform_is_row_independent(self) -> None:
        frame = make_auxiliary_history()
        fitted = TargetedPriorAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(frame, partition=FOLD_TRAIN_PARTITION)
        query = frame.copy()
        query["season"] = 2024
        batch = fitted.transform(query)
        for position in range(len(query)):
            single = fitted.transform(query.iloc[[position]])
            np.testing.assert_array_equal(
                single["categorical"][0], batch["categorical"][position]
            )
            np.testing.assert_array_equal(
                single["numerical"][0], batch["numerical"][position]
            )


if __name__ == "__main__":
    unittest.main()
