"""Contract tests for the row-independent tabular modelling pipeline.

These tests intentionally import the production modules directly.  A missing
module is an implementation failure, not a reason to skip the contract.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.lg_control.features import FeaturePreprocessor
from src.lg_control.inference import predict_probabilities
from src.lg_control.model import TabularResidualModel


TRAIN_COLUMNS = (
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "control_success",
)


def make_synthetic_frame() -> pd.DataFrame:
    """Return six rows containing the complete public train.csv schema."""

    rows = []
    categorical_values = (
        ("T", "R", "___", 1001, 2001, 1, 2, 11, 21),
        ("B", "R", "1__", 1002, 2002, 2, 1, 12, 22),
        ("T", "F", "_2_", 1001, 2003, 1, 1, 11, 23),
        ("B", "F", "123", 1003, 2001, 2, 2, 13, 21),
        # The last two rows exercise OOV handling after fitting rows 0..3.
        ("T", "R", "12_", 9999, 2999, 1, 2, 99, 29),
        ("B", "F", "__3", 1002, 2002, 2, 1, 12, 22),
    )

    for i, values in enumerate(categorical_values):
        (
            top_bottom,
            game_type,
            base_state,
            pitcher_id,
            batter_id,
            pitcher_hand,
            batter_hand,
            pitcher_team_id,
            batter_team_id,
        ) = values
        runner_1b = int("1" in base_state[:1])
        runner_2b = int("2" in base_state[1:2])
        runner_3b = int("3" in base_state[2:3])
        num_runners = runner_1b + runner_2b + runner_3b
        pitcher_n = (0, 12, 100, 450, 8, 900)[i]
        batter_n = (0, 20, 80, 300, 6, 700)[i]
        rate_shift = 0.01 * i
        cold_start = i == 0

        row = {
            "row_id": f"TRAIN_{i + 1:07d}",
            "season": 2019 + i,
            "game_month": 3 + i,
            "game_dayofweek": i % 7,
            "inning": 1 + i,
            "top_bottom": top_bottom,
            "game_type": game_type,
            "balls_before": i % 4,
            "strikes_before": i % 3,
            "outs_before": i % 3,
            "run_top_before": i,
            "run_bot_before": max(0, i - 1),
            "run_total_before": i + max(0, i - 1),
            "score_diff_home": max(0, i - 1) - i,
            "score_diff_pitcher_team": i - max(0, i - 1),
            "runner_on_1b": runner_1b,
            "runner_on_2b": runner_2b,
            "runner_on_3b": runner_3b,
            "num_runners_on": num_runners,
            "base_state": base_state,
            "home_win_expectancy": 45.0 + i,
            "away_win_expectancy": 55.0 - i,
            "li": 0.5 + 0.2 * i,
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "pitcher_hand": pitcher_hand,
            "batter_hand": batter_hand,
            "pitcher_team_id": pitcher_team_id,
            "batter_team_id": batter_team_id,
            "asof_pitcher_n": pitcher_n,
            "asof_pitcher_success_rate": np.nan if cold_start else 0.48 + rate_shift,
            "asof_pitcher_reverse_rate": np.nan if cold_start else 0.23 - rate_shift,
            "asof_pitcher_middle_rate": np.nan if cold_start else 0.14 + rate_shift,
            "asof_pitcher_ball_rate": np.nan if cold_start else 0.37 - rate_shift,
            "asof_pitcher_strike_rate": np.nan if cold_start else 0.42 + rate_shift,
            "asof_pitcher_prev1_game_success_rate": np.nan if cold_start else 0.46 + rate_shift,
            "asof_pitcher_prev3_game_success_rate": np.nan if cold_start else 0.47 + rate_shift,
            "asof_pitcher_prev5_game_success_rate": np.nan if cold_start else 0.49 + rate_shift,
            "asof_pitcher_prev1_game_middle_rate": np.nan if cold_start else 0.18 - rate_shift,
            "asof_pitcher_prev3_game_middle_rate": np.nan if cold_start else 0.17 - rate_shift,
            "asof_pitcher_prev5_game_middle_rate": np.nan if cold_start else 0.16 - rate_shift,
            "asof_batter_n": batter_n,
            "asof_batter_success_rate": np.nan if cold_start else 0.51 - rate_shift,
            "asof_batter_middle_rate": np.nan if cold_start else 0.15 + rate_shift,
            "asof_pitcher_pitchmix_n": pitcher_n,
            "asof_pitcher_fastball_rate": np.nan if cold_start else 0.50 - rate_shift,
            "asof_pitcher_breaking_rate": np.nan if cold_start else 0.35 + rate_shift,
            "asof_pitcher_offspeed_rate": np.nan if cold_start else 0.15,
            "control_success": i % 2,
        }
        rows.append(row)

    frame = pd.DataFrame(rows, columns=TRAIN_COLUMNS)
    if tuple(frame.columns) != TRAIN_COLUMNS:
        raise AssertionError("synthetic fixture drifted from the train schema")
    return frame


class FeaturePreprocessorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_synthetic_frame()
        self.fit_frame = self.frame.iloc[:4].copy()

    def test_identifiers_and_target_never_become_features(self) -> None:
        preprocessor = FeaturePreprocessor().fit(self.fit_frame)

        public_feature_metadata = (
            list(preprocessor.categorical_columns)
            + list(preprocessor.numerical_columns)
            + list(preprocessor.feature_names)
        )
        self.assertNotIn("row_id", public_feature_metadata)
        self.assertNotIn("control_success", public_feature_metadata)

        original = preprocessor.transform(self.fit_frame)
        changed = self.fit_frame.copy()
        changed["row_id"] = [f"REPLACED_{i}" for i in range(len(changed))]
        changed["control_success"] = 1 - changed["control_success"]
        transformed_changed = preprocessor.transform(changed)

        np.testing.assert_array_equal(
            original["categorical"], transformed_changed["categorical"]
        )
        np.testing.assert_allclose(
            original["numerical"], transformed_changed["numerical"], rtol=0, atol=0
        )

    def test_unseen_categories_use_reserved_oov_index(self) -> None:
        preprocessor = FeaturePreprocessor().fit(self.fit_frame)
        known = preprocessor.transform(self.fit_frame.iloc[[0]])["categorical"]
        unseen = preprocessor.transform(self.frame.iloc[[4]])["categorical"]
        pitcher_column = list(preprocessor.categorical_columns).index("pitcher_id")

        # 0 is reserved for missing values and 1 for a non-missing OOV token.
        self.assertGreater(int(known[0, pitcher_column]), 1)
        self.assertEqual(int(unseen[0, pitcher_column]), 1)

    def test_save_load_round_trip_is_exact(self) -> None:
        preprocessor = FeaturePreprocessor().fit(self.fit_frame)
        expected = preprocessor.transform(self.frame)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "preprocessor.json"
            preprocessor.save(path)
            self.assertTrue(path.is_file())
            restored = FeaturePreprocessor.load(path)
            actual = restored.transform(self.frame)

        np.testing.assert_array_equal(expected["categorical"], actual["categorical"])
        np.testing.assert_array_equal(expected["numerical"], actual["numerical"])
        self.assertEqual(
            list(preprocessor.category_cardinalities),
            list(restored.category_cardinalities),
        )

    def test_missing_required_column_and_duplicate_row_id_fail(self) -> None:
        preprocessor = FeaturePreprocessor().fit(self.fit_frame)

        with self.subTest("required input column"):
            missing = self.frame.drop(columns=["season"])
            with self.assertRaises((KeyError, ValueError)):
                preprocessor.transform(missing)

        with self.subTest("duplicate row_id during transform"):
            duplicate = self.frame.copy()
            duplicate.loc[1, "row_id"] = duplicate.loc[0, "row_id"]
            with self.assertRaises(ValueError):
                preprocessor.transform(duplicate)

        with self.subTest("duplicate row_id during fit"):
            duplicate = self.fit_frame.copy()
            duplicate.loc[1, "row_id"] = duplicate.loc[0, "row_id"]
            with self.assertRaises(ValueError):
                FeaturePreprocessor().fit(duplicate)


class ModelAndInferenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.frame = make_synthetic_frame()
        self.preprocessor = FeaturePreprocessor().fit(self.frame.iloc[:4].copy())
        transformed = self.preprocessor.transform(self.frame.iloc[:1])
        self.model = TabularResidualModel(
            category_cardinalities=self.preprocessor.category_cardinalities,
            num_numerical=transformed["numerical"].shape[1],
            hidden_dim=16,
            num_blocks=2,
            dropout=0.0,
        )
        self.model.eval()

    def test_model_forward_returns_one_logit_per_row(self) -> None:
        transformed = self.preprocessor.transform(self.frame)
        categorical = torch.as_tensor(transformed["categorical"], dtype=torch.long)
        numerical = torch.as_tensor(transformed["numerical"], dtype=torch.float32)

        with torch.inference_mode():
            logits = self.model(categorical, numerical)

        self.assertEqual(tuple(logits.shape), (len(self.frame),))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_predictions_are_probabilities_and_row_independent(self) -> None:
        inputs = self.frame.drop(columns=["control_success"])
        batched = predict_probabilities(
            self.model, self.preprocessor, inputs, batch_size=3
        )
        single = np.concatenate(
            [
                predict_probabilities(
                    self.model, self.preprocessor, inputs.iloc[[i]], batch_size=1
                )
                for i in range(len(inputs))
            ]
        )

        permutation = np.array([4, 0, 5, 2, 1, 3])
        shuffled = predict_probabilities(
            self.model,
            self.preprocessor,
            inputs.iloc[permutation].reset_index(drop=True),
            batch_size=2,
        )
        shuffled_back = np.empty_like(shuffled)
        shuffled_back[permutation] = shuffled

        self.assertEqual(batched.shape, (len(inputs),))
        self.assertTrue(np.isfinite(batched).all())
        self.assertTrue(((batched >= 0.0) & (batched <= 1.0)).all())
        np.testing.assert_allclose(batched, single, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(batched, shuffled_back, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
