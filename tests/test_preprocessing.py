from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lg_control.constants import TRAIN_COLUMNS
from lg_control.preprocessing import (
    DROPPED_DETERMINISTIC_COLUMNS,
    MISSING_CATEGORY_INDEX,
    OOV_CATEGORY_INDEX,
    DeepTabularPreprocessor,
    validate_preprocessing_schema,
)
from lg_control.schema import SchemaError


def make_frame() -> pd.DataFrame:
    specifications = [
        # top/bottom, top runs, bottom runs, occupied bases, pitcher, batter, n
        ("T", 0, 0, (0, 0, 0), 101, 201, 0),
        ("B", 1, 2, (1, 0, 0), 101, 202, 7),
        ("T", 2, 1, (1, 0, 1), 102, 203, 12),
        ("B", 4, 4, (1, 1, 1), 103, 204, 20),
        ("T", 3, 1, (0, 1, 1), 999, 205, 30),
        ("B", 2, 3, (0, 0, 1), 104, 206, 40),
    ]
    base_states = {
        (0, 0, 0): "___",
        (1, 0, 0): "1__",
        (0, 1, 0): "_2_",
        (0, 0, 1): "__3",
        (1, 1, 0): "12_",
        (1, 0, 1): "1_3",
        (0, 1, 1): "_23",
        (1, 1, 1): "123",
    }
    rows: list[dict[str, object]] = []
    for index, (half, run_top, run_bot, bases, pitcher, batter, pitcher_n) in enumerate(
        specifications
    ):
        home_diff = run_bot - run_top
        pitcher_diff = home_diff if half == "T" else -home_diff
        batter_n = max(0, pitcher_n - 2)
        cold_pitcher = pitcher_n == 0
        cold_batter = batter_n == 0
        first, second, third = bases
        row = {
            "row_id": f"ROW_{index}",
            "season": 2019 + (index // 3),
            "game_month": 3 + index,
            "game_dayofweek": index % 7,
            "inning": [1, 5, 8, 10, 7, 4][index],
            "top_bottom": half,
            "game_type": "R" if index < 4 else "F",
            "balls_before": [0, 3, 1, 3, 2, 1][index],
            "strikes_before": [0, 2, 2, 1, 0, 1][index],
            "outs_before": index % 3,
            "run_top_before": run_top,
            "run_bot_before": run_bot,
            "run_total_before": run_top + run_bot,
            "score_diff_home": home_diff,
            "score_diff_pitcher_team": pitcher_diff,
            "runner_on_1b": first,
            "runner_on_2b": second,
            "runner_on_3b": third,
            "num_runners_on": first + second + third,
            "base_state": base_states[bases],
            "home_win_expectancy": 50.0 + home_diff * 4.0,
            "away_win_expectancy": 50.0 - home_diff * 4.0,
            "li": 0.5 + index / 3.0,
            "pitcher_id": pitcher,
            "batter_id": batter,
            "pitcher_hand": 1 + index % 2,
            "batter_hand": 1 + (index + 1) % 2,
            "pitcher_team_id": 10 + index % 3,
            "batter_team_id": np.nan if index == 5 else 20 + index % 3,
            "asof_pitcher_n": pitcher_n,
            "asof_pitcher_success_rate": np.nan if cold_pitcher else 0.48,
            "asof_pitcher_reverse_rate": np.nan if cold_pitcher else 0.22,
            "asof_pitcher_middle_rate": np.nan if cold_pitcher else 0.15,
            "asof_pitcher_ball_rate": np.nan if cold_pitcher else 0.36,
            "asof_pitcher_strike_rate": np.nan if cold_pitcher else 0.43,
            "asof_pitcher_prev1_game_success_rate": np.nan if cold_pitcher else 0.46,
            "asof_pitcher_prev3_game_success_rate": np.nan if cold_pitcher else 0.47,
            "asof_pitcher_prev5_game_success_rate": np.nan if cold_pitcher else 0.49,
            "asof_pitcher_prev1_game_middle_rate": np.nan if cold_pitcher else 0.18,
            "asof_pitcher_prev3_game_middle_rate": np.nan if cold_pitcher else 0.17,
            "asof_pitcher_prev5_game_middle_rate": np.nan if cold_pitcher else 0.16,
            "asof_batter_n": batter_n,
            "asof_batter_success_rate": np.nan if cold_batter else 0.51,
            "asof_batter_middle_rate": np.nan if cold_batter else 0.14,
            "asof_pitcher_pitchmix_n": pitcher_n,
            "asof_pitcher_fastball_rate": np.nan if cold_pitcher else 0.50,
            "asof_pitcher_breaking_rate": np.nan if cold_pitcher else 0.30,
            "asof_pitcher_offspeed_rate": np.nan if cold_pitcher else 0.20,
            "control_success": index % 2,
        }
        rows.append(row)

    frame = pd.DataFrame(rows, columns=TRAIN_COLUMNS)
    if tuple(frame.columns) != tuple(TRAIN_COLUMNS):
        raise AssertionError("synthetic preprocessing fixture does not match official schema")
    return frame


class SchemaAndCleaningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()

    def test_schema_and_deterministic_redundancy_policy(self) -> None:
        validate_preprocessing_schema(self.frame)
        original = self.frame.copy(deep=True)
        clean = DeepTabularPreprocessor().clean_frame(self.frame)

        for column in DROPPED_DETERMINISTIC_COLUMNS:
            self.assertNotIn(column, clean.columns)
        self.assertIn("row_id", clean.columns)
        self.assertIn("control_success", clean.columns)
        self.assertEqual(clean.loc[1, "count_state"], "B3_S2")
        self.assertEqual(clean.loc[2, "base_out_state"], "1_3_O2")
        self.assertEqual(clean.loc[3, "inning_bucket"], "extra")
        self.assertEqual(clean.loc[1, "score_state"], "trailing")
        self.assertAlmostEqual(clean.loc[2, "run_top_log1p"], np.log1p(2.0))
        pd.testing.assert_frame_equal(self.frame, original)

    def test_schema_rejects_missing_columns_and_inconsistent_redundancy(self) -> None:
        with self.subTest("missing official column"):
            with self.assertRaises(SchemaError):
                validate_preprocessing_schema(self.frame.drop(columns=["season"]))

        with self.subTest("run total mismatch"):
            invalid = self.frame.copy()
            invalid.loc[0, "run_total_before"] = 1
            with self.assertRaisesRegex(SchemaError, "run_total_before"):
                validate_preprocessing_schema(invalid)

        with self.subTest("base-state mismatch"):
            invalid = self.frame.copy()
            invalid.loc[1, "base_state"] = "___"
            with self.assertRaisesRegex(SchemaError, "base_state"):
                validate_preprocessing_schema(invalid)

    def test_missing_masks_and_count_log_transforms_are_explicit(self) -> None:
        clean = DeepTabularPreprocessor().clean_frame(self.frame)
        self.assertEqual(clean.loc[0, "asof_pitcher_success_rate__missing"], 1.0)
        self.assertTrue(np.isnan(clean.loc[0, "asof_pitcher_success_rate"]))
        self.assertEqual(clean.loc[1, "asof_pitcher_success_rate__missing"], 0.0)
        self.assertAlmostEqual(clean.loc[2, "pitcher_n_log1p"], np.log1p(12.0))
        self.assertAlmostEqual(clean.loc[2, "batter_n_log1p"], np.log1p(10.0))


class FitTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()
        self.fit_frame = self.frame.iloc[:4].copy()

    def test_outputs_have_stable_types_shapes_and_finite_values(self) -> None:
        preprocessor = DeepTabularPreprocessor().fit(self.fit_frame)
        transformed = preprocessor.transform(self.frame)

        self.assertEqual(
            transformed["categorical"].shape,
            (len(self.frame), len(preprocessor.categorical_columns)),
        )
        self.assertEqual(
            transformed["numerical"].shape,
            (len(self.frame), len(preprocessor.numerical_columns)),
        )
        self.assertEqual(transformed["categorical"].dtype, np.int64)
        self.assertEqual(transformed["numerical"].dtype, np.float32)
        self.assertTrue(np.isfinite(transformed["numerical"]).all())
        np.testing.assert_array_equal(transformed["row_id"], self.frame["row_id"])
        np.testing.assert_array_equal(transformed["target"], self.frame["control_success"])

    def test_missing_and_unseen_categories_have_distinct_reserved_indices(self) -> None:
        preprocessor = DeepTabularPreprocessor().fit(self.fit_frame)
        transformed = preprocessor.transform(self.frame)
        pitcher_index = preprocessor.categorical_columns.index("pitcher_id")
        team_index = preprocessor.categorical_columns.index("batter_team_id")

        self.assertEqual(
            transformed["categorical"][4, pitcher_index], OOV_CATEGORY_INDEX
        )
        self.assertEqual(
            transformed["categorical"][5, team_index], MISSING_CATEGORY_INDEX
        )

        canonical = self.fit_frame.iloc[[0]].copy()
        canonical["pitcher_id"] = "101.0"
        encoded = preprocessor.transform(canonical)["categorical"]
        self.assertEqual(
            encoded[0, pitcher_index],
            transformed["categorical"][0, pitcher_index],
        )

    def test_fit_is_target_independent_and_transform_is_batch_independent(self) -> None:
        first = DeepTabularPreprocessor().fit(self.fit_frame)
        target_changed = self.fit_frame.copy()
        target_changed["control_success"] = 1 - target_changed["control_success"]
        second = DeepTabularPreprocessor().fit(target_changed)

        self.assertEqual(first.category_vocabs_, second.category_vocabs_)
        self.assertEqual(first.numeric_imputation_, second.numeric_imputation_)
        self.assertEqual(first.numeric_means_, second.numeric_means_)
        self.assertEqual(first.numeric_scales_, second.numeric_scales_)

        batch = first.transform(self.frame)
        for row_position in (0, 2, 5):
            single = first.transform(self.frame.iloc[[row_position]])
            np.testing.assert_array_equal(
                single["categorical"][0], batch["categorical"][row_position]
            )
            np.testing.assert_allclose(
                single["numerical"][0],
                batch["numerical"][row_position],
                rtol=0,
                atol=0,
            )

    def test_transform_does_not_require_or_consume_target(self) -> None:
        preprocessor = DeepTabularPreprocessor().fit(self.fit_frame)
        with_target = preprocessor.transform(self.frame)
        without_target = preprocessor.transform(self.frame.drop(columns=["control_success"]))

        self.assertNotIn("target", without_target)
        np.testing.assert_array_equal(
            with_target["categorical"], without_target["categorical"]
        )
        np.testing.assert_array_equal(
            with_target["numerical"], without_target["numerical"]
        )

    def test_transform_before_fit_fails_loudly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            DeepTabularPreprocessor().transform(self.frame)

    def test_deep_preprocessor_save_load_round_trip_and_tamper_check(self) -> None:
        fitted = DeepTabularPreprocessor().fit(self.fit_frame)
        expected = fitted.transform(self.frame)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "preprocessor.json"
            fitted.save(path)
            restored = DeepTabularPreprocessor.load(path)
            actual = restored.transform(self.frame)
            np.testing.assert_array_equal(
                expected["categorical"], actual["categorical"]
            )
            np.testing.assert_array_equal(expected["numerical"], actual["numerical"])

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["category_cardinalities"][0] += 1
            with self.assertRaisesRegex(ValueError, "cardinality"):
                DeepTabularPreprocessor.from_state_dict(payload)

    def test_serialized_state_is_a_snapshot_and_rejects_lossy_integers(self) -> None:
        fitted = DeepTabularPreprocessor().fit(self.fit_frame)
        snapshot = fitted.to_state_dict()
        first_column = fitted.categorical_columns[0]
        original_vocab = dict(fitted.category_vocabs_[first_column])
        snapshot["category_vocabs"][first_column].clear()
        snapshot["category_cardinalities"][0] = 999
        self.assertEqual(fitted.category_vocabs_[first_column], original_vocab)
        self.assertNotEqual(fitted.category_cardinalities_[0], 999)

        invalid = fitted.to_state_dict()
        token = next(iter(invalid["category_vocabs"][first_column]))
        invalid["category_vocabs"][first_column][token] = 2.5
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            DeepTabularPreprocessor.from_state_dict(invalid)


if __name__ == "__main__":
    unittest.main()
