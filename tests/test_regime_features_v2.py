from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lg_control.constants import RAW_FEATURE_COLUMNS, TRAIN_COLUMNS
from lg_control.regime_features import (
    REGIME_FEATURE_COLUMNS,
    RegimeFeatureBuilderV2,
    RegimeFeatureError,
)
from lg_control.validation import build_validation_plan, fit_preprocessor_on_fold_train


def make_regime_frame() -> pd.DataFrame:
    """Create a valid official-schema frame with four rows per season/type."""

    records: list[dict[str, object]] = []
    position = 0
    for season in range(2019, 2025):
        for game_type in ("F", "R"):
            if game_type == "F" and season <= 2022:
                targets = (1, 1, 1, 0)
            elif game_type == "F":
                targets = (0, 0, 1, 0)
            else:
                targets = (1, 0, 1, 0)
            for within_group, target in enumerate(targets):
                count = 20 + within_group
                record: dict[str, object] = {
                    column: 0.0 for column in RAW_FEATURE_COLUMNS
                }
                record.update(
                    {
                        "row_id": f"ROW_{position:04d}",
                        "season": season,
                        "game_month": 4 + within_group,
                        "game_dayofweek": within_group,
                        "inning": 1 + within_group,
                        "top_bottom": "T" if within_group % 2 == 0 else "B",
                        "game_type": game_type,
                        "balls_before": within_group % 4,
                        "strikes_before": within_group % 3,
                        "outs_before": within_group % 3,
                        "run_top_before": 0,
                        "run_bot_before": 0,
                        "run_total_before": 0,
                        "score_diff_home": 0,
                        "score_diff_pitcher_team": 0,
                        "runner_on_1b": 0,
                        "runner_on_2b": 0,
                        "runner_on_3b": 0,
                        "num_runners_on": 0,
                        "base_state": "___",
                        "home_win_expectancy": 50.0,
                        "away_win_expectancy": 50.0,
                        "li": 1.0,
                        "pitcher_id": 100 + within_group,
                        "batter_id": 200 + within_group,
                        "pitcher_hand": 1 + within_group % 2,
                        "batter_hand": 2 - within_group % 2,
                        "pitcher_team_id": 10 + within_group % 2,
                        "batter_team_id": 12 + within_group % 2,
                        "asof_pitcher_n": count,
                        "asof_pitcher_success_rate": 0.60,
                        "asof_pitcher_reverse_rate": 0.20,
                        "asof_pitcher_middle_rate": 0.35,
                        "asof_pitcher_ball_rate": 0.25,
                        "asof_pitcher_strike_rate": 0.20,
                        "asof_pitcher_prev1_game_success_rate": 0.350877,
                        "asof_pitcher_prev3_game_success_rate": 0.52,
                        "asof_pitcher_prev5_game_success_rate": 0.50,
                        "asof_pitcher_prev1_game_middle_rate": 0.175439,
                        "asof_pitcher_prev3_game_middle_rate": 0.31,
                        "asof_pitcher_prev5_game_middle_rate": 0.30,
                        "asof_batter_n": count + 5,
                        "asof_batter_success_rate": 0.55,
                        "asof_batter_middle_rate": 0.30,
                        "asof_pitcher_pitchmix_n": count,
                        "asof_pitcher_fastball_rate": 0.50,
                        "asof_pitcher_breaking_rate": 0.30,
                        "asof_pitcher_offspeed_rate": 0.20,
                        "control_success": target,
                    }
                )
                records.append(record)
                position += 1
    frame = pd.DataFrame(records, columns=TRAIN_COLUMNS)
    frame.index = pd.Index(
        [f"source-{1000 + 11 * index}" for index in range(len(frame))],
        name="source_index",
    )
    return frame


class RegimeFeatureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_regime_frame()
        self.train = self.frame.loc[self.frame["season"].le(2022)].copy()
        self.validation = self.frame.loc[self.frame["season"].eq(2023)].copy()
        self.builder = RegimeFeatureBuilderV2().fit(self.train)

    def test_exact_output_contract_regime_crosses_and_beta_binomial(self) -> None:
        transformed = self.builder.transform(self.validation)
        self.assertEqual(tuple(transformed.columns), REGIME_FEATURE_COLUMNS)
        self.assertEqual(transformed.index.tolist(), self.validation.index.tolist())
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertNotIn("row_id", transformed.columns)
        self.assertNotIn("control_success", transformed.columns)

        f_position = int(np.flatnonzero(self.validation["game_type"].eq("F"))[0])
        row = transformed.iloc[f_position]
        self.assertEqual(row["regime_season_offset"], 0.0)
        self.assertEqual(row["regime_post_start"], 1.0)
        self.assertEqual(row["regime_game_type_f"], 1.0)
        self.assertEqual(row["regime_post_start_x_f"], 1.0)

        source = self.validation.iloc[f_position]
        count = float(source["asof_pitcher_n"])
        prior = float(row["asof_global_target_rate_bb"])
        strength = self.builder.history_prior_strength
        expected_mean = (count * 0.60 + strength * prior) / (count + strength)
        expected_reliability = count / (count + strength)
        self.assertAlmostEqual(row["pitcher_success_bb_mean"], expected_mean)
        self.assertAlmostEqual(
            row["pitcher_success_bb_reliability"], expected_reliability
        )

    def test_transform_is_row_local_order_invariant_and_target_free(self) -> None:
        batch = self.builder.transform(self.validation)
        reversed_frame = self.validation.iloc[::-1].copy()
        reversed_result = self.builder.transform(reversed_frame).loc[batch.index]
        np.testing.assert_allclose(batch.to_numpy(), reversed_result.to_numpy())

        for source_index in self.validation.index:
            singleton = self.builder.transform(self.validation.loc[[source_index]])
            np.testing.assert_allclose(
                batch.loc[[source_index]].to_numpy(), singleton.to_numpy()
            )

        changed_target = self.validation.copy()
        changed_target["control_success"] = 1 - changed_target["control_success"]
        np.testing.assert_allclose(
            batch.to_numpy(), self.builder.transform(changed_target).to_numpy()
        )

    def test_same_and_future_season_targets_cannot_change_asof_features(self) -> None:
        transform_rows = self.frame.loc[self.frame["season"].eq(2022)].copy()
        original = RegimeFeatureBuilderV2().fit(self.frame).transform(transform_rows)

        changed_future = self.frame.copy()
        future_mask = changed_future["season"].ge(2022)
        changed_future.loc[future_mask, "control_success"] = (
            1 - changed_future.loc[future_mask, "control_success"]
        )
        future_result = RegimeFeatureBuilderV2().fit(changed_future).transform(
            transform_rows
        )
        np.testing.assert_allclose(original.to_numpy(), future_result.to_numpy())

        changed_past = self.frame.copy()
        past_mask = changed_past["season"].eq(2021)
        changed_past.loc[past_mask, "control_success"] = (
            1 - changed_past.loc[past_mask, "control_success"]
        )
        past_result = RegimeFeatureBuilderV2().fit(changed_past).transform(transform_rows)
        self.assertFalse(
            np.allclose(
                original["asof_global_target_rate_bb"],
                past_result["asof_global_target_rate_bb"],
            )
        )

    def test_recent_rates_do_not_invent_binomial_counts_when_missing(self) -> None:
        frame = self.validation.iloc[[0]].copy()
        recent_columns = [
            column
            for family in ("success", "middle")
            for window in (1, 3, 5)
            for column in [
                f"asof_pitcher_prev{window}_game_{family}_rate"
            ]
        ]
        frame.loc[:, recent_columns] = np.nan
        transformed = self.builder.transform(frame).iloc[0]
        self.assertEqual(transformed["recent_success_observed_fraction"], 0.0)
        self.assertEqual(transformed["recent_middle_observed_fraction"], 0.0)
        self.assertEqual(transformed["recent_success_prev1_rational_fit"], 0.0)
        self.assertEqual(
            transformed["recent_success_prev1_rational_min_denominator"], 0.0
        )
        self.assertEqual(
            transformed["recent_success_weighted_minus_cumulative"], 0.0
        )
        self.assertEqual(
            transformed["recent_middle_weighted_minus_cumulative"], 0.0
        )

    def test_bounded_rational_reconstruction_extracts_only_a_lower_bound(self) -> None:
        transformed = self.builder.transform(self.validation.iloc[[0]]).iloc[0]
        # 0.350877 and 0.175439 reconstruct 20/57 and 10/57 within the
        # six-decimal rounding tolerance.  Their joint lower bound is 57.
        self.assertEqual(
            transformed["recent_success_prev1_rational_min_denominator"], 57.0
        )
        self.assertEqual(
            transformed["recent_middle_prev1_rational_min_denominator"], 57.0
        )
        self.assertEqual(
            transformed["recent_prev1_joint_rational_min_denominator"], 57.0
        )
        self.assertEqual(transformed["recent_success_prev1_rational_fit"], 1.0)
        self.assertAlmostEqual(
            transformed[
                "recent_success_prev1_rational_min_denominator_log1p"
            ],
            np.log1p(57.0),
        )
        self.assertAlmostEqual(
            transformed[
                "recent_success_prev1_rational_reliability_lower_bound"
            ],
            57.0 / (57.0 + self.builder.history_prior_strength),
        )

    def test_temporal_fold_helper_fits_no_anchor_statistics(self) -> None:
        fold = build_validation_plan(self.frame).anchor_fold
        fitted = fit_preprocessor_on_fold_train(
            RegimeFeatureBuilderV2, self.frame, fold
        )
        builder = fitted.preprocessor
        self.assertIsInstance(builder, RegimeFeatureBuilderV2)
        assert isinstance(builder, RegimeFeatureBuilderV2)
        self.assertEqual(builder.fit_seasons, (2019, 2020, 2021, 2022, 2023))
        self.assertNotIn(2024, builder.fit_seasons)
        transformed = fitted.transform_validation(self.frame)
        self.assertEqual(len(transformed), int(self.frame["season"].eq(2024).sum()))
        fitted.audit.assert_train_only()


class RegimeFeatureStateAndFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_regime_frame()

    def test_state_round_trip_and_tamper_rejection(self) -> None:
        builder = RegimeFeatureBuilderV2().fit(
            self.frame.loc[self.frame["season"].le(2023)]
        )
        probe = self.frame.loc[self.frame["season"].eq(2024)]
        expected = builder.transform(probe)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "regime_features.json"
            builder.save(path)
            restored = RegimeFeatureBuilderV2.load(path)
            np.testing.assert_allclose(
                expected.to_numpy(), restored.transform(probe).to_numpy()
            )

        state = builder.to_state_dict()
        tampered = copy.deepcopy(state)
        tampered["target_statistics"][0]["successes"] = (
            tampered["target_statistics"][0]["rows"] + 1
        )
        with self.assertRaisesRegex(RegimeFeatureError, "cannot exceed"):
            RegimeFeatureBuilderV2.from_state_dict(tampered)

    def test_requires_fit_labels_and_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(RegimeFeatureError, "requires labelled"):
            RegimeFeatureBuilderV2().fit(self.frame.drop(columns="control_success"))
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            RegimeFeatureBuilderV2().transform(self.frame)
        with self.assertRaisesRegex(RegimeFeatureError, "non-decreasing"):
            RegimeFeatureBuilderV2(recent_max_denominators=(160, 100, 800))


if __name__ == "__main__":
    unittest.main()
