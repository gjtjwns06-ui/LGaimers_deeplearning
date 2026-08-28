from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from lg_control.constants import RAW_FEATURE_COLUMNS, TRAIN_COLUMNS
from lg_control.validation import (
    ValidationDesignError,
    assert_selection_scope,
    assert_train_only_fit_scope,
    build_development_plan,
    build_season_fold,
    build_validation_plan,
    fit_preprocessor_on_fold_train,
    fit_preprocessor_on_inner_train,
)


def make_validation_frame() -> pd.DataFrame:
    """Small official-schema frame whose seasons are deliberately interleaved."""

    season_order = [2024, 2019, 2022, 2020, 2023, 2021] * 2
    row_count = len(season_order)
    data: dict[str, object] = {
        column: np.zeros(row_count, dtype=np.float64)
        for column in RAW_FEATURE_COLUMNS
    }
    data.update(
        {
            "row_id": [f"ROW_{position:03d}" for position in range(row_count)],
            "season": season_order,
            "top_bottom": ["T", "B"] * (row_count // 2),
            "game_type": ["R", "F"] * (row_count // 2),
            "base_state": ["___"] * row_count,
            "pitcher_id": [100 + position % 4 for position in range(row_count)],
            "batter_id": [200 + position % 5 for position in range(row_count)],
            "pitcher_hand": [1, 2] * (row_count // 2),
            "batter_hand": [2, 1] * (row_count // 2),
            "pitcher_team_id": [10 + position % 3 for position in range(row_count)],
            "batter_team_id": [20 + position % 3 for position in range(row_count)],
            "control_success": [position % 2 for position in range(row_count)],
        }
    )
    frame = pd.DataFrame(data, columns=TRAIN_COLUMNS)
    frame.index = pd.Index(
        [f"source-index-{100 + 7 * position}" for position in range(row_count)],
        name="source_index",
    )
    if tuple(frame.columns) != tuple(TRAIN_COLUMNS):
        raise AssertionError("validation fixture drifted from the official train schema")
    return frame


class RecordingPreprocessor:
    def __init__(self) -> None:
        self.fit_ids: list[object] | None = None
        self.fit_seasons: list[int] | None = None
        self.fit_index: list[object] | None = None

    def fit(self, frame: pd.DataFrame) -> "RecordingPreprocessor":
        self.fit_ids = frame["row_id"].tolist()
        self.fit_seasons = sorted(int(value) for value in frame["season"].unique())
        self.fit_index = frame.index.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "row_id": frame["row_id"].to_numpy(copy=True),
            "season": frame["season"].to_numpy(dtype=np.int64, copy=True),
        }


class CanonicalValidationPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_validation_frame()

    def test_primary_rolling_and_recent_only_partitions(self) -> None:
        plan = build_validation_plan(self.frame, include_recent_only=True)

        self.assertEqual(
            [fold.validation_season for fold in plan.rolling_folds],
            [2022, 2023, 2024],
        )
        self.assertEqual(
            [fold.train_seasons for fold in plan.rolling_folds],
            [
                (2019, 2020, 2021),
                (2019, 2020, 2021, 2022),
                (2019, 2020, 2021, 2022, 2023),
            ],
        )
        self.assertIs(plan.primary_holdout, plan.rolling_folds[-1])
        self.assertIs(plan.anchor_fold, plan.primary_holdout)
        self.assertEqual(
            [fold.validation_season for fold in plan.development_folds],
            [2022, 2023],
        )
        self.assertNotIn(plan.anchor_fold, plan.development_folds)
        self.assertEqual(plan.primary_holdout.role, "anchor")
        self.assertEqual(len(plan.execution_folds()), 4)
        self.assertEqual(len(plan.execution_folds(include_diagnostic=False)), 3)

        diagnostic = plan.recent_only_diagnostic
        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(diagnostic.train_seasons, (2023,))
        self.assertEqual(diagnostic.validation_season, 2024)
        self.assertEqual(diagnostic.role, "diagnostic")
        self.assertIsNone(diagnostic.inner_stop_split)

        # The summary is metadata only and remains directly serialisable.
        json.dumps(plan.to_summary())

    def test_every_outer_fold_is_strictly_forward_and_preserves_rows(self) -> None:
        plan = build_validation_plan(self.frame, include_recent_only=True)
        for fold in plan.execution_folds():
            with self.subTest(fold=fold.name):
                train, validation = fold.split(self.frame)
                self.assertLess(int(train["season"].max()), int(validation["season"].min()))
                np.testing.assert_array_equal(
                    train["row_id"].to_numpy(), fold.train_row_ids
                )
                np.testing.assert_array_equal(
                    validation["row_id"].to_numpy(), fold.validation_row_ids
                )
                self.assertEqual(
                    train.index.tolist(),
                    self.frame.iloc[fold.train_positions].index.tolist(),
                )
                self.assertEqual(
                    validation.index.tolist(),
                    self.frame.iloc[fold.validation_positions].index.tolist(),
                )

    def test_nested_stop_split_is_inside_outer_train(self) -> None:
        plan = build_validation_plan(self.frame)
        expected = [
            ((2019, 2020), 2021),
            ((2019, 2020, 2021), 2022),
            ((2019, 2020, 2021, 2022), 2023),
        ]
        for fold, (inner_train_seasons, inner_validation_season) in zip(
            plan.rolling_folds, expected, strict=True
        ):
            with self.subTest(fold=fold.name):
                self.assertEqual(fold.inner_train_seasons, inner_train_seasons)
                self.assertEqual(
                    fold.inner_validation_season, inner_validation_season
                )
                inner_train, inner_stop = fold.inner_selection_frames(self.frame)
                self.assertEqual(
                    tuple(sorted(inner_train["season"].unique())), inner_train_seasons
                )
                self.assertEqual(set(inner_stop["season"]), {inner_validation_season})
                self.assertTrue(
                    set(inner_train["row_id"])
                    .union(inner_stop["row_id"])
                    .isdisjoint(fold.validation_row_ids.tolist())
                )
                self.assertEqual(
                    set(inner_train["row_id"]).union(inner_stop["row_id"]),
                    set(fold.train_row_ids),
                )

    def test_outer_validation_cannot_enter_selection_scope(self) -> None:
        fold = build_validation_plan(self.frame).primary_holdout
        inner = fold.inner_stop_split
        assert inner is not None
        safe_positions = np.concatenate(
            [inner.train_positions, inner.validation_positions]
        )
        assert_selection_scope(fold, safe_positions)

        with self.assertRaisesRegex(
            ValidationDesignError, "cannot select epochs/checkpoints"
        ):
            assert_selection_scope(
                fold,
                np.concatenate([safe_positions, fold.validation_positions[:1]]),
            )

    def test_plan_is_bound_to_original_row_identity_and_order(self) -> None:
        fold = build_validation_plan(self.frame).primary_holdout
        reordered = self.frame.iloc[::-1].copy()
        with self.assertRaisesRegex(ValidationDesignError, "order/identity changed"):
            fold.split(reordered)


class InvalidDataAndFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_validation_frame()

    def test_schema_duplicate_id_and_target_are_checked(self) -> None:
        cases: list[tuple[str, pd.DataFrame, str]] = []
        cases.append(("missing column", self.frame.drop(columns=["inning"]), "missing"))

        duplicate_id = self.frame.copy()
        duplicate_id.iloc[1, duplicate_id.columns.get_loc("row_id")] = duplicate_id.iloc[0][
            "row_id"
        ]
        cases.append(("duplicate ID", duplicate_id, "unique"))

        invalid_target = self.frame.copy()
        invalid_target.iloc[0, invalid_target.columns.get_loc("control_success")] = 2
        cases.append(("invalid target", invalid_target, "binary 0/1"))

        duplicate_column = pd.concat([self.frame, self.frame[["season"]]], axis=1)
        cases.append(("duplicate column", duplicate_column, "duplicate dataframe columns"))

        for name, invalid, message in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValidationDesignError, message):
                    build_validation_plan(invalid)

    def test_season_domain_and_complete_coverage_are_checked(self) -> None:
        fractional = self.frame.copy()
        fractional["season"] = fractional["season"].astype(float)
        fractional.iloc[0, fractional.columns.get_loc("season")] = 2024.5
        with self.assertRaisesRegex(ValidationDesignError, "finite integer years"):
            build_validation_plan(fractional)

        missing_2021 = self.frame.loc[self.frame["season"] != 2021].copy()
        with self.assertRaisesRegex(ValidationDesignError, r"missing=\[2021\]"):
            build_validation_plan(missing_2021)

        unexpected = self.frame.copy()
        unexpected.iloc[0, unexpected.columns.get_loc("season")] = 2025
        with self.assertRaisesRegex(ValidationDesignError, r"unexpected=\[2025\]"):
            build_validation_plan(unexpected)

    def test_empty_duplicate_and_non_forward_custom_folds_fail(self) -> None:
        with self.subTest("empty validation"):
            with self.assertRaisesRegex(ValidationDesignError, "validation partition is empty"):
                build_season_fold(
                    self.frame,
                    name="empty",
                    train_seasons=(2019, 2020),
                    validation_season=2025,
                )

        with self.subTest("duplicate train season"):
            with self.assertRaisesRegex(ValidationDesignError, "duplicate seasons"):
                build_season_fold(
                    self.frame,
                    name="duplicate",
                    train_seasons=(2019, 2019),
                    validation_season=2020,
                )

        with self.subTest("non-forward split"):
            with self.assertRaisesRegex(ValidationDesignError, "train max season"):
                build_season_fold(
                    self.frame,
                    name="backward",
                    train_seasons=(2019, 2024),
                    validation_season=2023,
                )


class DevelopmentValidationPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full_frame = make_validation_frame()
        self.development_frame = self.full_frame.loc[
            self.full_frame["season"].ne(2024)
        ].copy()

    def test_exposes_only_fixed_rolling_2022_and_2023_folds(self) -> None:
        plan = build_development_plan(self.development_frame)

        self.assertEqual(
            [fold.validation_season for fold in plan.development_folds],
            [2022, 2023],
        )
        self.assertEqual(
            [fold.train_seasons for fold in plan.development_folds],
            [
                (2019, 2020, 2021),
                (2019, 2020, 2021, 2022),
            ],
        )
        self.assertTrue(all(fold.role == "rolling" for fold in plan.development_folds))
        self.assertEqual(
            [fold.inner_validation_season for fold in plan.development_folds],
            [2021, 2022],
        )
        for prohibited_api in (
            "anchor_fold",
            "primary_holdout",
            "execution_folds",
            "rolling_folds",
            "recent_only_diagnostic",
        ):
            with self.subTest(api=prohibited_api):
                self.assertFalse(hasattr(plan, prohibited_api))

        summary = plan.to_summary()
        self.assertEqual(set(summary), {"scope", "development_folds"})
        self.assertEqual(summary["scope"], "development_only_2019_2023")
        self.assertNotIn("anchor", json.dumps(summary).lower())

    def test_folds_preserve_the_development_source_contract(self) -> None:
        plan = build_development_plan(self.development_frame)
        for fold in plan.development_folds:
            with self.subTest(fold=fold.name):
                train, validation = fold.split(self.development_frame)
                self.assertLess(int(train["season"].max()), int(validation["season"].min()))
                np.testing.assert_array_equal(
                    train["row_id"].to_numpy(dtype=object), fold.train_row_ids
                )
                np.testing.assert_array_equal(
                    validation["row_id"].to_numpy(dtype=object), fold.validation_row_ids
                )
                self.assertNotIn(2024, train["season"].unique())
                self.assertNotIn(2024, validation["season"].unique())

    def test_even_one_anchor_row_is_rejected_before_target_validation(self) -> None:
        one_anchor = self.full_frame.loc[
            self.full_frame["season"].eq(2024)
        ].iloc[[0]].copy()
        one_anchor["control_success"] = np.nan
        contaminated = pd.concat(
            [self.development_frame, one_anchor],
            axis=0,
        )

        with self.assertRaisesRegex(
            ValidationDesignError, "locked anchor season 2024"
        ):
            build_development_plan(contaminated)

    def test_exact_2019_to_2023_coverage_is_required(self) -> None:
        missing_2021 = self.development_frame.loc[
            self.development_frame["season"].ne(2021)
        ].copy()
        with self.assertRaisesRegex(ValidationDesignError, r"missing=\[2021\]"):
            build_development_plan(missing_2021)

        unexpected = self.development_frame.copy()
        unexpected.iloc[0, unexpected.columns.get_loc("season")] = 2025
        with self.assertRaisesRegex(ValidationDesignError, r"unexpected=\[2025\]"):
            build_development_plan(unexpected)


class PreprocessingFitScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_validation_frame()
        self.fold = build_validation_plan(self.frame).primary_holdout

    def test_outer_preprocessor_fit_receives_outer_train_only(self) -> None:
        fitted = fit_preprocessor_on_fold_train(
            RecordingPreprocessor, self.frame, self.fold
        )
        recorder = fitted.preprocessor
        self.assertIsInstance(recorder, RecordingPreprocessor)
        assert isinstance(recorder, RecordingPreprocessor)
        self.assertEqual(recorder.fit_seasons, [2019, 2020, 2021, 2022, 2023])
        self.assertEqual(recorder.fit_ids, self.fold.train_row_ids.tolist())
        self.assertEqual(
            recorder.fit_index,
            self.frame.iloc[self.fold.train_positions].index.tolist(),
        )
        self.assertTrue(
            set(recorder.fit_ids or []).isdisjoint(self.fold.validation_row_ids.tolist())
        )
        fitted.audit.assert_train_only()
        transformed = fitted.transform_validation(self.frame)
        np.testing.assert_array_equal(
            transformed["row_id"], self.fold.validation_row_ids
        )

    def test_inner_preprocessor_and_stop_frame_never_receive_outer_validation(self) -> None:
        fitted = fit_preprocessor_on_inner_train(
            RecordingPreprocessor, self.frame, self.fold
        )
        recorder = fitted.preprocessor
        assert isinstance(recorder, RecordingPreprocessor)
        self.assertEqual(recorder.fit_seasons, [2019, 2020, 2021, 2022])
        self.assertEqual(fitted.partition.validation_season, 2023)
        inner_stop = fitted.transform_validation(self.frame)
        self.assertEqual(set(inner_stop["season"]), {2023})
        outer_ids = set(self.fold.validation_row_ids)
        self.assertTrue(set(recorder.fit_ids or []).isdisjoint(outer_ids))
        self.assertTrue(set(inner_stop["row_id"]).isdisjoint(outer_ids))

    def test_fit_scope_audit_rejects_validation_and_mismatched_ids(self) -> None:
        assert_train_only_fit_scope(
            self.fold,
            self.fold.train_positions,
            fit_row_ids=self.fold.train_row_ids,
        )

        contaminated = np.concatenate(
            [self.fold.train_positions, self.fold.validation_positions[:1]]
        )
        with self.assertRaisesRegex(ValidationDesignError, "includes validation"):
            assert_train_only_fit_scope(self.fold, contaminated)

        with self.assertRaisesRegex(ValidationDesignError, "exactly all training"):
            assert_train_only_fit_scope(self.fold, self.fold.train_positions[:-1])
        assert_train_only_fit_scope(
            self.fold,
            self.fold.train_positions[:-1],
            require_all_training_rows=False,
        )

        wrong_ids = self.fold.train_row_ids.copy()
        wrong_ids.setflags(write=True)
        wrong_ids[0] = "WRONG_ID"
        with self.assertRaisesRegex(ValidationDesignError, "do not match"):
            assert_train_only_fit_scope(
                self.fold,
                self.fold.train_positions,
                fit_row_ids=wrong_ids,
            )

    def test_recent_only_diagnostic_cannot_supply_inner_selection_data(self) -> None:
        diagnostic = build_validation_plan(
            self.frame, include_recent_only=True
        ).recent_only_diagnostic
        assert diagnostic is not None
        with self.assertRaisesRegex(ValidationDesignError, "cannot select epochs"):
            diagnostic.inner_selection_frames(self.frame)
        with self.assertRaisesRegex(ValidationDesignError, "cannot select epochs"):
            fit_preprocessor_on_inner_train(
                RecordingPreprocessor, self.frame, diagnostic
            )


if __name__ == "__main__":
    unittest.main()
