from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from lg_control import auxiliary_targets
from lg_control.auxiliary_targets import (
    AuxiliaryTargetError,
    AuxiliaryTargetScopeError,
    recover_fold_train_auxiliary_targets,
)


_AUXILIARY_NAMES = ("reverse", "middle", "ball", "strike")


def make_fold_train_frame() -> pd.DataFrame:
    events = [
        # pitcher, batter, success, reverse, middle, ball, strike
        ("P_A", "B_X", 1, 0, 0, 0, 1),
        ("P_B", "B_Y", 0, 1, 0, 0, 0),
        ("P_A", "B_Y", 0, 0, 1, 0, 1),
        ("P_B", "B_X", 1, 0, 0, 1, 0),
        ("P_A", "B_X", 0, 0, 0, 1, 0),
        ("P_B", "B_Y", 1, 0, 0, 0, 1),
    ]
    pitcher_state: dict[str, dict[str, int]] = {}
    batter_state: dict[str, dict[str, int]] = {}
    rows: list[dict[str, object]] = []

    for index, (pitcher, batter, success, reverse, middle, ball, strike) in enumerate(
        events
    ):
        pstate = pitcher_state.setdefault(
            pitcher,
            {"n": 0, "success": 0, "reverse": 0, "middle": 0, "ball": 0, "strike": 0},
        )
        bstate = batter_state.setdefault(batter, {"n": 0, "success": 0, "middle": 0})

        def rate(state: dict[str, int], name: str) -> float:
            return np.nan if state["n"] == 0 else state[name] / state["n"]

        row: dict[str, object] = {
            "row_id": f"TRAIN_SYNTH_{index}",
            "pitcher_id": pitcher,
            "batter_id": batter,
            "asof_pitcher_n": pstate["n"],
            "asof_batter_n": bstate["n"],
            "control_success": success,
        }
        for name in ("success", *_AUXILIARY_NAMES):
            row[f"asof_pitcher_{name}_rate"] = rate(pstate, name)
        for name in ("success", "middle"):
            row[f"asof_batter_{name}_rate"] = rate(bstate, name)
        rows.append(row)

        pstate["n"] += 1
        pstate["success"] += success
        pstate["reverse"] += reverse
        pstate["middle"] += middle
        pstate["ball"] += ball
        pstate["strike"] += strike
        bstate["n"] += 1
        bstate["success"] += success
        bstate["middle"] += middle

    # Non-consecutive index values verify that the result preserves both the
    # source index and row_id rather than silently resetting or sorting them.
    return pd.DataFrame(rows, index=[11, 3, 19, 7, 23, 5])


class AuxiliaryTargetRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_fold_train_frame()

    def test_recovers_labels_masks_and_preserves_source_identity(self) -> None:
        original = self.frame.copy(deep=True)
        result = recover_fold_train_auxiliary_targets(
            self.frame,
            partition="fold_train",
        )

        self.assertEqual(
            list(result.columns),
            [
                "row_id",
                "aux_reverse",
                "aux_middle",
                "aux_ball",
                "aux_strike",
                "auxiliary_valid_mask",
                "consistency_mask",
            ],
        )
        pd.testing.assert_index_equal(result.index, self.frame.index)
        pd.testing.assert_series_equal(
            result["row_id"], self.frame["row_id"], check_names=True
        )
        pd.testing.assert_frame_equal(self.frame, original)

        expected = {
            "aux_reverse": [0.0, 1.0, 0.0, 0.0, np.nan, np.nan],
            "aux_middle": [0.0, 0.0, 1.0, 0.0, np.nan, np.nan],
            "aux_ball": [0.0, 0.0, 0.0, 1.0, np.nan, np.nan],
            "aux_strike": [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
        }
        for column, values in expected.items():
            np.testing.assert_allclose(
                result[column].to_numpy(),
                np.asarray(values, dtype=np.float32),
                equal_nan=True,
            )
            self.assertEqual(result[column].dtype, np.float32)
        np.testing.assert_array_equal(
            result["auxiliary_valid_mask"], [True, True, True, True, False, False]
        )
        np.testing.assert_array_equal(
            result["consistency_mask"], [False, True, True, False, False, False]
        )
        self.assertEqual(result["auxiliary_valid_mask"].dtype, bool)
        self.assertEqual(result["consistency_mask"].dtype, bool)

    def test_n_plus_one_is_required_and_terminal_rows_stay_missing(self) -> None:
        discontinuous = self.frame.copy(deep=True)
        # P_A's second source occurrence now skips from n=0 to n=2.  Neither
        # adjacent P_A pair is a valid n+1 transition inside this fold.
        discontinuous.loc[19, "asof_pitcher_n"] = 2
        result = recover_fold_train_auxiliary_targets(
            discontinuous,
            partition="fold_train",
        )

        p_a_positions = [0, 2, 4]
        self.assertFalse(result.iloc[p_a_positions]["auxiliary_valid_mask"].any())
        self.assertTrue(
            result.iloc[p_a_positions][
                [f"aux_{name}" for name in _AUXILIARY_NAMES]
            ].isna().all().all()
        )
        # The final occurrence of every pitcher remains missing even where a
        # batter has another occurrence elsewhere in the fold.
        self.assertFalse(result.iloc[[4, 5]]["auxiliary_valid_mask"].any())

    def test_success_and_middle_cross_checks_are_strict(self) -> None:
        wrong_target = self.frame.copy(deep=True)
        wrong_target.loc[11, "control_success"] = 0
        with self.assertRaisesRegex(AuxiliaryTargetError, "pitcher success"):
            recover_fold_train_auxiliary_targets(
                wrong_target,
                partition="fold_train",
            )

        wrong_batter_success = self.frame.copy(deep=True)
        # Make B_X's cumulative path imply success labels 0, 1 rather than the
        # true 1, 1 while keeping its cumulative increments binary.
        wrong_batter_success.loc[7, "asof_batter_success_rate"] = 0.0
        wrong_batter_success.loc[23, "asof_batter_success_rate"] = 0.5
        with self.assertRaisesRegex(AuxiliaryTargetError, "batter success"):
            recover_fold_train_auxiliary_targets(
                wrong_batter_success,
                partition="fold_train",
            )

        wrong_batter_middle = self.frame.copy(deep=True)
        # Make B_X's reconstructed middle labels 1, 0 instead of 0, 0.
        wrong_batter_middle.loc[7, "asof_batter_middle_rate"] = 1.0
        wrong_batter_middle.loc[23, "asof_batter_middle_rate"] = 0.5
        with self.assertRaisesRegex(AuxiliaryTargetError, "pitcher middle versus batter middle"):
            recover_fold_train_auxiliary_targets(
                wrong_batter_middle,
                partition="fold_train",
            )

    def test_result_is_deterministic_label_only_and_has_no_artifact_api(self) -> None:
        first = recover_fold_train_auxiliary_targets(
            self.frame,
            partition="fold_train",
        )
        second = recover_fold_train_auxiliary_targets(
            self.frame.copy(deep=True),
            partition="fold_train",
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            auxiliary_targets.__all__,
            (
                "AuxiliaryTargetError",
                "AuxiliaryTargetScopeError",
                "recover_fold_train_auxiliary_targets",
            ),
        )
        for forbidden_name in ("save", "load", "state_dict", "to_artifact", "transform"):
            self.assertFalse(hasattr(auxiliary_targets, forbidden_name))
        self.assertNotIn("control_success", first.columns)
        self.assertNotIn("pitcher_id", first.columns)
        self.assertNotIn("asof_pitcher_n", first.columns)


class AuxiliaryTargetScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_fold_train_frame()

    def test_validation_and_test_are_rejected_before_shift(self) -> None:
        for partition in ("validation", "test", "inference", "fold_stop"):
            with self.subTest(partition=partition):
                with mock.patch.object(
                    auxiliary_targets,
                    "_shift_within_entity",
                    side_effect=AssertionError("shift must not run"),
                ) as shift:
                    with self.assertRaises(AuxiliaryTargetScopeError):
                        recover_fold_train_auxiliary_targets(
                            self.frame,
                            partition=partition,
                        )
                    shift.assert_not_called()

    def test_unlabelled_fold_train_is_rejected_before_shift(self) -> None:
        unlabelled = self.frame.drop(columns=["control_success"])
        with mock.patch.object(
            auxiliary_targets,
            "_shift_within_entity",
            side_effect=AssertionError("shift must not run"),
        ) as shift:
            with self.assertRaisesRegex(AuxiliaryTargetError, "control_success"):
                recover_fold_train_auxiliary_targets(
                    unlabelled,
                    partition="fold_train",
                )
            shift.assert_not_called()


if __name__ == "__main__":
    unittest.main()
