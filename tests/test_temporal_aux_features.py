from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from lg_control.temporal_aux_features import (
    AUXILIARY_ENCODING_SPECS,
    AUXILIARY_OUTCOMES,
    PriorSeasonAuxiliaryEncoder,
    TEMPORAL_AUX_FEATURE_COLUMNS,
    TemporalAuxAugmentedPreprocessor,
)
from tests.test_preprocessing import make_frame


def make_temporal_fold_train(*, future_variant: bool = False) -> pd.DataFrame:
    base = make_frame()
    ordinary = [
        ("P1", "B1", 1, 0, 0, 0, 1),
        ("P2", "B2", 0, 1, 0, 0, 0),
        ("P1", "B1", 0, 0, 1, 0, 1),
        ("P2", "B2", 1, 0, 0, 1, 0),
    ]
    future = [
        ("P1", "B1", 0, 1, 0, 1, 0),
        ("P2", "B2", 1, 0, 0, 0, 1),
        ("P1", "B1", 0, 0, 1, 0, 1),
        ("P2", "B2", 1, 0, 0, 1, 0),
    ]
    pitcher_state: dict[str, dict[str, int]] = {}
    batter_state: dict[str, dict[str, int]] = {}
    rows: list[pd.Series] = []
    position = 0
    for season in (2019, 2020, 2021):
        events = future if future_variant and season == 2021 else ordinary
        for local, (pitcher, batter, success, reverse, middle, ball, strike) in enumerate(
            events
        ):
            row = base.iloc[position % len(base)].copy(deep=True)
            pstate = pitcher_state.setdefault(
                pitcher,
                {
                    "n": 0,
                    "success": 0,
                    "reverse": 0,
                    "middle": 0,
                    "ball": 0,
                    "strike": 0,
                },
            )
            bstate = batter_state.setdefault(
                batter, {"n": 0, "success": 0, "middle": 0}
            )

            def rate(state: dict[str, int], name: str) -> float:
                return np.nan if state["n"] == 0 else state[name] / state["n"]

            row["row_id"] = f"S{season}_{local}"
            row["season"] = season
            row["game_type"] = "R" if local % 2 == 0 else "F"
            row["pitcher_id"] = pitcher
            row["batter_id"] = batter
            row["pitcher_hand"] = 1 if pitcher == "P1" else 2
            row["batter_hand"] = 2 if batter == "B1" else 1
            row["asof_pitcher_n"] = pstate["n"]
            for name in ("success", "reverse", "middle", "ball", "strike"):
                row[f"asof_pitcher_{name}_rate"] = rate(pstate, name)
            row["asof_batter_n"] = bstate["n"]
            for name in ("success", "middle"):
                row[f"asof_batter_{name}_rate"] = rate(bstate, name)
            row["asof_pitcher_pitchmix_n"] = pstate["n"]
            row["control_success"] = success
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
            position += 1
    return pd.DataFrame(rows).reset_index(drop=True).loc[:, base.columns]


def make_inference_rows(frame: pd.DataFrame, season: int = 2022) -> pd.DataFrame:
    result = frame.iloc[:4].copy(deep=True)
    result["season"] = season
    result["row_id"] = [f"INFER_{season}_{index}" for index in range(len(result))]
    return result.drop(columns=["control_success"])


class PriorSeasonAuxiliaryEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fold_train = make_temporal_fold_train()

    def test_feature_contract_and_game_type_prior(self) -> None:
        encoder = PriorSeasonAuxiliaryEncoder().fit(self.fold_train)
        self.assertEqual(len(TEMPORAL_AUX_FEATURE_COLUMNS), 64)
        self.assertEqual(len(AUXILIARY_ENCODING_SPECS), 4)
        self.assertEqual(AUXILIARY_OUTCOMES, ("reverse", "middle", "ball", "strike"))
        for spec in AUXILIARY_ENCODING_SPECS:
            for outcome in AUXILIARY_OUTCOMES:
                for suffix in ("prev1_rate", "prev2_rate", "prev1_minus_prev2"):
                    self.assertIn(
                        f"auxte_{spec.name}_{outcome}_{suffix}",
                        TEMPORAL_AUX_FEATURE_COLUMNS,
                    )

        cold = make_inference_rows(self.fold_train, season=2020).iloc[:2].copy()
        cold["pitcher_id"] = "UNSEEN"
        cold["game_type"] = ["R", "F"]
        encoded = encoder.transform(cold)
        self.assertNotEqual(
            encoded.iloc[0]["auxte_pitcher_reverse_prev1_rate"],
            encoded.iloc[1]["auxte_pitcher_reverse_prev1_rate"],
        )
        self.assertTrue(np.isfinite(encoded.to_numpy()).all())

    def test_strict_fold_and_prior_season_boundaries(self) -> None:
        first = PriorSeasonAuxiliaryEncoder().fit(self.fold_train)
        changed_future = PriorSeasonAuxiliaryEncoder().fit(
            make_temporal_fold_train(future_variant=True)
        )
        before_future = make_inference_rows(self.fold_train, season=2020)
        np.testing.assert_array_equal(
            first.transform(before_future).to_numpy(),
            changed_future.transform(before_future).to_numpy(),
        )

        earliest = make_inference_rows(self.fold_train, season=2019)
        earliest_values = first.transform(earliest)
        rate_columns = [name for name in earliest_values if name.endswith("_rate")]
        reliability_columns = [
            name for name in earliest_values if name.endswith("_reliability")
        ]
        np.testing.assert_array_equal(
            earliest_values[rate_columns].to_numpy(), 0.5
        )
        np.testing.assert_array_equal(
            earliest_values[reliability_columns].to_numpy(), 0.0
        )

    def test_fit_calls_fold_train_recovery_contract(self) -> None:
        import lg_control.temporal_aux_features as module

        with mock.patch.object(
            module,
            "recover_fold_train_auxiliary_targets",
            wraps=module.recover_fold_train_auxiliary_targets,
        ) as recover:
            PriorSeasonAuxiliaryEncoder().fit(self.fold_train)
        recover.assert_called_once()
        self.assertEqual(recover.call_args.kwargs, {"partition": "fold_train"})


class TemporalAuxAugmentedPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fold_train = make_temporal_fold_train()
        self.inference = make_inference_rows(self.fold_train)

    def test_layout_target_independence_and_batch_independence(self) -> None:
        preprocessor = TemporalAuxAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(self.fold_train)
        expected = preprocessor.transform(self.inference)
        layout = preprocessor.feature_layout()
        self.assertEqual(expected["numerical"].shape[1], len(layout.numerical_names))
        self.assertEqual(expected["categorical"].shape[1], len(layout.categorical_names))
        self.assertTrue(np.isfinite(expected["numerical"]).all())

        labelled = self.inference.copy()
        labelled["control_success"] = [1, 1, 0, 0]
        changed = labelled.copy()
        changed["control_success"] = 1 - changed["control_success"]
        np.testing.assert_array_equal(
            preprocessor.transform(labelled)["numerical"],
            preprocessor.transform(changed)["numerical"],
        )

        extended = pd.concat(
            [self.inference, make_inference_rows(self.fold_train, season=2099)],
            ignore_index=True,
        )
        np.testing.assert_array_equal(
            preprocessor.transform(extended)["numerical"][: len(self.inference)],
            expected["numerical"],
        )

    def test_singleton_and_order_are_bit_exact(self) -> None:
        preprocessor = TemporalAuxAugmentedPreprocessor(
            min_category_frequency=1
        ).fit(self.fold_train)
        expected = preprocessor.transform(self.inference)
        order = [2, 0, 3, 1]
        shuffled = preprocessor.transform(self.inference.iloc[order])
        inverse = np.argsort(order)
        np.testing.assert_array_equal(
            shuffled["numerical"][inverse], expected["numerical"]
        )
        np.testing.assert_array_equal(
            shuffled["categorical"][inverse], expected["categorical"]
        )
        for position in range(len(self.inference)):
            singleton = preprocessor.transform(self.inference.iloc[[position]])
            np.testing.assert_array_equal(
                singleton["numerical"][0], expected["numerical"][position]
            )
            np.testing.assert_array_equal(
                singleton["categorical"][0], expected["categorical"][position]
            )

    def test_json_state_round_trip_is_bit_exact(self) -> None:
        fitted = TemporalAuxAugmentedPreprocessor(min_category_frequency=1).fit(
            self.fold_train
        )
        serialized = json.dumps(
            fitted.to_state_dict(), sort_keys=True, allow_nan=False
        )
        restored = TemporalAuxAugmentedPreprocessor.from_state_dict(
            json.loads(serialized)
        )
        first = fitted.transform(self.inference)
        second = restored.transform(self.inference)
        for name in ("row_id", "categorical", "numerical"):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(
            fitted.feature_layout().preprocessor_fingerprint,
            restored.feature_layout().preprocessor_fingerprint,
        )
        self.assertEqual(
            serialized,
            json.dumps(restored.to_state_dict(), sort_keys=True, allow_nan=False),
        )

    def test_unfitted_use_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            PriorSeasonAuxiliaryEncoder().transform(self.inference)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            TemporalAuxAugmentedPreprocessor().transform(self.inference)


if __name__ == "__main__":
    unittest.main()
