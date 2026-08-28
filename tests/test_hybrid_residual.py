import unittest

import numpy as np
import pandas as pd
import torch

from lg_control.hybrid_residual import (
    CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    HybridResidualPreprocessor,
    HybridResidualTrainConfig,
)


def _frame() -> pd.DataFrame:
    rows = 6
    payload: dict[str, object] = {
        "row_id": [f"TRAIN_{index:07d}" for index in range(rows)],
        "season": [2021] * rows,
        "y_true": [0, 1, 0, 1, 0, 1],
        "prediction_v7": np.linspace(0.40, 0.60, rows),
        "prediction_rate_exclusive": np.linspace(0.42, 0.58, rows),
        "prediction_temporal_te": np.linspace(0.38, 0.62, rows),
        "prediction_equal": np.linspace(0.40, 0.60, rows),
        "v7_seed_std": np.linspace(0.005, 0.01, rows),
        "season_success_delta_k20": np.linspace(-0.1, 0.1, rows),
        "rate_exclusive_head_outcome_reverse_only": np.linspace(0.1, 0.2, rows),
        "temporal_te_head_aux_middle": np.linspace(0.2, 0.1, rows),
    }
    for position, column in enumerate(RAW_NUMERIC_COLUMNS):
        payload[column] = np.linspace(position, position + 1, rows)
    payload.update(
        {
            "game_type": ["R", "F", "R", "F", "R", "F"],
            "game_month": [4, 5, 6, 7, 8, 9],
            "game_dayofweek": [0, 1, 2, 3, 4, 5],
            "top_bottom": [0, 1, 0, 1, 0, 1],
            "balls_before": [0, 1, 2, 3, 0, 1],
            "strikes_before": [0, 1, 2, 0, 1, 2],
            "outs_before": [0, 1, 2, 0, 1, 2],
            "base_state": [0, 1, 2, 3, 4, 5],
            "pitcher_hand": [0, 1, 0, 1, 0, 1],
            "batter_hand": [0, 1, 1, 0, 0, 1],
        }
    )
    return pd.DataFrame(payload)


class HybridResidualTests(unittest.TestCase):
    def test_preprocessor_round_trip_is_stable_and_target_free(self) -> None:
        frame = _frame()
        preprocessor = HybridResidualPreprocessor().fit(frame.iloc[:4])
        expected = preprocessor.transform(frame)
        restored = HybridResidualPreprocessor.from_state_dict(
            preprocessor.to_state_dict()
        )
        observed = restored.transform(frame)
        np.testing.assert_array_equal(expected, observed)
        self.assertTrue(np.isfinite(observed).all())
        self.assertFalse(
            any(
                forbidden in name
                for name in restored.feature_names_
                for forbidden in ("row_id", "y_true", "season==")
            )
        )

    def test_zero_initialized_residual_is_exactly_the_base(self) -> None:
        model = BoundedLogitResidualNet(
            HybridResidualModelConfig(input_dim=3, correction_cap=0.05, dropout=0.0)
        )
        features = torch.zeros((2, 3))
        base_probability = torch.tensor([0.2, 0.8])
        base_logit = torch.logit(base_probability)
        probability, correction = model(features, base_logit)
        torch.testing.assert_close(correction, torch.zeros_like(correction))
        torch.testing.assert_close(probability, base_probability)

    def test_residual_is_bounded_by_configured_cap(self) -> None:
        cap = 0.03
        model = BoundedLogitResidualNet(
            HybridResidualModelConfig(input_dim=2, correction_cap=cap, dropout=0.0)
        )
        with torch.no_grad():
            model.output.bias.fill_(100.0)
        _, correction = model(torch.zeros((4, 2)), torch.zeros(4))
        self.assertTrue(torch.all(correction <= cap))
        self.assertTrue(torch.all(correction >= -cap))

    def test_scheduler_horizon_must_cover_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "scheduler_epochs"):
            HybridResidualTrainConfig(epochs=5, scheduler_epochs=4)

    def test_declared_categoricals_are_present_in_fixture(self) -> None:
        self.assertTrue(set(CATEGORICAL_COLUMNS).issubset(_frame().columns))


if __name__ == "__main__":
    unittest.main()
