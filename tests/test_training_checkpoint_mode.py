import unittest

import numpy as np
import torch
from torch import nn

from lg_control.training import TrainConfig, train_model


class _ConstantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        del categorical, numerical
        return self.bias.expand(4)


class TrainingCheckpointModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.categorical = np.zeros((4, 1), dtype=np.int64)
        self.numerical = np.zeros((4, 1), dtype=np.float32)
        self.target = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        self.config = TrainConfig(
            epochs=3,
            scheduler_epochs=4,
            batch_size=4,
            eval_batch_size=4,
            learning_rate=0.0,
            patience=1,
            use_amp=False,
            verbose=False,
        )

    def test_last_checkpoint_runs_all_preselected_epochs(self) -> None:
        result = train_model(
            _ConstantModel(),
            self.categorical,
            self.numerical,
            self.target,
            self.categorical,
            self.numerical,
            self.target,
            config=self.config,
            checkpoint_mode="last",
        )
        self.assertEqual(result.best_epoch, 3)
        self.assertEqual(len(result.history), 3)

    def test_best_checkpoint_preserves_early_stopping(self) -> None:
        result = train_model(
            _ConstantModel(),
            self.categorical,
            self.numerical,
            self.target,
            self.categorical,
            self.numerical,
            self.target,
            config=self.config,
        )
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(len(result.history), 2)

    def test_invalid_checkpoint_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_mode"):
            train_model(
                _ConstantModel(),
                self.categorical,
                self.numerical,
                self.target,
                self.categorical,
                self.numerical,
                self.target,
                config=self.config,
                checkpoint_mode="outer_validation_best",
            )

    def test_scheduler_horizon_cannot_end_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "scheduler_epochs"):
            TrainConfig(epochs=4, scheduler_epochs=3)


if __name__ == "__main__":
    unittest.main()
