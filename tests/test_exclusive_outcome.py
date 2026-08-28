from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from lg_control.exclusive_outcome import (
    OUTCOME_NAMES,
    ExclusiveOutcomeGatedCrossModel,
    ExclusiveOutcomeHeadConfig,
    ExclusiveOutcomeLabelError,
    build_exclusive_outcome_model,
    compose_exclusive_outcome_targets,
    predict_exclusive_success_arrays,
    recover_fold_train_exclusive_targets,
    train_exclusive_outcome_fixed_epochs,
    train_exclusive_outcome_model,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.regime_model import GatedCrossModelConfig
from lg_control.training import TrainConfig
from tests.test_auxiliary_targets import make_fold_train_frame
from tests.test_preprocessing import make_frame


class ExclusiveOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()
        self.preprocessor = DeepTabularPreprocessor().fit(self.frame)
        self.arrays = self.preprocessor.transform(self.frame)
        self.model_config = GatedCrossModelConfig(
            embedding_dim_cap=8,
            interaction_dim=24,
            cross_layers=1,
            num_experts=2,
            low_rank=8,
            deep_hidden_dim=32,
            deep_layers=1,
            dropout=0.0,
        )
        self.head_config = ExclusiveOutcomeHeadConfig(
            hidden_dim=16,
            dropout=0.0,
            use_ball_strike_auxiliary=True,
        )
        self.model = build_exclusive_outcome_model(
            self.preprocessor,
            config=self.model_config,
            head_config=self.head_config,
            initialization_seed=123,
        )

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.arrays["categorical"]),
            torch.from_numpy(self.arrays["numerical"]),
        )

    def _targets(self):
        success = self.arrays["target"].astype(np.float32)
        rows = len(success)
        observed = np.ones(rows, dtype=bool)
        observed[-1] = False
        reverse = np.zeros(rows, dtype=np.float32)
        middle = np.zeros(rows, dtype=np.float32)
        failures = np.flatnonzero(observed & (success == 0.0))
        if len(failures) > 0:
            reverse[failures[0]] = 1.0
        if len(failures) > 1:
            middle[failures[1]] = 1.0
        ball = np.arange(rows, dtype=np.float32) % 2
        strike = 1.0 - ball
        return compose_exclusive_outcome_targets(
            success, reverse, middle, ball, strike, observed
        )

    def test_probabilities_are_exhaustive_and_forward_is_success_log_odds(self) -> None:
        self.assertEqual(
            OUTCOME_NAMES,
            (
                "control_success",
                "reverse_only",
                "middle_only",
                "reverse_and_middle",
                "other_failure",
            ),
        )
        self.assertEqual(
            self.model.architecture_name,
            "dcnv2_gated_experts_exclusive_outcome_v2",
        )
        categorical, numerical = self._tensors()
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.forward_with_aux(categorical, numerical)
            forward = self.model(categorical, numerical)
        probabilities = outputs["outcome_probabilities"]
        self.assertEqual(tuple(probabilities.shape), (len(self.frame), 5))
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones(len(self.frame)), rtol=1e-6, atol=1e-7
        )
        self.assertTrue((probabilities >= 0.0).all().item())
        self.assertTrue((probabilities <= 1.0).all().item())
        torch.testing.assert_close(
            outputs["success_probability"], probabilities[:, 0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.sigmoid(forward), probabilities[:, 0], rtol=1e-6, atol=1e-7
        )
        self.assertEqual(tuple(outputs["ball_strike_logits"].shape), (len(self.frame), 2))

    def test_invalid_overlap_is_rejected_and_unobserved_rows_are_masked(self) -> None:
        success = np.array([1, 0, 0], dtype=np.float32)
        with self.assertRaisesRegex(ExclusiveOutcomeLabelError, "exclusive with"):
            compose_exclusive_outcome_targets(
                success,
                np.array([1, 0, 0]),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.ones(3, dtype=bool),
            )

        targets = compose_exclusive_outcome_targets(
            success,
            np.array([0, 1, np.nan]),
            np.array([0, 0, np.nan]),
            np.array([1, 0, np.nan]),
            np.array([0, 1, np.nan]),
            np.array([True, True, False]),
        )
        np.testing.assert_array_equal(targets.class_indices, [0, 1, -1])
        np.testing.assert_array_equal(targets.outcome_mask, [True, True, False])
        np.testing.assert_array_equal(targets.ball_strike_mask[-1], [False, False])
        np.testing.assert_array_equal(targets.ball_strike_targets[-1], [0.0, 0.0])

    def test_fold_train_recovery_produces_exhaustive_labels_and_scope_guard(self) -> None:
        frame = make_fold_train_frame()
        targets = recover_fold_train_exclusive_targets(frame, partition="fold_train")
        np.testing.assert_array_equal(targets.outcome_mask, [True, True, True, True, False, False])
        np.testing.assert_array_equal(targets.class_indices, [0, 1, 2, 0, -1, -1])
        self.assertEqual(int((targets.class_indices == 3).sum()), 0)
        with self.assertRaisesRegex(ExclusiveOutcomeLabelError, "fold_train"):
            recover_fold_train_exclusive_targets(frame, partition="validation")

    def test_reverse_middle_joint_and_other_failure_have_distinct_classes(self) -> None:
        targets = compose_exclusive_outcome_targets(
            [0, 0], [1, 0], [1, 0], [1, 1], [0, 0], [True, True]
        )
        self.assertEqual(targets.class_indices.tolist(), [3, 4])

    def test_config_state_and_parameter_report_round_trip(self) -> None:
        payload = json.loads(json.dumps(self.model.get_config()))
        restored = ExclusiveOutcomeGatedCrossModel.from_config(payload)
        incompatibilities = restored.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])
        self.assertEqual(restored.get_config(), payload)
        report = restored.parameter_report()
        self.assertEqual(
            report["total_parameters"], sum(value.numel() for value in restored.parameters())
        )

    def test_optional_ball_strike_head_can_be_disabled(self) -> None:
        model = build_exclusive_outcome_model(
            self.preprocessor,
            config=self.model_config,
            head_config=ExclusiveOutcomeHeadConfig(
                hidden_dim=16, dropout=0.0, use_ball_strike_auxiliary=False
            ),
            initialization_seed=5,
        ).eval()
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            outputs = model.forward_with_aux(categorical, numerical)
        self.assertNotIn("ball_strike_logits", outputs)

    def test_eval_and_prediction_are_independent_of_batch_composition(self) -> None:
        categorical, numerical = self._tensors()
        self.model.eval()
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        with torch.inference_mode():
            full = self.model.forward_with_aux(categorical, numerical)["success_probability"]
            singles = torch.cat(
                [
                    self.model.forward_with_aux(
                        categorical[i : i + 1], numerical[i : i + 1]
                    )["success_probability"]
                    for i in range(len(categorical))
                ]
            )
            shuffled = self.model.forward_with_aux(
                categorical.index_select(0, permutation),
                numerical.index_select(0, permutation),
            )["success_probability"]
            restored = torch.empty_like(shuffled)
            restored[permutation] = shuffled
        torch.testing.assert_close(full, singles, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(full, restored, rtol=1e-6, atol=1e-7)

        device = torch.device("cpu")
        batch_two = predict_exclusive_success_arrays(
            self.model,
            self.arrays["categorical"],
            self.arrays["numerical"],
            batch_size=2,
            device=device,
        )
        batch_all = predict_exclusive_success_arrays(
            self.model,
            self.arrays["categorical"],
            self.arrays["numerical"],
            batch_size=len(self.frame),
            device=device,
        )
        np.testing.assert_allclose(batch_two, batch_all, rtol=1e-6, atol=1e-7)

    def test_one_step_training_is_deterministic_and_uses_fallback_mask(self) -> None:
        targets = self._targets()
        config = TrainConfig(
            epochs=1,
            batch_size=3,
            eval_batch_size=4,
            learning_rate=1e-3,
            patience=1,
            seed=987,
            use_amp=False,
            verbose=False,
        )
        models = [
            build_exclusive_outcome_model(
                self.preprocessor,
                config=self.model_config,
                head_config=self.head_config,
                initialization_seed=44,
            )
            for _ in range(2)
        ]
        results = [
            train_exclusive_outcome_model(
                model,
                self.arrays["categorical"],
                self.arrays["numerical"],
                targets,
                self.arrays["categorical"],
                self.arrays["numerical"],
                self.arrays["target"],
                config=config,
                device="cpu",
            )
            for model in models
        ]
        np.testing.assert_array_equal(results[0].probabilities, results[1].probabilities)
        self.assertEqual(results[0].best_epoch, 1)
        self.assertTrue(np.isfinite(results[0].history[0]["train_loss"]))
        for name, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[name]), name)

    def test_fixed_epoch_training_has_no_validation_selection_path(self) -> None:
        targets = self._targets()
        config = TrainConfig(
            epochs=2,
            scheduler_epochs=4,
            batch_size=3,
            eval_batch_size=4,
            learning_rate=1e-3,
            patience=1,
            seed=222,
            use_amp=False,
            verbose=False,
        )
        models = [
            build_exclusive_outcome_model(
                self.preprocessor,
                config=self.model_config,
                head_config=self.head_config,
                initialization_seed=55,
            )
            for _ in range(2)
        ]
        histories = [
            train_exclusive_outcome_fixed_epochs(
                model,
                self.arrays["categorical"],
                self.arrays["numerical"],
                targets,
                config=config,
                device="cpu",
            )
            for model in models
        ]
        self.assertEqual([row["epoch"] for row in histories[0]], [1.0, 2.0])
        self.assertNotIn("brier", histories[0][-1])
        self.assertEqual(histories[0], histories[1])
        for name, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
