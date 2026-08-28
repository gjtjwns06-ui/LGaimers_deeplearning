from __future__ import annotations

import json
import unittest

import numpy as np
import torch
from torch import nn

from lg_control.model import assert_preprocessor_compatible
from lg_control.numerical_embedding_model import (
    PeriodicPiecewiseModelConfig,
    PeriodicPiecewiseResidualModel,
    build_periodic_piecewise_model,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.training import predict_arrays
from tests.test_preprocessing import make_frame


class PeriodicPiecewiseResidualModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()
        self.preprocessor = DeepTabularPreprocessor().fit(self.frame)
        self.arrays = self.preprocessor.transform(self.frame)
        self.config = PeriodicPiecewiseModelConfig(
            embedding_dim_cap=7,
            numerical_embedding_dim=3,
            periodic_frequencies=2,
            piecewise_knots=(-2.0, -0.5, 0.5, 2.0),
            raw_value_clip=5.0,
            hidden_dim=24,
            ffn_dim=32,
            num_blocks=2,
            dropout=0.10,
        )
        self.model = build_periodic_piecewise_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=137,
        )

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.arrays["categorical"]),
            torch.from_numpy(self.arrays["numerical"]),
        )

    def test_layout_forward_and_existing_predict_api(self) -> None:
        assert_preprocessor_compatible(self.model.feature_layout, self.preprocessor)
        categorical, numerical = self._tensors()
        self.model.eval()
        with torch.inference_mode():
            details = self.model.forward_with_details(categorical, numerical)
            logits = self.model(categorical, numerical)
        self.assertEqual(tuple(logits.shape), (len(self.frame),))
        torch.testing.assert_close(logits, details["logits"], rtol=0, atol=0)
        for value in details.values():
            self.assertTrue(torch.isfinite(value).all().item())

        probabilities = predict_arrays(
            self.model,
            self.arrays["categorical"],
            self.arrays["numerical"],
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(probabilities.shape, (len(self.frame),))
        self.assertTrue(np.all((probabilities > 0.0) & (probabilities < 1.0)))

    def test_eval_is_batch_composition_and_order_independent(self) -> None:
        categorical, numerical = self._tensors()
        self.model.eval()
        with torch.inference_mode():
            together = self.model(categorical, numerical)
            separately = torch.cat(
                [
                    self.model(categorical[index : index + 1], numerical[index : index + 1])
                    for index in range(len(categorical))
                ]
            )
            permutation = torch.arange(len(categorical) - 1, -1, -1)
            reversed_logits = self.model(
                categorical[permutation], numerical[permutation]
            )[permutation]
        torch.testing.assert_close(together, separately, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(together, reversed_logits, rtol=1e-6, atol=1e-7)
        self.assertFalse(
            any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                for module in self.model.modules()
            )
        )

    def test_seeded_initialization_does_not_change_ambient_rng(self) -> None:
        torch.manual_seed(89)
        expected = torch.rand(5)
        torch.manual_seed(89)
        first = build_periodic_piecewise_model(
            self.preprocessor, config=self.config, initialization_seed=41
        )
        observed = torch.rand(5)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        second = build_periodic_piecewise_model(
            self.preprocessor, config=self.config, initialization_seed=41
        )
        for (first_name, first_value), (second_name, second_value) in zip(
            first.state_dict().items(), second.state_dict().items(), strict=True
        ):
            self.assertEqual(first_name, second_name)
            torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)

    def test_strict_config_and_state_round_trip(self) -> None:
        payload = json.loads(json.dumps(self.model.get_config()))
        restored = PeriodicPiecewiseResidualModel.from_config(payload)
        incompatibilities = restored.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])
        self.assertEqual(restored.get_config(), payload)
        categorical, numerical = self._tensors()
        self.model.eval()
        restored.eval()
        with torch.inference_mode():
            expected = self.model(categorical, numerical)
            actual = restored(categorical, numerical)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        malformed = dict(payload)
        malformed["unexpected"] = True
        with self.assertRaises(ValueError):
            PeriodicPiecewiseResidualModel.from_config(malformed)
        malformed_model = dict(payload)
        malformed_model["model"] = dict(payload["model"], unexpected=True)
        with self.assertRaises(ValueError):
            PeriodicPiecewiseResidualModel.from_config(malformed_model)

    def test_all_parameter_groups_receive_finite_one_step_gradients(self) -> None:
        categorical, numerical = self._tensors()
        # Nonzero perturbation ensures both Fourier frequency and phase gradients.
        numerical = numerical + torch.linspace(
            -0.2, 0.2, len(numerical), dtype=torch.float32
        ).unsqueeze(1)
        targets = torch.linspace(0.0, 1.0, len(numerical))
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        logits = self.model(categorical, numerical)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        before = self.model.numerical_embedding.raw_frequencies.detach().clone()
        optimizer.step()
        self.assertFalse(
            torch.equal(before, self.model.numerical_embedding.raw_frequencies.detach())
        )

    def test_parameter_and_compute_reports_are_exact_and_cpu_feasible(self) -> None:
        parameter_report = self.model.parameter_report()
        exact = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(parameter_report["total_parameters"], exact)
        self.assertEqual(parameter_report["trainable_parameters"], exact)
        self.assertLess(exact, 1_000_000)
        self.assertEqual(parameter_report["architecture"], self.model.architecture_name)
        compute_report = self.model.compute_report(batch_size=8192)
        self.assertGreater(compute_report["estimated_multiply_adds_per_row"], 0)
        self.assertLess(compute_report["estimated_fp32_activation_mib"], 1024.0)
        self.assertEqual(compute_report["batch_size"], 8192)
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")

    def test_input_and_config_validation(self) -> None:
        categorical, numerical = self._tensors()
        with self.assertRaises(TypeError):
            self.model(categorical.float(), numerical)
        with self.assertRaises(TypeError):
            self.model(categorical, numerical.double())
        invalid = categorical.clone()
        invalid[0, 0] = self.model.feature_layout.categorical_cardinalities[0]
        with self.assertRaises(ValueError):
            self.model(invalid, numerical)
        nonfinite = numerical.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(categorical, nonfinite)

        with self.assertRaises(ValueError):
            PeriodicPiecewiseModelConfig(piecewise_knots=(0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            PeriodicPiecewiseModelConfig(periodic_frequencies=0)
        with self.assertRaises(TypeError):
            build_periodic_piecewise_model(
                self.preprocessor, initialization_seed=True
            )


if __name__ == "__main__":
    unittest.main()
