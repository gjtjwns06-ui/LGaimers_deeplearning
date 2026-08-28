from __future__ import annotations

import json
import unittest

import numpy as np
import torch
from torch import nn

from lg_control.model import assert_preprocessor_compatible, validate_transformed_arrays
from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.regime_model import (
    AUXILIARY_TARGET_NAMES,
    GatedCrossModelConfig,
    GatedCrossRegimeModel,
    MultiTaskGatedCrossRegimeModel,
    MultiTaskHeadConfig,
    StructuredMultiTaskGatedCrossRegimeModel,
    StructuredResidualConfig,
    build_multitask_regime_model,
    build_regime_model,
    build_structured_multitask_regime_model,
)
from tests.test_preprocessing import make_frame


class GatedCrossRegimeModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()
        self.preprocessor = DeepTabularPreprocessor().fit(self.frame)
        self.arrays = self.preprocessor.transform(self.frame)
        self.config = GatedCrossModelConfig(
            embedding_dim_cap=8,
            interaction_dim=24,
            cross_layers=2,
            num_experts=3,
            low_rank=8,
            deep_hidden_dim=32,
            deep_layers=2,
            dropout=0.10,
        )
        self.model = build_regime_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=71,
        )

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.arrays["categorical"]),
            torch.from_numpy(self.arrays["numerical"]),
        )

    def test_preprocessor_contract_forward_shapes_and_rowwise_gates(self) -> None:
        audit = validate_transformed_arrays(
            self.model.feature_layout,
            self.arrays,
            expected_row_ids=self.frame["row_id"].to_numpy(),
        )
        self.assertEqual(audit["rows"], len(self.frame))
        self.assertFalse(audit["target_consumed_by_model"])
        assert_preprocessor_compatible(self.model.feature_layout, self.preprocessor)

        categorical, numerical = self._tensors()
        self.model.eval()
        with torch.inference_mode():
            auxiliary = self.model.forward_with_aux(categorical, numerical)
        self.assertEqual(tuple(auxiliary["logits"].shape), (len(self.frame),))
        self.assertEqual(
            tuple(auxiliary["expert_weights"].shape),
            (len(self.frame), self.config.cross_layers, self.config.num_experts),
        )
        torch.testing.assert_close(
            auxiliary["expert_weights"].sum(dim=2),
            torch.ones(len(self.frame), self.config.cross_layers),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertTrue((auxiliary["expert_weights"] >= 0).all().item())
        self.assertTrue((auxiliary["fusion_weight"] >= 0).all().item())
        self.assertTrue((auxiliary["fusion_weight"] <= 1).all().item())
        for value in auxiliary.values():
            self.assertTrue(torch.isfinite(value).all().item())

    def test_seeded_initialization_and_eval_are_deterministic(self) -> None:
        first = build_regime_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=101,
        ).eval()
        second = build_regime_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=101,
        ).eval()
        third = build_regime_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=102,
        ).eval()
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        self.assertTrue(
            any(
                value.is_floating_point()
                and not torch.equal(value, third.state_dict()[name])
                for name, value in first.state_dict().items()
            )
        )

        categorical, numerical = self._tensors()
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        with torch.inference_mode():
            batched = first(categorical, numerical)
            repeated = first(categorical, numerical)
            singles = torch.cat(
                [
                    first(
                        categorical[position : position + 1],
                        numerical[position : position + 1],
                    )
                    for position in range(len(categorical))
                ]
            )
            shuffled = first(
                categorical.index_select(0, permutation),
                numerical.index_select(0, permutation),
            )
            restored = torch.empty_like(shuffled)
            restored[permutation] = shuffled
        torch.testing.assert_close(batched, repeated, rtol=0, atol=0)
        torch.testing.assert_close(batched, singles, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(batched, restored, rtol=1e-6, atol=1e-7)

    def test_every_trainable_branch_receives_finite_gradients(self) -> None:
        model = build_regime_model(
            self.preprocessor,
            config=GatedCrossModelConfig(**{**self.config.to_dict(), "dropout": 0.0}),
            initialization_seed=303,
        ).train()
        categorical, numerical = self._tensors()
        target = torch.from_numpy(self.arrays["target"])
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(categorical, numerical), target
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(gradients)
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        for prefix in (
            "embeddings.",
            "input_projection.",
            "cross_network.0.experts.0.",
            "cross_network.0.expert_gate.",
            "deep_network.0.",
            "fusion_gate.",
            "cross_head.",
            "deep_head.",
            "numeric_wide_head.",
        ):
            branch_norm = sum(
                float(gradient.abs().sum())
                for name, gradient in gradients.items()
                if name.startswith(prefix) and gradient is not None
            )
            self.assertGreater(branch_norm, 0.0, prefix)
        self.assertGreater(float(gradients["output_bias"].abs().sum()), 0.0)

    def test_invalid_inputs_fail_before_embedding_or_cross_operations(self) -> None:
        categorical, numerical = self._tensors()
        with self.assertRaisesRegex(TypeError, "torch.long"):
            self.model(categorical.to(torch.int32), numerical)
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            self.model(categorical, numerical.to(torch.float64))
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            self.model(categorical[:-1], numerical)
        with self.assertRaisesRegex(ValueError, "at least one row"):
            self.model(categorical[:0], numerical[:0])

        out_of_range = categorical.clone()
        out_of_range[0, 0] = self.model.feature_layout.categorical_cardinalities[0]
        with self.assertRaisesRegex(ValueError, "outside its vocabulary"):
            self.model(out_of_range, numerical)
        non_finite = numerical.clone()
        non_finite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            self.model(categorical, non_finite)

    def test_config_and_state_round_trip_is_exact(self) -> None:
        self.model.eval()
        payload = json.loads(json.dumps(self.model.get_config()))
        restored = GatedCrossRegimeModel.from_config(payload)
        incompatibilities = restored.load_state_dict(
            self.model.state_dict(), strict=True
        )
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])
        restored.eval()
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            expected = self.model(categorical, numerical)
            actual = restored(categorical, numerical)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(restored.get_config(), payload)

    def test_reserved_categories_parameter_report_and_cpu_contract(self) -> None:
        report = self.model.parameter_report()
        exact_total = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(report["total_parameters"], exact_total)
        self.assertEqual(report["trainable_parameters"], exact_total)
        self.assertGreater(exact_total, 0)
        self.assertLess(exact_total, 1_000_000)
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")
        self.assertFalse(
            any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                for module in self.model.modules()
            )
        )
        for embedding in self.model.embeddings:
            self.assertTrue(
                torch.equal(
                    embedding.weight[:2],
                    torch.zeros_like(embedding.weight[:2]),
                )
            )

        categorical, numerical = self._tensors()
        reserved = categorical.clone()
        reserved[0].zero_()
        reserved[1].fill_(1)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(reserved, numerical)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_config_validation_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "low_rank"):
            GatedCrossModelConfig(interaction_dim=8, low_rank=9)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            GatedCrossModelConfig(embedding_dim_cap=1)
        for dropout in (-0.1, 1.0, np.inf, np.nan):
            with self.subTest(dropout=dropout):
                with self.assertRaises(ValueError):
                    GatedCrossModelConfig(dropout=dropout)
        with self.assertRaises(TypeError):
            GatedCrossModelConfig(num_experts=True)
        with self.assertRaises(ValueError):
            build_regime_model(self.preprocessor, initialization_seed=-1)

    def test_multitask_forward_keeps_main_api_and_fixed_auxiliary_order(self) -> None:
        model = build_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=MultiTaskHeadConfig(hidden_dim=12, dropout=0.05),
            initialization_seed=401,
        ).eval()
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            main_logits = model(categorical, numerical)
            outputs = model.forward_with_aux(categorical, numerical)
        self.assertEqual(tuple(main_logits.shape), (len(self.frame),))
        self.assertEqual(
            tuple(outputs["auxiliary_logits"].shape),
            (len(self.frame), 4),
        )
        torch.testing.assert_close(main_logits, outputs["logits"], rtol=0, atol=0)
        self.assertEqual(
            model.auxiliary_target_names,
            ("reverse", "middle", "ball", "strike"),
        )
        self.assertEqual(model.auxiliary_target_names, AUXILIARY_TARGET_NAMES)
        self.assertTrue(torch.isfinite(outputs["auxiliary_logits"]).all().item())

    def test_multitask_backbone_and_main_logits_match_legacy_candidate(self) -> None:
        seed = 509
        base = build_regime_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=seed,
        ).eval()
        multitask = build_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=MultiTaskHeadConfig(hidden_dim=16, dropout=0.0),
            initialization_seed=seed,
        ).eval()
        multitask_state = multitask.state_dict()
        for name, value in base.state_dict().items():
            self.assertIn(name, multitask_state)
            self.assertTrue(torch.equal(value, multitask_state[name]), name)

        categorical, numerical = self._tensors()
        with torch.inference_mode():
            expected = base(categorical, numerical)
            actual = multitask(categorical, numerical)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_multitask_gradients_reach_shared_and_auxiliary_branches(self) -> None:
        model = build_multitask_regime_model(
            self.preprocessor,
            config=GatedCrossModelConfig(**{**self.config.to_dict(), "dropout": 0.0}),
            multitask_config=MultiTaskHeadConfig(hidden_dim=16, dropout=0.0),
            initialization_seed=601,
        ).train()
        categorical, numerical = self._tensors()
        success = torch.from_numpy(self.arrays["target"])
        auxiliary_target = torch.stack(
            (
                1.0 - success,
                success,
                torch.arange(len(success)).remainder(2).to(torch.float32),
                torch.arange(len(success)).add(1).remainder(2).to(torch.float32),
            ),
            dim=1,
        )
        outputs = model.forward_with_aux(categorical, numerical)
        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["logits"], success
        ) + nn.functional.binary_cross_entropy_with_logits(
            outputs["auxiliary_logits"], auxiliary_target
        )
        loss.backward()

        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        for prefix in (
            "embeddings.",
            "input_projection.",
            "cross_network.",
            "deep_network.",
            "fusion_gate.",
            "cross_head.",
            "deep_head.",
            "numeric_wide_head.",
            "auxiliary_norm.",
            "auxiliary_hidden.",
            "auxiliary_head.",
        ):
            norm = sum(
                float(gradient.abs().sum())
                for name, gradient in gradients.items()
                if name.startswith(prefix) and gradient is not None
            )
            self.assertGreater(norm, 0.0, prefix)
        for layer_index in range(model.config.cross_layers):
            for expert_index in range(model.config.num_experts):
                prefix = f"cross_network.{layer_index}.experts.{expert_index}."
                norm = sum(
                    float(gradient.abs().sum())
                    for name, gradient in gradients.items()
                    if name.startswith(prefix) and gradient is not None
                )
                self.assertGreater(norm, 0.0, prefix)
        self.assertGreater(float(gradients["output_bias"].abs().sum()), 0.0)

    def test_multitask_determinism_and_config_state_round_trip(self) -> None:
        head_config = MultiTaskHeadConfig(hidden_dim=20, dropout=0.15)
        first = build_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=head_config,
            initialization_seed=701,
        ).eval()
        second = build_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=head_config,
            initialization_seed=701,
        ).eval()
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

        payload = json.loads(json.dumps(first.get_config()))
        self.assertEqual(
            payload["multitask"]["auxiliary_targets"],
            list(AUXILIARY_TARGET_NAMES),
        )
        restored = MultiTaskGatedCrossRegimeModel.from_config(payload)
        restored.load_state_dict(first.state_dict(), strict=True)
        restored.eval()
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            expected = first.forward_with_aux(categorical, numerical)
            actual = restored.forward_with_aux(categorical, numerical)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(restored.get_config(), payload)

        tampered = dict(payload["multitask"])
        tampered["auxiliary_targets"] = ["strike", "ball", "middle", "reverse"]
        with self.assertRaisesRegex(ValueError, "target order"):
            MultiTaskHeadConfig.from_dict(tampered)

    def test_structured_variant_starts_bit_exact_to_multitask_main(self) -> None:
        seed = 809
        head_config = MultiTaskHeadConfig(hidden_dim=16, dropout=0.0)
        multitask = build_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=head_config,
            initialization_seed=seed,
        ).eval()
        structured = build_structured_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=head_config,
            structured_config=StructuredResidualConfig(correction_cap=0.2),
            initialization_seed=seed,
        ).eval()
        structured_state = structured.state_dict()
        for name, value in multitask.state_dict().items():
            self.assertIn(name, structured_state)
            self.assertTrue(torch.equal(value, structured_state[name]), name)

        categorical, numerical = self._tensors()
        with torch.inference_mode():
            reference = multitask.forward_with_aux(categorical, numerical)
            outputs = structured.forward_with_aux(categorical, numerical)
            forward_main = structured(categorical, numerical)
        self.assertTrue(
            torch.equal(
                outputs["structured_correction"],
                torch.zeros_like(outputs["structured_correction"]),
            )
        )
        torch.testing.assert_close(outputs["base_logits"], reference["logits"], rtol=0, atol=0)
        torch.testing.assert_close(outputs["logits"], reference["logits"], rtol=0, atol=0)
        torch.testing.assert_close(
            outputs["auxiliary_logits"],
            reference["auxiliary_logits"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(forward_main, outputs["logits"], rtol=0, atol=0)

    def test_structured_correction_is_bounded_and_uses_exclusive_failure(self) -> None:
        cap = 0.15
        model = build_structured_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=MultiTaskHeadConfig(hidden_dim=12, dropout=0.0),
            structured_config=StructuredResidualConfig(correction_cap=cap),
            initialization_seed=907,
        ).eval()
        categorical, numerical = self._tensors()
        with torch.no_grad():
            model.auxiliary_head.weight.zero_()
            model.auxiliary_head.bias.copy_(
                torch.tensor([10.0, 10.0, 10.0, -10.0])
            )
            model.structured_gate.weight.zero_()
            # Ball/strike have opposite centered evidence and cancel here,
            # leaving the exclusive reverse/middle failure pressure negative.
            model.structured_gate.bias.copy_(torch.tensor([10.0, 10.0, 10.0]))
        with torch.inference_mode():
            outputs = model.forward_with_aux(categorical, numerical)
        correction = outputs["structured_correction"]
        self.assertTrue((torch.abs(correction) <= cap).all().item())
        self.assertTrue((correction < 0.0).all().item())
        torch.testing.assert_close(
            outputs["logits"],
            outputs["base_logits"] + correction,
            rtol=0,
            atol=0,
        )
        self.assertEqual(tuple(outputs["auxiliary_logits"].shape), (len(self.frame), 4))

    def test_structured_zero_initialized_gate_is_immediately_trainable(self) -> None:
        model = build_structured_multitask_regime_model(
            self.preprocessor,
            config=GatedCrossModelConfig(**{**self.config.to_dict(), "dropout": 0.0}),
            multitask_config=MultiTaskHeadConfig(hidden_dim=16, dropout=0.0),
            structured_config=StructuredResidualConfig(correction_cap=0.25),
            initialization_seed=1009,
        ).train()
        categorical, numerical = self._tensors()
        success = torch.from_numpy(self.arrays["target"])
        auxiliary_target = torch.stack(
            (
                1.0 - success,
                success,
                torch.arange(len(success)).remainder(2).to(torch.float32),
                torch.arange(len(success)).add(1).remainder(2).to(torch.float32),
            ),
            dim=1,
        )
        outputs = model.forward_with_aux(categorical, numerical)
        self.assertTrue(
            torch.equal(
                outputs["structured_correction"],
                torch.zeros_like(outputs["structured_correction"]),
            )
        )
        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["logits"], success
        ) + nn.functional.binary_cross_entropy_with_logits(
            outputs["auxiliary_logits"], auxiliary_target
        )
        loss.backward()
        gate_weight_gradient = model.structured_gate.weight.grad
        gate_bias_gradient = model.structured_gate.bias.grad
        self.assertIsNotNone(gate_weight_gradient)
        self.assertIsNotNone(gate_bias_gradient)
        assert gate_weight_gradient is not None and gate_bias_gradient is not None
        self.assertTrue(torch.isfinite(gate_weight_gradient).all().item())
        self.assertTrue(torch.isfinite(gate_bias_gradient).all().item())
        self.assertGreater(float(gate_weight_gradient.abs().sum()), 0.0)
        # Each of failure, ball, and strike residual channels is trainable.
        self.assertTrue((gate_bias_gradient.abs() > 0.0).all().item())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_structured_determinism_and_strict_config_state_round_trip(self) -> None:
        structured_config = StructuredResidualConfig(correction_cap=0.3)
        first = build_structured_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=MultiTaskHeadConfig(hidden_dim=20, dropout=0.1),
            structured_config=structured_config,
            initialization_seed=1103,
        ).eval()
        second = build_structured_multitask_regime_model(
            self.preprocessor,
            config=self.config,
            multitask_config=MultiTaskHeadConfig(hidden_dim=20, dropout=0.1),
            structured_config=structured_config,
            initialization_seed=1103,
        ).eval()
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

        payload = json.loads(json.dumps(first.get_config()))
        restored = StructuredMultiTaskGatedCrossRegimeModel.from_config(payload)
        restored.load_state_dict(first.state_dict(), strict=True)
        restored.eval()
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            expected = first.forward_with_aux(categorical, numerical)
            actual = restored.forward_with_aux(categorical, numerical)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(restored.get_config(), payload)

        tampered = dict(payload["structured"])
        tampered["exclusive_outcomes"] = ["reverse", "middle", "control_success"]
        with self.assertRaisesRegex(ValueError, "exclusive_outcomes"):
            StructuredResidualConfig.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
