from __future__ import annotations

import json
import unittest

import torch
from torch import nn

from lg_control.model import TabularFeatureLayout
from lg_control.regime_model import GatedCrossModelConfig, build_regime_model
from lg_control.regime_split_head import (
    RegimeSplitGatedCrossModel,
    RegimeSplitHeadConfig,
    build_regime_split_model,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from tests.test_preprocessing import make_frame


class RegimeSplitHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()
        # Ensure both routes occur even if the shared fixture changes.
        self.frame.loc[self.frame.index[::2], "game_type"] = "F"
        self.frame.loc[self.frame.index[1::2], "game_type"] = "R"
        self.preprocessor = DeepTabularPreprocessor().fit(self.frame)
        self.arrays = self.preprocessor.transform(self.frame)
        self.backbone_config = GatedCrossModelConfig(
            embedding_dim_cap=8,
            interaction_dim=20,
            cross_layers=2,
            num_experts=2,
            low_rank=6,
            deep_hidden_dim=24,
            deep_layers=2,
            dropout=0.0,
        )
        self.targeted = (
            "asof_pitcher_success_rate",
            "asof_pitcher_prev1_game_success_rate",
        )

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.arrays["categorical"]),
            torch.from_numpy(self.arrays["numerical"]),
        )

    def _model(self, seed: int = 17) -> RegimeSplitGatedCrossModel:
        return build_regime_split_model(
            self.preprocessor,
            config=self.backbone_config,
            hidden_dim=12,
            dropout=0.0,
            correction_cap=0.2,
            targeted_numerical_names=self.targeted,
            initialization_seed=seed,
        )

    def test_builder_resolves_exact_f_index_and_zero_init_matches_backbone(self) -> None:
        seed = 31
        base = build_regime_model(
            self.preprocessor,
            config=self.backbone_config,
            initialization_seed=seed,
        ).eval()
        split = self._model(seed).eval()
        self.assertEqual(
            split.split_head_config.f_vocabulary_index,
            self.preprocessor.category_vocabs_["game_type"]["F"],
        )
        self.assertEqual(
            split.split_head_config.game_type_categorical_index,
            split.feature_layout.categorical_index("game_type"),
        )
        split_state = split.state_dict()
        for name, value in base.state_dict().items():
            self.assertIn(name, split_state)
            self.assertTrue(torch.equal(value, split_state[name]), name)

        categorical, numerical = self._tensors()
        with torch.inference_mode():
            expected = base(categorical, numerical)
            outputs = split.forward_with_aux(categorical, numerical)
        torch.testing.assert_close(outputs["base_logits"], expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs["logits"], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(outputs["regime_correction"], torch.zeros_like(expected)))

    def test_route_is_exact_bounded_and_row_local(self) -> None:
        model = self._model().eval()
        categorical, numerical = self._tensors()
        with torch.no_grad():
            model.f_residual_head.output.bias.fill_(10.0)
            model.r_residual_head.output.bias.fill_(-10.0)
        with torch.inference_mode():
            outputs = model.forward_with_aux(categorical, numerical)
        expected_mask = categorical[:, model.feature_layout.categorical_index("game_type")].eq(
            self.preprocessor.category_vocabs_["game_type"]["F"]
        )
        self.assertTrue(torch.equal(outputs["is_f_regime"], expected_mask))
        self.assertTrue((outputs["regime_correction"][expected_mask] > 0).all().item())
        self.assertTrue((outputs["regime_correction"][~expected_mask] < 0).all().item())
        self.assertTrue((outputs["regime_correction"].abs() <= 0.2).all().item())

        permutation = torch.tensor([4, 1, 5, 0, 3, 2])
        with torch.inference_mode():
            batched = model(categorical, numerical)
            singles = torch.cat(
                [model(categorical[i : i + 1], numerical[i : i + 1]) for i in range(6)]
            )
            shuffled = model(categorical[permutation], numerical[permutation])
            restored = torch.empty_like(shuffled)
            restored[permutation] = shuffled
        torch.testing.assert_close(batched, singles, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(batched, restored, rtol=1e-6, atol=1e-7)

    def test_each_route_blocks_the_opposite_head_gradient(self) -> None:
        categorical, numerical = self._tensors()
        game_type_position = self.preprocessor.categorical_columns.index("game_type")
        f_index = self.preprocessor.category_vocabs_["game_type"]["F"]
        f_rows = categorical[:, game_type_position].eq(f_index)
        for active, active_prefix, blocked_prefix in (
            (f_rows, "f_residual_head", "r_residual_head"),
            (~f_rows, "r_residual_head", "f_residual_head"),
        ):
            with self.subTest(active_prefix=active_prefix):
                model = self._model().train()
                outputs = model(categorical, numerical)[active]
                target = torch.from_numpy(self.arrays["target"])[active]
                nn.functional.binary_cross_entropy_with_logits(outputs, target).backward()
                active_gradient = sum(
                    float(parameter.grad.abs().sum())
                    for name, parameter in model.named_parameters()
                    if name.startswith(active_prefix) and parameter.grad is not None
                )
                blocked_gradient = sum(
                    float(parameter.grad.abs().sum())
                    for name, parameter in model.named_parameters()
                    if name.startswith(blocked_prefix) and parameter.grad is not None
                )
                self.assertGreater(active_gradient, 0.0)
                self.assertEqual(blocked_gradient, 0.0)

    def test_zero_initialized_output_has_live_first_step_gradient(self) -> None:
        model = self._model().train()
        categorical, numerical = self._tensors()
        target = torch.from_numpy(self.arrays["target"])
        outputs = model.forward_with_aux(categorical, numerical)
        self.assertTrue(
            torch.equal(
                outputs["regime_correction"],
                torch.zeros_like(outputs["regime_correction"]),
            )
        )
        nn.functional.binary_cross_entropy_with_logits(outputs["logits"], target).backward()
        for head in (model.f_residual_head, model.r_residual_head):
            self.assertIsNotNone(head.output.weight.grad)
            self.assertIsNotNone(head.output.bias.grad)
            assert head.output.weight.grad is not None
            self.assertGreater(float(head.output.weight.grad.abs().sum()), 0.0)

    def test_config_state_and_parameter_report_round_trip(self) -> None:
        model = self._model(73).eval()
        payload = json.loads(json.dumps(model.get_config()))
        restored = RegimeSplitGatedCrossModel.from_config(payload)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            expected = model.forward_with_aux(categorical, numerical)
            actual = restored.forward_with_aux(categorical, numerical)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(restored.get_config(), payload)
        report = model.parameter_report()
        self.assertEqual(
            report["total_parameters"],
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertGreater(report["regime_split_head_parameters"]["F"], 0)
        self.assertEqual(
            report["regime_split_head_parameters"]["F"],
            report["regime_split_head_parameters"]["R"],
        )

    def test_strict_name_and_vocabulary_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown targeted"):
            build_regime_split_model(
                self.preprocessor,
                config=self.backbone_config,
                targeted_numerical_names=("not_a_feature",),
            )
        discovered = self.preprocessor.category_vocabs_["game_type"]["F"]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_regime_split_model(
                self.preprocessor,
                config=self.backbone_config,
                f_vocabulary_index=discovered + 1,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            RegimeSplitHeadConfig(
                game_type_categorical_index=0,
                f_vocabulary_index=2,
                targeted_numerical_names=("inning", "inning"),
            )

    def test_builder_supports_wrappers_and_requires_explicit_hidden_vocab(self) -> None:
        layout = TabularFeatureLayout.from_preprocessor(self.preprocessor)

        class VisibleBaseWrapper:
            def __init__(self, base: DeepTabularPreprocessor) -> None:
                self.base = base

            def feature_layout(self) -> TabularFeatureLayout:
                return layout

        wrapped = build_regime_split_model(
            VisibleBaseWrapper(self.preprocessor),
            config=self.backbone_config,
            initialization_seed=5,
        )
        self.assertEqual(
            wrapped.split_head_config.f_vocabulary_index,
            self.preprocessor.category_vocabs_["game_type"]["F"],
        )

        class HiddenVocabularyWrapper:
            def feature_layout(self) -> TabularFeatureLayout:
                return layout

        with self.assertRaisesRegex(ValueError, "pass f_vocabulary_index explicitly"):
            build_regime_split_model(
                HiddenVocabularyWrapper(),
                config=self.backbone_config,
            )
        explicit = build_regime_split_model(
            HiddenVocabularyWrapper(),
            config=self.backbone_config,
            f_vocabulary_index=self.preprocessor.category_vocabs_["game_type"]["F"],
        )
        self.assertEqual(
            explicit.split_head_config.f_vocabulary_index,
            self.preprocessor.category_vocabs_["game_type"]["F"],
        )


if __name__ == "__main__":
    unittest.main()
