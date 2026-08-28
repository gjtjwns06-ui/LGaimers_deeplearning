from __future__ import annotations

import json
import unittest

import numpy as np
import torch
from torch import nn

from lg_control.model import (
    ContextGatedModelConfig,
    ContextGatedResidualModel,
    TabularFeatureLayout,
    assert_preprocessor_compatible,
    build_context_gated_model,
    validate_transformed_arrays,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from tests.test_preprocessing import make_frame


class ContextGatedModelDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()
        self.fit_frame = self.frame.iloc[:4].copy()
        self.preprocessor = DeepTabularPreprocessor().fit(self.fit_frame)
        self.arrays = self.preprocessor.transform(self.frame)
        self.config = ContextGatedModelConfig(
            tower_dim=16,
            hidden_dim=24,
            num_blocks=2,
            ffn_dim=32,
            dropout=0.2,
            entity_embedding_dropout=0.25,
            embedding_dim_cap=12,
        )
        self.model = build_context_gated_model(
            self.preprocessor,
            config=self.config,
            initialization_seed=17,
        ).eval()

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.arrays["categorical"]),
            torch.from_numpy(self.arrays["numerical"]),
        )

    def _full_vocab_case(
        self,
        config: ContextGatedModelConfig,
        *,
        initialization_seed: int = 31,
    ) -> tuple[
        DeepTabularPreprocessor,
        ContextGatedResidualModel,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build a tiny case in which both R and F are known categories."""

        preprocessor = DeepTabularPreprocessor().fit(self.frame)
        arrays = preprocessor.transform(self.frame)
        model = build_context_gated_model(
            preprocessor,
            config=config,
            initialization_seed=initialization_seed,
        ).eval()
        return (
            preprocessor,
            model,
            torch.from_numpy(arrays["categorical"]),
            torch.from_numpy(arrays["numerical"]),
        )

    @staticmethod
    def _paired_game_types(
        preprocessor: DeepTabularPreprocessor,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Return otherwise-identical rows encoded once as F and once as R."""

        game_type_position = preprocessor.categorical_columns.index("game_type")
        game_type_vocab = preprocessor.category_vocabs_["game_type"]
        f_index = game_type_vocab["F"]
        r_index = game_type_vocab["R"]
        paired_categorical = categorical[[1, 1]].clone()
        paired_numerical = numerical[[1, 1]].clone()
        paired_categorical[0, game_type_position] = f_index
        paired_categorical[1, game_type_position] = r_index
        return (
            paired_categorical,
            paired_numerical,
            game_type_position,
            f_index,
            r_index,
        )

    def test_layout_is_name_resolved_and_exhaustive(self) -> None:
        layout = self.model.feature_layout
        self.assertEqual(layout.categorical_names[6:8], ("pitcher_id", "batter_id"))
        self.assertEqual(self.model.pitcher_id_index, 6)
        self.assertEqual(self.model.batter_id_index, 7)

        categorical_groups = (
            set(layout.context_categorical_names)
            | set(layout.pitcher_categorical_names)
            | set(layout.batter_categorical_names)
        )
        numerical_groups = (
            set(layout.context_numerical_names)
            | set(layout.pitcher_numerical_names)
            | set(layout.batter_numerical_names)
        )
        self.assertEqual(categorical_groups, set(layout.categorical_names))
        self.assertEqual(numerical_groups, set(layout.numerical_names))
        self.assertTrue(
            set(layout.pitcher_numerical_names).isdisjoint(
                layout.batter_numerical_names
            )
        )
        self.assertNotIn("row_id", layout.categorical_names + layout.numerical_names)
        self.assertNotIn(
            "control_success", layout.categorical_names + layout.numerical_names
        )

    def test_array_contract_and_forward_outputs(self) -> None:
        audit = validate_transformed_arrays(
            self.model.feature_layout,
            self.arrays,
            expected_row_ids=self.frame["row_id"].to_numpy(),
        )
        self.assertEqual(audit["rows"], len(self.frame))
        self.assertFalse(audit["target_consumed_by_model"])

        categorical, numerical = self._tensors()
        with torch.inference_mode():
            auxiliary = self.model.forward_with_aux(categorical, numerical)
        self.assertEqual(tuple(auxiliary["logits"].shape), (len(self.frame),))
        self.assertTrue(torch.isfinite(auxiliary["logits"]).all().item())
        for key in ("pitcher_id_gate", "batter_id_gate"):
            self.assertEqual(tuple(auxiliary[key].shape), (len(self.frame),))
            self.assertTrue((auxiliary[key] >= 0).all().item())
            self.assertTrue((auxiliary[key] <= 1).all().item())

    def test_missing_oov_and_future_season_use_safe_fallback(self) -> None:
        categorical, numerical = self._tensors()
        self.assertNotIn(
            str(self.model.pitcher_id_index), self.model.categorical_biases
        )
        self.assertNotIn(
            str(self.model.batter_id_index), self.model.categorical_biases
        )

        categorical = categorical.clone()
        season_index = self.model.feature_layout.categorical_index("season")
        categorical[0, season_index] = 1  # future season OOV
        categorical[0, self.model.pitcher_id_index] = 1  # OOV
        categorical[1, self.model.batter_id_index] = 0  # missing

        with torch.inference_mode():
            auxiliary = self.model.forward_with_aux(categorical, numerical)
        self.assertEqual(float(auxiliary["pitcher_id_gate"][0]), 0.0)
        self.assertEqual(float(auxiliary["batter_id_gate"][1]), 0.0)
        self.assertTrue(torch.isfinite(auxiliary["logits"]).all().item())

        for position in (
            season_index,
            self.model.pitcher_id_index,
            self.model.batter_id_index,
        ):
            for reserved_index in (0, 1):
                embedded = self.model.embeddings[position].weight[reserved_index]
                self.assertTrue(torch.equal(embedded, torch.zeros_like(embedded)))

    def test_eval_is_single_row_batch_and_order_independent(self) -> None:
        categorical, numerical = self._tensors()
        with torch.inference_mode():
            batched = self.model(categorical, numerical)
            singles = torch.cat(
                [
                    self.model(categorical[index : index + 1], numerical[index : index + 1])
                    for index in range(len(categorical))
                ]
            )
            permutation = torch.tensor([4, 0, 5, 2, 1, 3])
            shuffled = self.model(
                categorical.index_select(0, permutation),
                numerical.index_select(0, permutation),
            )
            restored = torch.empty_like(shuffled)
            restored[permutation] = shuffled
        torch.testing.assert_close(batched, singles, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(batched, restored, rtol=1e-6, atol=1e-7)

    def test_seeded_factory_controls_initial_weights(self) -> None:
        first = build_context_gated_model(
            self.preprocessor, config=self.config, initialization_seed=101
        )
        second = build_context_gated_model(
            self.preprocessor, config=self.config, initialization_seed=101
        )
        third = build_context_gated_model(
            self.preprocessor, config=self.config, initialization_seed=102
        )
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        self.assertTrue(
            any(
                not torch.equal(value, third.state_dict()[name])
                for name, value in first.state_dict().items()
                if value.is_floating_point()
            )
        )

    def test_config_and_state_round_trip_is_exact(self) -> None:
        categorical, numerical = self._tensors()
        payload = json.loads(json.dumps(self.model.get_config()))
        restored = ContextGatedResidualModel.from_config(payload)
        restored.load_state_dict(self.model.state_dict(), strict=True)
        restored.eval()
        with torch.inference_mode():
            expected = self.model(categorical, numerical)
            actual = restored(categorical, numerical)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(restored.get_config(), self.model.get_config())

    def test_default_config_preserves_legacy_checkpoint_contract(self) -> None:
        legacy_model_config = {
            "tower_dim": self.config.tower_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_blocks": self.config.num_blocks,
            "ffn_dim": self.config.ffn_dim,
            "dropout": self.config.dropout,
            "entity_embedding_dropout": self.config.entity_embedding_dropout,
            "embedding_dim_cap": self.config.embedding_dim_cap,
        }
        self.assertEqual(self.config.to_dict(), legacy_model_config)
        for new_field in (
            "game_type_mode",
            "game_type_embedding_dropout",
            "regime_logit_cap",
        ):
            self.assertNotIn(new_field, self.config.to_dict())

        legacy_checkpoint_config = {
            "architecture": ContextGatedResidualModel.architecture_name,
            "feature_layout": self.model.feature_layout.to_dict(),
            "model": legacy_model_config,
        }
        restored = ContextGatedResidualModel.from_config(legacy_checkpoint_config)
        incompatibilities = restored.load_state_dict(
            self.model.state_dict(),
            strict=True,
        )
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])
        self.assertEqual(restored.get_config(), legacy_checkpoint_config)

        categorical, numerical = self._tensors()
        restored.eval()
        with torch.inference_mode():
            expected = self.model(categorical, numerical)
            actual = restored(categorical, numerical)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_game_type_config_rejects_invalid_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "game_type_mode"):
            ContextGatedModelConfig(game_type_mode="unknown")
        for invalid_dropout in (-0.01, 1.0, np.inf, np.nan):
            with self.subTest(game_type_embedding_dropout=invalid_dropout):
                with self.assertRaises((TypeError, ValueError)):
                    ContextGatedModelConfig(
                        game_type_embedding_dropout=invalid_dropout
                    )
        for invalid_cap in (0.0, -0.1, np.inf, np.nan):
            with self.subTest(regime_logit_cap=invalid_cap):
                with self.assertRaises(ValueError):
                    ContextGatedModelConfig(regime_logit_cap=invalid_cap)
        for mode in ("neutral", "bounded_history_residual"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    ValueError,
                    "game_type_embedding_dropout applies only",
                ):
                    ContextGatedModelConfig(
                        game_type_mode=mode,
                        game_type_embedding_dropout=0.1,
                    )
        self.assertEqual(
            ContextGatedModelConfig(game_type_mode=" No_Bias ").game_type_mode,
            "no_bias",
        )

    def test_no_bias_and_neutral_game_type_counterfactual_contracts(self) -> None:
        base_config = self.config.to_dict()
        no_bias_config = ContextGatedModelConfig(
            **base_config,
            game_type_mode="no_bias",
        )
        preprocessor, no_bias, categorical, numerical = self._full_vocab_case(
            no_bias_config
        )
        paired_cat, paired_num, game_type_position, f_index, r_index = (
            self._paired_game_types(preprocessor, categorical, numerical)
        )
        self.assertNotIn(str(game_type_position), no_bias.categorical_biases)
        self.assertIn(game_type_position, no_bias._context_cat_positions)
        with torch.no_grad():
            no_bias.embeddings[game_type_position].weight[f_index].fill_(3.0)
            no_bias.embeddings[game_type_position].weight[r_index].fill_(-3.0)
        with torch.inference_mode():
            no_bias_aux = no_bias.forward_with_aux(paired_cat, paired_num)
        torch.testing.assert_close(
            no_bias_aux["categorical_bias_logit"][0],
            no_bias_aux["categorical_bias_logit"][1],
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            abs(float(no_bias_aux["deep_logit"][0] - no_bias_aux["deep_logit"][1])),
            1e-8,
        )

        neutral_config = ContextGatedModelConfig(
            **base_config,
            game_type_mode="neutral",
        )
        preprocessor, neutral, categorical, numerical = self._full_vocab_case(
            neutral_config
        )
        paired_cat, paired_num, game_type_position, f_index, r_index = (
            self._paired_game_types(preprocessor, categorical, numerical)
        )
        self.assertNotIn(str(game_type_position), neutral.categorical_biases)
        self.assertNotIn(game_type_position, neutral._context_cat_positions)
        with torch.no_grad():
            neutral.embeddings[game_type_position].weight[f_index].fill_(100.0)
            neutral.embeddings[game_type_position].weight[r_index].fill_(-100.0)
        with torch.inference_mode():
            neutral_aux = neutral.forward_with_aux(paired_cat, paired_num)
        for key, values in neutral_aux.items():
            with self.subTest(auxiliary=key):
                torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)

    def test_bounded_history_residual_contract_and_strict_round_trip(self) -> None:
        cap = 0.2
        config = ContextGatedModelConfig(
            **self.config.to_dict(),
            game_type_mode="bounded_history_residual",
            regime_logit_cap=cap,
        )
        preprocessor, model, categorical, numerical = self._full_vocab_case(config)
        paired_cat, paired_num, game_type_position, f_index, r_index = (
            self._paired_game_types(preprocessor, categorical, numerical)
        )
        self.assertNotIn(str(game_type_position), model.categorical_biases)
        self.assertNotIn(game_type_position, model._context_cat_positions)
        with torch.inference_mode():
            initial = model.forward_with_aux(paired_cat, paired_num)
        self.assertTrue(
            torch.equal(
                initial["regime_residual_logit"],
                torch.zeros_like(initial["regime_residual_logit"]),
            )
        )
        torch.testing.assert_close(initial["logits"][0], initial["logits"][1], rtol=0, atol=0)

        assert isinstance(model.regime_history_norm, nn.LayerNorm)
        assert isinstance(model.regime_history_projection, nn.Linear)
        assert isinstance(model.regime_residual_head, nn.Linear)
        self.assertTrue(
            set(model._REGIME_HISTORY_NUMERICAL_NAMES).issubset(
                model.feature_layout.numerical_names
            )
        )
        self.assertTrue(
            all(
                name.startswith("asof_pitcher_")
                or name in {"pitcher_n_log1p", "pitcher_cold_start"}
                for name in model._REGIME_HISTORY_NUMERICAL_NAMES
            )
        )

        stress_cat = paired_cat[[0, 1, 0, 0]].clone()
        stress_cat[2, game_type_position] = 0  # missing
        stress_cat[3, game_type_position] = 1  # OOV
        stress_num = paired_num[[0, 1, 0, 0]].clone()
        history_positions = model._regime_history_num_index
        stress_num[:, history_positions] = 0.0
        stress_num[:, history_positions[0]] = 10.0
        with torch.no_grad():
            model.regime_history_norm.weight.fill_(1.0)
            model.regime_history_norm.bias.zero_()
            model.regime_history_projection.weight.zero_()
            model.regime_history_projection.weight[:, 0].fill_(1.0)
            model.regime_residual_head.weight.fill_(1_000.0)
            model.embeddings[game_type_position].weight[f_index].fill_(1.0)
            model.embeddings[game_type_position].weight[r_index].fill_(-1.0)
        with torch.inference_mode():
            stressed = model.forward_with_aux(stress_cat, stress_num)
        residual = stressed["regime_residual_logit"]
        self.assertTrue(torch.all(torch.abs(residual) <= cap).item())
        self.assertGreaterEqual(abs(float(residual[0])), cap * 0.99)
        self.assertGreaterEqual(abs(float(residual[1])), cap * 0.99)
        self.assertEqual(float(residual[2]), 0.0)
        self.assertEqual(float(residual[3]), 0.0)

        payload = json.loads(json.dumps(model.get_config()))
        self.assertEqual(payload["model"]["game_type_mode"], "bounded_history_residual")
        self.assertEqual(payload["model"]["regime_logit_cap"], cap)
        restored = ContextGatedResidualModel.from_config(payload)
        incompatibilities = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatibilities.missing_keys, [])
        self.assertEqual(incompatibilities.unexpected_keys, [])
        restored.eval()
        with torch.inference_mode():
            actual = restored.forward_with_aux(stress_cat, stress_num)
        for key in stressed:
            with self.subTest(round_trip_auxiliary=key):
                torch.testing.assert_close(actual[key], stressed[key], rtol=0, atol=0)

    def test_all_game_type_modes_are_deterministic_in_evaluation(self) -> None:
        configs = (
            ContextGatedModelConfig(
                **self.config.to_dict(),
                game_type_mode="no_bias",
                game_type_embedding_dropout=0.75,
            ),
            ContextGatedModelConfig(
                **self.config.to_dict(),
                game_type_mode="neutral",
            ),
            ContextGatedModelConfig(
                **self.config.to_dict(),
                game_type_mode="bounded_history_residual",
            ),
        )
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        for config in configs:
            with self.subTest(mode=config.game_type_mode):
                _, model, categorical, numerical = self._full_vocab_case(config)
                with torch.inference_mode():
                    first = model(categorical, numerical)
                    repeated = model(categorical, numerical)
                    singles = torch.cat(
                        [
                            model(
                                categorical[position : position + 1],
                                numerical[position : position + 1],
                            )
                            for position in range(len(categorical))
                        ]
                    )
                    shuffled = model(
                        categorical.index_select(0, permutation),
                        numerical.index_select(0, permutation),
                    )
                    restored = torch.empty_like(shuffled)
                    restored[permutation] = shuffled
                torch.testing.assert_close(first, repeated, rtol=0, atol=0)
                torch.testing.assert_close(first, singles, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(first, restored, rtol=1e-6, atol=1e-7)

    def test_invalid_numpy_and_tensor_inputs_fail_loudly(self) -> None:
        invalid_dtype = dict(self.arrays)
        invalid_dtype["categorical"] = self.arrays["categorical"].astype(np.int32)
        with self.assertRaisesRegex(TypeError, "int64"):
            validate_transformed_arrays(self.model.feature_layout, invalid_dtype)

        invalid_numeric = dict(self.arrays)
        invalid_numeric["numerical"] = self.arrays["numerical"].copy()
        invalid_numeric["numerical"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            validate_transformed_arrays(self.model.feature_layout, invalid_numeric)

        categorical, numerical = self._tensors()
        out_of_range = categorical.clone()
        out_of_range[0, 0] = self.model.feature_layout.categorical_cardinalities[0]
        with self.assertRaisesRegex(ValueError, "outside its vocabulary"):
            self.model(out_of_range, numerical)
        non_finite = numerical.clone()
        non_finite[0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            self.model(categorical, non_finite)
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            self.model(categorical, numerical.to(torch.float64))

    def test_architecture_is_compact_and_has_no_batch_normalization(self) -> None:
        report = self.model.parameter_report()
        self.assertLess(int(report["total_parameters"]), 1_000_000)
        self.assertEqual(
            int(report["total_parameters"]), int(report["trainable_parameters"])
        )
        self.assertFalse(
            any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in self.model.modules())
        )
        self.assertAlmostEqual(
            torch.sigmoid(self.model.output_bias).item(),
            0.5,
            places=6,
        )
        with torch.no_grad():
            self.model.output_bias.fill_(7.0)
        self.model.reset_parameters()
        self.assertAlmostEqual(
            torch.sigmoid(self.model.output_bias).item(),
            0.5,
            places=6,
        )

    def test_layout_rejects_forbidden_or_overlapping_features(self) -> None:
        layout = self.model.feature_layout
        with self.assertRaisesRegex(ValueError, "forbidden"):
            TabularFeatureLayout(
                categorical_names=(*layout.categorical_names, "row_id"),
                categorical_cardinalities=(*layout.categorical_cardinalities, 3),
                numerical_names=layout.numerical_names,
                pitcher_numerical_names=layout.pitcher_numerical_names,
                batter_numerical_names=layout.batter_numerical_names,
                preprocessor_fingerprint=layout.preprocessor_fingerprint,
            )
        overlap = layout.pitcher_numerical_names[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            TabularFeatureLayout(
                categorical_names=layout.categorical_names,
                categorical_cardinalities=layout.categorical_cardinalities,
                numerical_names=layout.numerical_names,
                pitcher_numerical_names=layout.pitcher_numerical_names,
                batter_numerical_names=(*layout.batter_numerical_names, overlap),
                preprocessor_fingerprint=layout.preprocessor_fingerprint,
            )

    def test_preprocessor_fingerprint_rejects_same_shape_different_vocab(self) -> None:
        layout = self.model.feature_layout
        assert_preprocessor_compatible(layout, self.preprocessor)

        changed = DeepTabularPreprocessor().fit(self.fit_frame)
        pitcher_vocab = changed.category_vocabs_["pitcher_id"]
        tokens = sorted(pitcher_vocab)
        self.assertGreaterEqual(len(tokens), 2)
        first, second = tokens[:2]
        pitcher_vocab[first], pitcher_vocab[second] = (
            pitcher_vocab[second],
            pitcher_vocab[first],
        )
        self.assertEqual(
            changed.category_cardinalities_, self.preprocessor.category_cardinalities_
        )
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            assert_preprocessor_compatible(layout, changed)


if __name__ == "__main__":
    unittest.main()
