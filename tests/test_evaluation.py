from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lg_control.evaluation import (
    REQUIRED_SLICE_FAMILIES,
    EvaluationConfig,
    aggregate_development_evaluations,
    evaluate_outer_validation,
    probability_metrics,
)
from lg_control.learning import (
    OptimizationProcedure,
    SeedSpec,
    refit_outer_fixed_epochs,
    select_epoch_on_inner_stop,
)
from lg_control.learning_artifacts import (
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from lg_control.model import ContextGatedModelConfig
from lg_control.validation import ValidationDesignError, build_validation_plan
from tests.test_learning import make_learning_frame


def make_balanced_learning_frame() -> pd.DataFrame:
    """Keep the tiny official-schema fixture while balancing every season."""

    frame = make_learning_frame()
    for season in sorted(frame["season"].unique()):
        positions = np.flatnonzero(frame["season"].to_numpy() == season)
        frame.iloc[positions, frame.columns.get_loc("control_success")] = (
            np.arange(len(positions), dtype=np.int64) % 2
        )
    return frame


class OuterValidationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = make_balanced_learning_frame()
        cls.model_config = ContextGatedModelConfig(
            tower_dim=4,
            hidden_dim=8,
            num_blocks=1,
            ffn_dim=8,
            dropout=0.0,
            entity_embedding_dropout=0.0,
            embedding_dim_cap=4,
        )
        cls.optimization = OptimizationProcedure(
            max_selection_epochs=1,
            scheduler_horizon_epochs=2,
            batch_size=8,
            eval_batch_size=8,
            learning_rate=1e-3,
            weight_decay=1e-5,
            brier_weight=0.5,
            patience=1,
            min_delta=0.0,
            gradient_clip_norm=5.0,
            num_workers=0,
            use_amp=False,
            cpu_threads=1,
        )
        cls.seeds = SeedSpec(
            initialization_seed=101,
            training_seed=102,
            loader_seed=103,
        )
        cls.evaluation_config = EvaluationConfig(
            eval_batch_size=2,
            num_workers=0,
            cpu_threads=1,
        )
        cls.temporary = tempfile.TemporaryDirectory()
        cls.checkpoint_root = Path(cls.temporary.name)
        cls.folds = build_validation_plan(cls.frame).development_folds
        cls.checkpoints: dict[int, Path] = {}
        cls.evaluations = {}

        for fold in cls.folds:
            selection = select_epoch_on_inner_stop(
                cls.frame,
                fold,
                model_config=cls.model_config,
                optimization=cls.optimization,
                seeds=cls.seeds,
                device="cpu",
            )
            fitted = refit_outer_fixed_epochs(
                cls.frame,
                fold,
                selection,
                model_config=cls.model_config,
                optimization=cls.optimization,
                seeds=cls.seeds,
                device="cpu",
            )
            checkpoint = cls.checkpoint_root / fold.name
            save_learning_checkpoint(
                checkpoint,
                preprocessor=fitted.preprocessor,
                model=fitted.model,
                training_record=fitted.training_record,
            )
            cls.checkpoints[fold.validation_season] = checkpoint
            cls.evaluations[fold.validation_season] = evaluate_outer_validation(
                cls.frame,
                fold,
                checkpoint,
                config=cls.evaluation_config,
                device="cpu",
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_predictions_ignore_validation_target_and_preserve_row_contract(self) -> None:
        fold = self.folds[0]
        original = self.evaluations[fold.validation_season]
        changed = self.frame.copy(deep=True)
        validation_mask = changed["season"].eq(fold.validation_season)
        changed.loc[validation_mask, "control_success"] = (
            1 - changed.loc[validation_mask, "control_success"].astype(np.int64)
        )
        changed_fold = build_validation_plan(changed).development_folds[0]
        reevaluated = evaluate_outer_validation(
            changed,
            changed_fold,
            self.checkpoints[fold.validation_season],
            config=self.evaluation_config,
            device="cpu",
        )

        np.testing.assert_array_equal(
            original.predictions["row_id"].to_numpy(dtype=object),
            fold.validation_row_ids,
        )
        np.testing.assert_array_equal(
            reevaluated.predictions["row_id"].to_numpy(dtype=object),
            fold.validation_row_ids,
        )
        np.testing.assert_array_equal(
            original.predictions["prediction"].to_numpy(),
            reevaluated.predictions["prediction"].to_numpy(),
        )
        self.assertFalse(
            np.array_equal(
                original.predictions["y_true"].to_numpy(),
                reevaluated.predictions["y_true"].to_numpy(),
            )
        )
        probabilities = original.predictions["prediction"].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((0.0 <= probabilities) & (probabilities <= 1.0)).all())
        self.assertEqual(len(probabilities), fold.validation_row_count)
        self.assertTrue(original.predictions["row_id"].is_unique)
        self.assertIs(
            original.evaluation_record["target_removed_before_preprocessing"], True
        )
        self.assertIs(original.evaluation_record["target_consumed_by_model"], False)
        self.assertIs(original.evaluation_record["prediction_order_preserved"], True)

    def test_checkpoint_and_fold_identity_are_strict(self) -> None:
        fold_2022, fold_2023 = self.folds
        with self.assertRaisesRegex(ValueError, "different outer fold"):
            evaluate_outer_validation(
                self.frame,
                fold_2023,
                self.checkpoints[2022],
                config=self.evaluation_config,
                device="cpu",
            )

        reordered = self.frame.iloc[::-1].copy()
        with self.assertRaises((ValueError, ValidationDesignError)):
            evaluate_outer_validation(
                reordered,
                fold_2022,
                self.checkpoints[2022],
                config=self.evaluation_config,
                device="cpu",
            )

        changed_identity = self.frame.copy(deep=True)
        changed_identity.loc[changed_identity.index[0], "row_id"] = "NEW_ROW_ID"
        changed_fold = build_validation_plan(changed_identity).development_folds[0]
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluate_outer_validation(
                changed_identity,
                changed_fold,
                self.checkpoints[2022],
                config=self.evaluation_config,
                device="cpu",
            )

    def test_evaluation_does_not_mutate_checkpoint_model_or_preprocessor(self) -> None:
        fold = self.folds[0]
        checkpoint = self.checkpoints[fold.validation_season]
        files_before = {
            path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()
        }
        loaded_before = load_learning_checkpoint(checkpoint, device="cpu")
        preprocessor_before = loaded_before.preprocessor.to_state_dict()
        tensors_before = {
            name: tensor.detach().clone()
            for name, tensor in loaded_before.model.state_dict().items()
        }

        result = evaluate_outer_validation(
            self.frame,
            fold,
            checkpoint,
            config=self.evaluation_config,
            device="cpu",
        )

        files_after = {
            path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()
        }
        self.assertEqual(files_before, files_after)
        loaded_after = load_learning_checkpoint(checkpoint, device="cpu")
        self.assertEqual(
            preprocessor_before,
            loaded_after.preprocessor.to_state_dict(),
        )
        for name, tensor in loaded_after.model.state_dict().items():
            self.assertTrue(torch.equal(tensors_before[name], tensor), name)
        self.assertIs(result.evaluation_record["model_state_unchanged"], True)
        self.assertIs(result.evaluation_record["preprocessor_state_unchanged"], True)
        self.assertEqual(
            result.evaluation_record["loaded_model_state_sha256_before"],
            result.evaluation_record["loaded_model_state_sha256_after"],
        )

    def test_required_slices_are_complete_and_empty_slices_have_null_metrics(self) -> None:
        result = self.evaluations[2022]
        slices = result.slice_metrics
        self.assertEqual(set(slices["slice"]), set(REQUIRED_SLICE_FAMILIES))
        expected_values = {
            "overall": {"all"},
            "game_type": {"F", "R"},
            "entity_status": {
                "both_seen",
                "pitcher_only",
                "batter_only",
                "neither_seen",
            },
            "recent_status": {"recent_missing", "recent_observed"},
            "pitcher_history": {
                "cold_start_0",
                "low_history_1_49",
                "history_50_plus",
                "history_missing",
            },
            "batter_history": {
                "cold_start_0",
                "low_history_1_49",
                "history_50_plus",
                "history_missing",
            },
        }
        metric_columns = (
            "target_rate",
            "prediction_mean",
            "prediction_std",
            "prediction_min",
            "prediction_max",
            "baseline_brier",
            "brier",
            "brier_skill",
            "competition_score_analogue",
            "log_loss",
            "ece_15",
        )
        for family, values in expected_values.items():
            part = slices.loc[slices["slice"].eq(family)]
            self.assertEqual(set(part["value"]), values)
            self.assertEqual(int(part["rows"].sum()), len(result.predictions))
            self.assertAlmostEqual(float(part["share"].sum()), 1.0)

        empty = slices.loc[slices["rows"].eq(0)]
        self.assertGreater(len(empty), 0)
        self.assertTrue(empty.loc[:, metric_columns].isna().all().all())
        nonempty = slices.loc[slices["rows"].gt(0)]
        self.assertTrue(nonempty["brier"].notna().all())

    def test_development_aggregate_recomputes_pooled_metrics_from_oof_rows(self) -> None:
        aggregate = aggregate_development_evaluations(
            [self.evaluations[2023], self.evaluations[2022]]
        )
        self.assertEqual(
            aggregate.fold_metrics["validation_season"].tolist(), [2022, 2023]
        )
        expected_ids = np.concatenate(
            [self.folds[0].validation_row_ids, self.folds[1].validation_row_ids]
        )
        np.testing.assert_array_equal(
            aggregate.oof_predictions["row_id"].to_numpy(dtype=object),
            expected_ids,
        )
        oof = aggregate.oof_predictions
        recomputed = probability_metrics(
            oof["y_true"].to_numpy(dtype=np.float64),
            oof["prediction"].to_numpy(dtype=np.float64),
        )
        pooled = aggregate.aggregate_metrics["row_pooled"]
        self.assertEqual(pooled.keys(), recomputed.keys())
        for name, expected in recomputed.items():
            with self.subTest(metric=name):
                if expected is None:
                    self.assertIsNone(pooled[name])
                elif isinstance(expected, int):
                    self.assertEqual(pooled[name], expected)
                else:
                    self.assertAlmostEqual(float(pooled[name]), float(expected), places=15)
        self.assertAlmostEqual(
            aggregate.aggregate_metrics["row_weighted_fold_brier"],
            recomputed["brier"],
            places=15,
        )
        self.assertIs(aggregate.aggregate_metrics["anchor_2024_included"], False)
        self.assertIsNone(aggregate.aggregate_metrics["official_test_score"])

    def test_anchor_and_recent_only_diagnostic_are_rejected(self) -> None:
        plan = build_validation_plan(self.frame, include_recent_only=True)
        diagnostic = plan.recent_only_diagnostic
        assert diagnostic is not None
        for fold in (plan.anchor_fold, diagnostic):
            with self.subTest(fold=fold.name):
                with self.assertRaisesRegex(
                    ValidationDesignError, "development folds 2022 and 2023"
                ):
                    evaluate_outer_validation(
                        self.frame,
                        fold,
                        self.checkpoints[2023],
                        config=self.evaluation_config,
                        device="cpu",
                    )


if __name__ == "__main__":
    unittest.main()
