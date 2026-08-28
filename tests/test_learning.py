from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
from lg_control.validation import build_validation_plan
from tests.test_preprocessing import make_frame as make_preprocessing_frame
from tests.test_validation import make_validation_frame


def make_learning_frame() -> pd.DataFrame:
    """Return a tiny, semantically valid frame with the official season layout."""

    frame = make_validation_frame()
    valid_rows = make_preprocessing_frame()
    protected = {"row_id", "season", "control_success"}
    for column in frame.columns:
        if column not in protected:
            frame[column] = [
                valid_rows.iloc[position % len(valid_rows)][column]
                for position in range(len(frame))
            ]
    frame["control_success"] = np.arange(len(frame), dtype=np.int64) % 2
    return frame


def mutate_outer_validation(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Change only reserved outer rows while retaining a valid input schema."""

    changed = frame.copy(deep=True)
    mask = changed["season"].eq(season)
    count = int(mask.sum())
    changed.loc[mask, "pitcher_id"] = np.arange(900_000, 900_000 + count)
    changed.loc[mask, "batter_id"] = np.arange(910_000, 910_000 + count)
    changed.loc[mask, "pitcher_team_id"] = np.arange(920_000, 920_000 + count)
    changed.loc[mask, "control_success"] = (
        1 - changed.loc[mask, "control_success"].astype(np.int64)
    )
    return changed


class LearningStageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = make_learning_frame()
        cls.config = ContextGatedModelConfig(
            tower_dim=4,
            hidden_dim=8,
            num_blocks=1,
            ffn_dim=8,
            dropout=0.10,
            entity_embedding_dropout=0.10,
            embedding_dim_cap=4,
        )
        cls.procedure = OptimizationProcedure(
            max_selection_epochs=2,
            scheduler_horizon_epochs=7,
            batch_size=8,
            eval_batch_size=8,
            learning_rate=1e-3,
            weight_decay=1e-5,
            brier_weight=0.5,
            patience=2,
            min_delta=0.0,
            gradient_clip_norm=5.0,
            num_workers=0,
            use_amp=False,
            cpu_threads=1,
        )
        cls.seeds = SeedSpec(
            initialization_seed=17,
            training_seed=18,
            loader_seed=19,
        )

    def _select(self, frame: pd.DataFrame, fold_index: int = 0):
        fold = build_validation_plan(frame).development_folds[fold_index]
        record = select_epoch_on_inner_stop(
            frame,
            fold,
            model_config=self.config,
            optimization=self.procedure,
            seeds=self.seeds,
            device="cpu",
        )
        return fold, record

    def _refit(self, frame: pd.DataFrame, fold, record, *, seeds=None):
        return refit_outer_fixed_epochs(
            frame,
            fold,
            selection_record=record,
            model_config=self.config,
            optimization=self.procedure,
            seeds=self.seeds if seeds is None else seeds,
            device="cpu",
        )

    def test_inner_selection_ignores_outer_validation_and_keeps_fixed_horizon(self) -> None:
        fold, original = self._select(self.frame)
        changed_frame = mutate_outer_validation(self.frame, fold.validation_season)
        _, changed = self._select(changed_frame)

        self.assertEqual(original.to_dict(), changed.to_dict())
        self.assertGreaterEqual(original.selected_epoch, 1)
        self.assertLessEqual(
            original.selected_epoch, self.procedure.max_selection_epochs
        )
        self.assertEqual(
            original.scheduler_horizon_epochs,
            self.procedure.scheduler_horizon_epochs,
        )
        self.assertEqual(original.fit_rows, 4)
        self.assertEqual(original.stop_rows, 2)
        serialized = json.dumps(original.to_dict(), sort_keys=True).lower()
        self.assertNotIn("prediction", serialized)
        self.assertNotIn("model_state", serialized)

    def test_outer_refit_does_not_materialize_outer_values_and_is_deterministic(self) -> None:
        fold, selection = self._select(self.frame)
        changed_frame = mutate_outer_validation(self.frame, fold.validation_season)
        changed_fold = build_validation_plan(changed_frame).development_folds[0]

        first = self._refit(self.frame, fold, selection)
        second = self._refit(changed_frame, changed_fold, selection)

        self.assertEqual(
            first.preprocessor.to_state_dict(), second.preprocessor.to_state_dict()
        )
        for name, tensor in first.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.model.state_dict()[name]), name)

        for result in (first, second):
            record = result.training_record
            self.assertIs(record["outer_validation_rows_accessed"], False)
            self.assertIs(record["contains_outer_validation_predictions"], False)
            self.assertIs(record["contains_outer_validation_metrics"], False)
            self.assertEqual(record["selected_epoch"], selection.selected_epoch)
            self.assertEqual(
                record["scheduler_horizon_epochs"],
                self.procedure.scheduler_horizon_epochs,
            )
            self.assertEqual(len(record["history"]), selection.selected_epoch)
        self.assertEqual(
            first.training_record["history"], second.training_record["history"]
        )

    def test_same_explicit_seeds_reproduce_outer_model_exactly(self) -> None:
        fold, selection = self._select(self.frame, fold_index=1)
        first = self._refit(self.frame, fold, selection)
        second = self._refit(self.frame, fold, selection)

        self.assertEqual(
            first.preprocessor.to_state_dict(), second.preprocessor.to_state_dict()
        )
        self.assertEqual(
            first.training_record["history"], second.training_record["history"]
        )
        for name, tensor in first.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.model.state_dict()[name]), name)

    def test_anchor_and_recent_only_diagnostic_are_rejected(self) -> None:
        plan = build_validation_plan(self.frame, include_recent_only=True)
        for fold in (plan.anchor_fold, plan.recent_only_diagnostic):
            assert fold is not None
            with self.subTest(operation="select", fold=fold.name):
                with self.assertRaises(ValueError):
                    select_epoch_on_inner_stop(
                        self.frame,
                        fold,
                        model_config=self.config,
                        optimization=self.procedure,
                        seeds=self.seeds,
                        device="cpu",
                    )

        development_fold, selection = self._select(self.frame)
        self.assertIsNotNone(development_fold)
        for fold in (plan.anchor_fold, plan.recent_only_diagnostic):
            assert fold is not None
            with self.subTest(operation="refit", fold=fold.name):
                with self.assertRaises(ValueError):
                    self._refit(self.frame, fold, selection)

    def test_checkpoint_round_trip_tamper_detection_and_no_overwrite(self) -> None:
        fold, selection = self._select(self.frame)
        fitted = self._refit(self.frame, fold, selection)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "fold_2022"
            manifest_path = save_learning_checkpoint(
                checkpoint,
                preprocessor=fitted.preprocessor,
                model=fitted.model,
                training_record=fitted.training_record,
            )
            self.assertEqual(manifest_path, checkpoint / "manifest.json")

            loaded = load_learning_checkpoint(checkpoint, device="cpu")
            self.assertEqual(
                loaded.preprocessor.to_state_dict(),
                fitted.preprocessor.to_state_dict(),
            )
            self.assertEqual(loaded.model.get_config(), fitted.model.get_config())
            self.assertEqual(loaded.training_record, fitted.training_record)
            self.assertIs(
                loaded.manifest["contains_outer_validation_predictions"], False
            )
            self.assertIs(loaded.manifest["contains_outer_validation_metrics"], False)
            for name, tensor in fitted.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.model.state_dict()[name]), name)

            with self.assertRaises(FileExistsError):
                save_learning_checkpoint(
                    checkpoint,
                    preprocessor=fitted.preprocessor,
                    model=fitted.model,
                    training_record=fitted.training_record,
                )

            invalid_record = dict(fitted.training_record)
            invalid_record["outer_validation_rows_accessed"] = True
            with self.assertRaisesRegex(ValueError, "must set"):
                save_learning_checkpoint(
                    Path(temporary) / "invalid_scope",
                    preprocessor=fitted.preprocessor,
                    model=fitted.model,
                    training_record=invalid_record,
                )

            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(
                json.dumps({**manifest_payload, "files_sha256": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash inventory"):
                load_learning_checkpoint(checkpoint, device="cpu")
            manifest_path.write_text(
                json.dumps(manifest_payload, sort_keys=True), encoding="utf-8"
            )

            record_path = checkpoint / "training_record.json"
            record_path.write_text(
                record_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_learning_checkpoint(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
