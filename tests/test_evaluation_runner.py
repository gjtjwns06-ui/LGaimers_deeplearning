from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lg_control.evaluation import EvaluationConfig, probability_metrics
from lg_control.learning import (
    OptimizationProcedure,
    SeedSpec,
    refit_outer_fixed_epochs,
    select_epoch_on_inner_stop,
)
from lg_control.learning_artifacts import save_learning_checkpoint
from lg_control.model import ContextGatedModelConfig
from lg_control.validation import build_validation_plan
from scripts.run_validation_evaluation import (
    run_validation_evaluation,
    verify_evaluation_artifacts,
)
from tests.test_evaluation import make_balanced_learning_frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class ValidationEvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.input_path = cls.root / "train.csv"
        cls.learning_dir = cls.root / "learning"
        cls.output_dir = cls.root / "evaluation"

        source = make_balanced_learning_frame()
        source.to_csv(cls.input_path, index=False, encoding="utf-8")
        # Train against the same CSV round trip consumed by the runner so all
        # checkpoint source and fold identity hashes describe its exact input.
        frame = pd.read_csv(cls.input_path, low_memory=False)
        folds = build_validation_plan(frame).development_folds
        model_config = ContextGatedModelConfig(
            tower_dim=4,
            hidden_dim=8,
            num_blocks=1,
            ffn_dim=8,
            dropout=0.0,
            entity_embedding_dropout=0.0,
            embedding_dim_cap=4,
        )
        optimization = OptimizationProcedure(
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
        seeds = SeedSpec(
            initialization_seed=211,
            training_seed=212,
            loader_seed=213,
        )
        fold_records: list[dict[str, object]] = []
        for fold in folds:
            selection = select_epoch_on_inner_stop(
                frame,
                fold,
                model_config=model_config,
                optimization=optimization,
                seeds=seeds,
                device="cpu",
            )
            fitted = refit_outer_fixed_epochs(
                frame,
                fold,
                selection,
                model_config=model_config,
                optimization=optimization,
                seeds=seeds,
                device="cpu",
            )
            fold_dir = cls.learning_dir / fold.name
            selection_path = fold_dir / "selection_record.json"
            checkpoint_dir = fold_dir / "checkpoint"
            _write_json(selection_path, selection.to_dict())
            save_learning_checkpoint(
                checkpoint_dir,
                preprocessor=fitted.preprocessor,
                model=fitted.model,
                training_record=fitted.training_record,
            )
            fold_records.append(
                {
                    "fold": fold.name,
                    "reserved_outer_validation_season": fold.validation_season,
                    "selected_epoch": selection.selected_epoch,
                    "selection_record_sha256": _sha256(selection_path),
                    "checkpoint_manifest_sha256": _sha256(
                        checkpoint_dir / "manifest.json"
                    ),
                    "outer_validation_rows_accessed": False,
                }
            )

        _write_json(
            cls.learning_dir / "manifest.json",
            {
                "input": {
                    "sha256": _sha256(cls.input_path),
                    "size_bytes": cls.input_path.stat().st_size,
                    "rows": len(frame),
                },
                "folds": fold_records,
                "stage_boundary": {
                    "outer_validation_metrics_computed": False,
                    "outer_validation_predictions_created": False,
                    "hyperparameters_tuned": False,
                    "anchor_2024_trained_or_evaluated": False,
                    "calibration_or_ensemble_executed": False,
                    "inference_or_submission_executed": False,
                },
            },
        )
        cls.manifest = run_validation_evaluation(
            input_path=cls.input_path,
            learning_dir=cls.learning_dir,
            output_dir=cls.output_dir,
            config=EvaluationConfig(
                eval_batch_size=2,
                num_workers=0,
                cpu_threads=1,
            ),
            device="cpu",
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_full_run_verify_inventory_oof_and_metrics_contract(self) -> None:
        verified = verify_evaluation_artifacts(self.output_dir)
        self.assertEqual(verified, self.manifest)
        expected_inventory = {
            "aggregate_metrics.json",
            "development_oof_predictions.csv",
            "fold_metrics.csv",
            "slice_metrics.csv",
            "rolling_origin_2022/evaluation_record.json",
            "rolling_origin_2022/predictions.csv",
            "rolling_origin_2023/evaluation_record.json",
            "rolling_origin_2023/predictions.csv",
        }
        self.assertEqual(set(verified["files_sha256"]), expected_inventory)
        actual_inventory = {
            path.relative_to(self.output_dir).as_posix()
            for path in self.output_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        self.assertEqual(actual_inventory, expected_inventory)

        fold_predictions = [
            pd.read_csv(
                self.output_dir
                / f"rolling_origin_{season}"
                / "predictions.csv",
                float_precision="round_trip",
            )
            for season in (2022, 2023)
        ]
        expected_oof = pd.concat(fold_predictions, ignore_index=True)
        saved_oof = pd.read_csv(
            self.output_dir / "development_oof_predictions.csv",
            float_precision="round_trip",
        )
        pd.testing.assert_frame_equal(
            saved_oof,
            expected_oof,
            check_dtype=False,
            check_exact=True,
        )
        recomputed = probability_metrics(
            saved_oof["y_true"].to_numpy(dtype=np.float64),
            saved_oof["prediction"].to_numpy(dtype=np.float64),
        )
        aggregate = json.loads(
            (self.output_dir / "aggregate_metrics.json").read_text(encoding="utf-8")
        )
        for name, expected in recomputed.items():
            with self.subTest(metric=name):
                if expected is None:
                    self.assertIsNone(aggregate["row_pooled"][name])
                elif isinstance(expected, int):
                    self.assertEqual(aggregate["row_pooled"][name], expected)
                else:
                    self.assertAlmostEqual(
                        float(aggregate["row_pooled"][name]),
                        float(expected),
                        places=14,
                    )

    def test_overwrite_and_artifact_tampering_are_rejected(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            run_validation_evaluation(
                input_path=self.input_path,
                learning_dir=self.learning_dir,
                output_dir=self.output_dir,
                config=EvaluationConfig(
                    eval_batch_size=2,
                    num_workers=0,
                    cpu_threads=1,
                ),
                device="cpu",
                verbose=False,
            )

        prediction_path = (
            self.output_dir / "rolling_origin_2022" / "predictions.csv"
        )
        prediction_bytes = prediction_path.read_bytes()
        try:
            prediction_path.write_bytes(prediction_bytes + b"\n")
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                verify_evaluation_artifacts(self.output_dir)
        finally:
            prediction_path.write_bytes(prediction_bytes)
        verify_evaluation_artifacts(self.output_dir)

        manifest_path = self.output_dir / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        try:
            tampered = json.loads(manifest_bytes.decode("utf-8"))
            tampered["files_sha256"].pop("fold_metrics.csv")
            _write_json(manifest_path, tampered)
            with self.assertRaisesRegex(ValueError, "hash inventory"):
                verify_evaluation_artifacts(self.output_dir)
        finally:
            manifest_path.write_bytes(manifest_bytes)
        verify_evaluation_artifacts(self.output_dir)


if __name__ == "__main__":
    unittest.main()
