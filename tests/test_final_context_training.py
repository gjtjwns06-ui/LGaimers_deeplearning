from __future__ import annotations

import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from lg_control.anchor_confirmation import ANCHOR_SEEDS, PRODUCTION_CAMPAIGN_LOCK
from lg_control import final_context_training
from lg_control.final_context_training import (
    EXPECTED_ANCHOR_CONFIRMATION_SHA256,
    EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256,
    EXPECTED_ANCHOR_METRICS_FILE_SHA256,
    EXPECTED_ANCHOR_METRICS_SHA256,
    EXPECTED_ANCHOR_READY_SHA256,
    EXPECTED_ANCHOR_READY_FILE_SHA256,
    EXPECTED_ANCHOR_PREDICTION_FILE_SHA256,
    EXPECTED_FULL_SOURCE_SHA256,
    EXPECTED_PREDICTIONS_COMMIT_SHA256,
    EXPECTED_PROMOTION_LOCK_SHA256,
    VerifiedFinalAnchor,
    _save_final_artifact,
    _train_fresh_model,
    load_final_context_artifact,
    predict_final_context,
)
from lg_control.learning import OptimizationProcedure, row_id_sha256
from lg_control.regime_tuning import _sha256_file
from lg_control.regime_tuning import (
    _payload_with_hash,
    _read_json,
    _serialize_json,
)
from lg_control.schema import SchemaError
from tests.test_preprocessing import make_frame


def synthetic_frame() -> pd.DataFrame:
    base = make_frame()
    copies: list[pd.DataFrame] = []
    for repeat in range(2):
        item = base.copy(deep=True)
        item["row_id"] = [f"FINAL_{repeat}_{index}" for index in range(len(item))]
        copies.append(item)
    frame = pd.concat(copies, ignore_index=True)
    frame["season"] = np.repeat(np.arange(2019, 2025), 2)
    frame["control_success"] = np.arange(len(frame), dtype=np.int64) % 2
    return frame


def fixed_optimization() -> OptimizationProcedure:
    return OptimizationProcedure(
        max_selection_epochs=8,
        scheduler_horizon_epochs=8,
        batch_size=4096,
        eval_batch_size=16384,
        learning_rate=0.001,
        weight_decay=0.00001,
        brier_weight=0.5,
        patience=3,
        min_delta=0.000001,
        gradient_clip_norm=5.0,
        num_workers=0,
        use_amp=False,
        cpu_threads=8,
    )


def verified_anchor() -> VerifiedFinalAnchor:
    spec = PRODUCTION_CAMPAIGN_LOCK.selected_candidate_spec()
    return VerifiedFinalAnchor(
        anchor_directory="synthetic",
        anchor_ready_sha256=EXPECTED_ANCHOR_READY_SHA256,
        anchor_ready_file_sha256=EXPECTED_ANCHOR_READY_FILE_SHA256,
        confirmation_sha256=EXPECTED_ANCHOR_CONFIRMATION_SHA256,
        confirmation_file_sha256=EXPECTED_ANCHOR_CONFIRMATION_FILE_SHA256,
        metrics_sha256=EXPECTED_ANCHOR_METRICS_SHA256,
        metrics_file_sha256=EXPECTED_ANCHOR_METRICS_FILE_SHA256,
        predictions_commit_sha256=EXPECTED_PREDICTIONS_COMMIT_SHA256,
        prediction_file_sha256=EXPECTED_ANCHOR_PREDICTION_FILE_SHA256,
        promotion_lock_sha256=EXPECTED_PROMOTION_LOCK_SHA256,
        selected_candidate_id="c2_neutral",
        selected_epoch=1,
        selected_model_config=dict(spec["model_config"]),
        optimization=fixed_optimization().to_dict(),
        seeds=ANCHOR_SEEDS.to_dict(),
        expected_full_source_sha256=EXPECTED_FULL_SOURCE_SHA256,
        anchor_rows=2,
        gates=final_context_training._expected_anchor_gates(),
    )


def synthetic_contract(frame: pd.DataFrame) -> ExitStack:
    counts = {str(year): 2 for year in range(2019, 2025)}
    stack = ExitStack()
    stack.enter_context(
        mock.patch.object(final_context_training, "EXPECTED_FULL_ROWS", len(frame))
    )
    stack.enter_context(
        mock.patch.object(
            final_context_training,
            "EXPECTED_FULL_ROW_ID_SHA256",
            row_id_sha256(frame["row_id"].to_numpy(dtype=object, copy=False)),
        )
    )
    stack.enter_context(
        mock.patch.object(final_context_training, "EXPECTED_SEASON_COUNTS", counts)
    )
    stack.enter_context(
        mock.patch.object(
            final_context_training,
            "EXPECTED_FULL_TARGET_RATE",
            float(frame["control_success"].mean()),
        )
    )
    return stack


def train_and_save(directory: Path, frame: pd.DataFrame):
    anchor = verified_anchor()
    source = {
        "path": "synthetic.csv",
        "sha256": EXPECTED_FULL_SOURCE_SHA256,
        "rows": len(frame),
        "row_id_sha256": row_id_sha256(
            frame["row_id"].to_numpy(dtype=object, copy=False)
        ),
        "seasons": list(range(2019, 2025)),
        "season_counts": {str(year): 2 for year in range(2019, 2025)},
        "target_rate": float(frame["control_success"].mean()),
    }
    preprocessor, model, record = _train_fresh_model(
        frame,
        anchor=anchor,
        source_provenance=source,
        verbose=False,
    )
    _save_final_artifact(
        directory,
        preprocessor=preprocessor,
        model=model,
        training_record=record,
        anchor=anchor,
        source_provenance=source,
    )
    return frame


class FinalContextContractTests(unittest.TestCase):
    def test_fresh_single_model_fixed_epoch_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                loaded = load_final_context_artifact(root, verify_code=True)
            record = loaded.training_record
            self.assertEqual(record["selected_epoch"], 1)
            self.assertEqual(len(record["history"]), 1)
            self.assertEqual(record["model_count"], 1)
            self.assertTrue(record["fresh_model"])
            self.assertFalse(record["warm_start"])
            self.assertFalse(record["validation_performed"])
            self.assertFalse(record["early_stopping_performed"])
            self.assertFalse(record["recency_or_group_weighting"])
            self.assertFalse(record["trackman_used"])
            self.assertFalse(record["calibration_used"])
            self.assertFalse(record["ensemble_used"])
            self.assertEqual(record["train_rows"], len(frame))
            with self.assertRaises(FileExistsError):
                train_and_save(root, frame)

    def test_prediction_is_target_free_order_preserving_and_row_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                loaded = load_final_context_artifact(root, verify_code=True)
                features = frame.drop(columns=["control_success"])
                batch = predict_final_context(loaded, features, batch_size=7)
            self.assertEqual(batch.shape, (len(features),))
            self.assertTrue(np.isfinite(batch).all())
            self.assertTrue(((batch >= 0.0) & (batch <= 1.0)).all())

            single = np.array(
                [
                    predict_final_context(
                        loaded, features.iloc[[index]], batch_size=1
                    )[0]
                    for index in range(len(features))
                ]
            )
            np.testing.assert_allclose(batch, single, rtol=0.0, atol=1e-7)

            permutation = np.array([4, 1, 9, 0, 11, 2, 8, 3, 10, 5, 7, 6])
            shuffled = features.iloc[permutation].copy()
            shuffled["pitcher_id"] = shuffled["pitcher_id"].astype(object)
            shuffled.iloc[0, shuffled.columns.get_loc("pitcher_id")] = 999_999_999
            shuffled_probability = predict_final_context(
                loaded, shuffled, batch_size=5
            )
            self.assertTrue(np.isfinite(shuffled_probability).all())
            unchanged = permutation[1:]
            np.testing.assert_allclose(
                shuffled_probability[1:], batch[unchanged], rtol=0.0, atol=1e-7
            )
            with self.assertRaises(SchemaError):
                predict_final_context(loaded, frame, batch_size=4)

    def test_strict_loader_rejects_hash_and_tensor_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                config_path = root / "model_config.json"
                config_path.write_bytes(config_path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_final_context_artifact(root, verify_code=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                state_path = root / "model_state.npz"
                original_hash = _sha256_file(state_path)
                state_path.write_bytes(state_path.read_bytes()[:-1] + b"X")
                self.assertNotEqual(_sha256_file(state_path), original_hash)
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_final_context_artifact(root, verify_code=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                hidden = root / "ensemble"
                hidden.mkdir()
                (hidden / "model2.npz").write_bytes(b"not allowed")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    load_final_context_artifact(root, verify_code=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                state_path = root / "model_state.npz"
                with self.assertWarnsRegex(UserWarning, "Duplicate name"):
                    with zipfile.ZipFile(state_path, mode="a") as archive:
                        duplicated_name = archive.namelist()[0]
                        duplicated_payload = archive.read(duplicated_name)
                        archive.writestr(duplicated_name, duplicated_payload)

                manifest_path = root / "manifest.json"
                manifest = _read_json(manifest_path)
                manifest.pop("artifact_manifest_sha256")
                manifest["files_sha256"]["model_state.npz"] = _sha256_file(
                    state_path
                )
                manifest = _payload_with_hash(manifest, "artifact_manifest_sha256")
                manifest_path.write_bytes(_serialize_json(manifest))
                with self.assertRaisesRegex(ValueError, "tensor keys"):
                    load_final_context_artifact(root, verify_code=True)

    def test_anchor_epoch_and_mode_are_fail_closed(self) -> None:
        frame = synthetic_frame()
        source = {
            "sha256": EXPECTED_FULL_SOURCE_SHA256,
            "rows": len(frame),
            "row_id_sha256": row_id_sha256(frame["row_id"].to_numpy()),
        }
        changed_epoch = verified_anchor()
        object.__setattr__(changed_epoch, "selected_epoch", 2)
        with self.assertRaisesRegex(ValueError, "epoch"):
            _train_fresh_model(
                frame,
                anchor=changed_epoch,
                source_provenance=source,
                verbose=False,
            )

        changed_mode = verified_anchor()
        changed_config = dict(changed_mode.selected_model_config)
        changed_config["game_type_mode"] = "no_bias"
        object.__setattr__(changed_mode, "selected_model_config", changed_config)
        with self.assertRaisesRegex(ValueError, "neutral"):
            _train_fresh_model(
                frame,
                anchor=changed_mode,
                source_provenance=source,
                verbose=False,
            )

    def test_coherently_rehashed_provenance_and_freshness_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                provenance_path = root / "anchor_provenance.json"
                provenance = _read_json(provenance_path)
                provenance.pop("anchor_provenance_sha256")
                provenance["source_provenance"]["rows"] += 1
                provenance = _payload_with_hash(
                    provenance, "anchor_provenance_sha256"
                )
                provenance_path.write_bytes(_serialize_json(provenance))

                manifest_path = root / "manifest.json"
                manifest = _read_json(manifest_path)
                manifest.pop("artifact_manifest_sha256")
                manifest["files_sha256"]["anchor_provenance.json"] = _sha256_file(
                    provenance_path
                )
                manifest["anchor_provenance_sha256"] = provenance[
                    "anchor_provenance_sha256"
                ]
                manifest = _payload_with_hash(manifest, "artifact_manifest_sha256")
                manifest_path.write_bytes(_serialize_json(manifest))
                with self.assertRaisesRegex(ValueError, "source provenance"):
                    load_final_context_artifact(root, verify_code=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                record_path = root / "training_record.json"
                record = _read_json(record_path)
                record.pop("training_record_sha256")
                record["initial_model_state_sha256"] = record[
                    "final_model_state_sha256"
                ]
                record = _payload_with_hash(record, "training_record_sha256")
                record_path.write_bytes(_serialize_json(record))

                manifest_path = root / "manifest.json"
                manifest = _read_json(manifest_path)
                manifest.pop("artifact_manifest_sha256")
                manifest["files_sha256"]["training_record.json"] = _sha256_file(
                    record_path
                )
                manifest["training_record_sha256"] = record[
                    "training_record_sha256"
                ]
                manifest = _payload_with_hash(manifest, "artifact_manifest_sha256")
                manifest_path.write_bytes(_serialize_json(manifest))
                with self.assertRaisesRegex(ValueError, "fresh"):
                    load_final_context_artifact(root, verify_code=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "final"
            frame = synthetic_frame()
            with synthetic_contract(frame):
                train_and_save(root, frame)
                record_path = root / "training_record.json"
                record = _read_json(record_path)
                record.pop("training_record_sha256")
                record["anchor_ready_sha256"] = "0" * 64
                record = _payload_with_hash(record, "training_record_sha256")
                record_path.write_bytes(_serialize_json(record))

                manifest_path = root / "manifest.json"
                manifest = _read_json(manifest_path)
                manifest.pop("artifact_manifest_sha256")
                manifest["files_sha256"]["training_record.json"] = _sha256_file(
                    record_path
                )
                manifest["training_record_sha256"] = record[
                    "training_record_sha256"
                ]
                manifest = _payload_with_hash(manifest, "artifact_manifest_sha256")
                manifest_path.write_bytes(_serialize_json(manifest))
                with self.assertRaisesRegex(ValueError, "anchor_ready_sha256"):
                    load_final_context_artifact(root, verify_code=True)


if __name__ == "__main__":
    unittest.main()
