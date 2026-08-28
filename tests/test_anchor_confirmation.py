from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from lg_control import anchor_confirmation
from lg_control.anchor_confirmation import (
    ExpectedCampaignLock,
    PRODUCTION_CAMPAIGN_LOCK,
    confirm_anchor,
    evaluate_confirmation_gates,
    prepare_anchor_confirmation,
    run_anchor_confirmation,
)
from lg_control.evaluation import EvaluationConfig
from lg_control.learning import OptimizationProcedure
from lg_control.regime_tuning import (
    CANDIDATE_SPECS,
    _candidate_spec_payload,
    _payload_with_hash,
    _serialize_json,
    _sha256_file,
    _tree_file_hashes,
    candidate_ranking_key,
    choose_candidate,
    evaluate_hard_gates,
    load_tuning_frame,
)
from tests.test_learning import make_learning_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCRIPT = PROJECT_ROOT / "scripts" / "build_development_snapshot.py"
SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "anchor_test_build_development_snapshot", SNAPSHOT_SCRIPT
)
assert SNAPSHOT_SPEC is not None and SNAPSHOT_SPEC.loader is not None
SNAPSHOT_MODULE = importlib.util.module_from_spec(SNAPSHOT_SPEC)
SNAPSHOT_SPEC.loader.exec_module(SNAPSHOT_MODULE)
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "run_anchor_confirmation.py"
CLI_SPEC = importlib.util.spec_from_file_location(
    "anchor_test_run_anchor_confirmation", CLI_SCRIPT
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI_MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialize_json(payload))


def _baseline_metrics() -> dict:
    return {
        "fold_brier": {"2022": 0.245, "2023": 0.255},
        "max_fold_brier": 0.255,
        "game_type_2023": {
            "F": {
                "rows": 10,
                "brier": 0.30,
                "target_rate": 0.48,
                "prediction_mean": 0.68,
                "absolute_bias": 0.20,
            },
            "R": {
                "rows": 20,
                "brier": 0.249,
                "target_rate": 0.50,
                "prediction_mean": 0.50,
                "absolute_bias": 0.0,
            },
        },
    }


def _candidate_metrics(
    *,
    brier_2022: float,
    brier_2023: float,
    f_brier: float,
    f_bias: float,
    r_brier: float,
    parameters: int,
) -> dict:
    return {
        "fold_brier": {"2022": brier_2022, "2023": brier_2023},
        "max_fold_brier": max(brier_2022, brier_2023),
        "game_type_2023": {
            "F": {
                "rows": 10,
                "brier": f_brier,
                "target_rate": 0.48,
                "prediction_mean": 0.48 + f_bias,
                "absolute_bias": abs(f_bias),
            },
            "R": {
                "rows": 20,
                "brier": r_brier,
                "target_rate": 0.50,
                "prediction_mean": 0.50,
                "absolute_bias": 0.0,
            },
        },
        "parameter_count": parameters,
    }


def _make_full_frame() -> pd.DataFrame:
    frame = make_learning_frame().reset_index(drop=True)
    alternate = frame.copy(deep=True)
    alternate["row_id"] = alternate["row_id"].astype(str) + "_ALT"
    alternate["game_type"] = alternate["game_type"].map({"F": "R", "R": "F"})
    alternate["control_success"] = 1 - alternate["control_success"].astype(int)
    return pd.concat([frame, alternate], ignore_index=True)


def _source_inventory() -> dict[str, str]:
    return {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in sorted(anchor_confirmation.REQUIRED_REGIME_SOURCE_FILES)
    }


def _build_synthetic_regime(
    root: Path,
    provenance: dict,
    *,
    selected_candidate_id: str = "c2_neutral",
) -> tuple[Path, ExpectedCampaignLock]:
    regime_root = root / "regime"
    regime_root.mkdir()
    baseline = {
        "learning_directory": "synthetic-learning",
        "evaluation_directory": "synthetic-evaluation",
        "learning_manifest_sha256": "1" * 64,
        "evaluation_manifest_sha256": "2" * 64,
        "source_input_sha256": provenance["snapshot_manifest"]["source_sha256"],
        "checkpoint_manifest_sha256": {
            "rolling_origin_2022": "3" * 64,
            "rolling_origin_2023": "4" * 64,
        },
        "metrics": _baseline_metrics(),
    }
    optimization = OptimizationProcedure(
        max_selection_epochs=1,
        scheduler_horizon_epochs=1,
        batch_size=4,
        eval_batch_size=8,
        patience=1,
        num_workers=0,
        use_amp=False,
        cpu_threads=1,
    )
    plan = _payload_with_hash(
        {
            "schema_version": 1,
            "phase": "regime_tuning_plan",
            "allowed_input_seasons": [2019, 2020, 2021, 2022, 2023],
            "development_validation_seasons": [2022, 2023],
            "input": provenance,
            "candidates": [item.to_dict() for item in CANDIDATE_SPECS],
            "optimization": optimization.to_dict(),
            "evaluation_config": EvaluationConfig(
                eval_batch_size=8, num_workers=0, cpu_threads=1
            ).to_dict(),
            "baseline_reference": baseline,
            "seed_policy": {"base_seed": 2026},
            "source_files_sha256": _source_inventory(),
            "stage_boundary": {
                "anchor_2024_allowed": False,
                "calibration_or_ensemble_allowed": False,
                "test_inference_or_submission_allowed": False,
            },
        },
        "plan_sha256",
    )
    _write_json(regime_root / "plan.json", plan)
    plan_sha256 = plan["plan_sha256"]
    metrics_by_id = {
        "c1_no_bias": _candidate_metrics(
            brier_2022=0.244,
            brier_2023=0.244,
            f_brier=0.28,
            f_bias=0.10,
            r_brier=0.2490,
            parameters=300_100,
        ),
        "c2_neutral": _candidate_metrics(
            brier_2022=0.235,
            brier_2023=0.235,
            f_brier=0.25,
            f_bias=0.04,
            r_brier=0.2480,
            parameters=300_000,
        ),
        "c3_bounded_history_residual": _candidate_metrics(
            brier_2022=0.240,
            brier_2023=0.240,
            f_brier=0.26,
            f_bias=0.06,
            r_brier=0.2485,
            parameters=300_050,
        ),
    }
    if selected_candidate_id != "c2_neutral":
        metrics_by_id[selected_candidate_id] = _candidate_metrics(
            brier_2022=0.225,
            brier_2023=0.225,
            f_brier=0.235,
            f_bias=0.02,
            r_brier=0.240,
            parameters=299_000,
        )
    manifests = []
    candidate_hashes = {}
    for candidate in CANDIDATE_SPECS:
        candidate_root = regime_root / "candidates" / candidate.candidate_id
        candidate_root.mkdir(parents=True)
        _write_json(
            candidate_root / "candidate_spec.json",
            _candidate_spec_payload(candidate=candidate, plan_sha256=plan_sha256),
        )
        metrics = metrics_by_id[candidate.candidate_id]
        manifest = _payload_with_hash(
            {
                "schema_version": 1,
                "phase": "regime_candidate_complete",
                "plan_sha256": plan_sha256,
                "candidate": candidate.to_dict(),
                "metrics": metrics,
                "hard_gates": evaluate_hard_gates(metrics, baseline["metrics"]),
                "ranking_key": list(
                    candidate_ranking_key(candidate.candidate_id, metrics)
                ),
                "anchor_2024_accessed": False,
                "files_sha256": _tree_file_hashes(candidate_root),
            },
            "candidate_manifest_sha256",
        )
        _write_json(candidate_root / "manifest.json", manifest)
        manifests.append(manifest)
        candidate_hashes[candidate.candidate_id] = _sha256_file(
            candidate_root / "manifest.json"
        )
    decision = choose_candidate(manifests)
    assert decision["selected_candidate_id"] == selected_candidate_id
    selection = _payload_with_hash(
        {
            "schema_version": 1,
            "phase": "regime_tuning_selection",
            "plan_sha256": plan_sha256,
            "candidate_manifest_sha256": candidate_hashes,
            "baseline_reference": baseline,
            "decision": decision,
            "anchor_2024_accessed": False,
            "calibration_or_ensemble_performed": False,
            "test_inference_or_submission_performed": False,
        },
        "selection_sha256",
    )
    _write_json(regime_root / "selection.json", selection)
    root_manifest = {
        "schema_version": 1,
        "phase": "regime_tuning_complete",
        "plan_sha256": plan_sha256,
        "plan_file_sha256": _sha256_file(regime_root / "plan.json"),
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": _sha256_file(regime_root / "selection.json"),
        "candidate_manifest_sha256": candidate_hashes,
        "selected_candidate_id": selected_candidate_id,
        "anchor_2024_accessed": False,
    }
    _write_json(regime_root / "manifest.json", root_manifest)
    selected_candidate = next(
        item.to_dict()
        for item in CANDIDATE_SPECS
        if item.candidate_id == selected_candidate_id
    )
    expected_campaign_lock = ExpectedCampaignLock(
        selected_candidate_id=selected_candidate_id,
        selected_candidate_spec_json=json.dumps(
            selected_candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        plan_sha256=plan_sha256,
        plan_file_sha256=_sha256_file(regime_root / "plan.json"),
        selection_sha256=selection["selection_sha256"],
        selection_file_sha256=_sha256_file(regime_root / "selection.json"),
    )
    return regime_root, expected_campaign_lock


class SyntheticAnchorFixture:
    def __init__(
        self,
        root: Path,
        *,
        selected_candidate_id: str = "c2_neutral",
    ) -> None:
        self.root = root
        self.full_path = root / "train.csv"
        self.development_path = root / "development.csv"
        _make_full_frame().to_csv(self.full_path, index=False, lineterminator="\n")
        SNAPSHOT_MODULE.build_snapshot(self.full_path, self.development_path)
        _, provenance = load_tuning_frame(self.development_path)
        self.regime_root, self.expected_campaign_lock = _build_synthetic_regime(
            root,
            provenance,
            selected_candidate_id=selected_candidate_id,
        )
        self.output_root = root / "anchor_confirmation"


class AnchorConfirmationContractTests(unittest.TestCase):
    def test_production_campaign_lock_is_literal_and_rejects_other_winners(self) -> None:
        self.assertEqual(PRODUCTION_CAMPAIGN_LOCK.selected_candidate_id, "c2_neutral")
        self.assertEqual(
            PRODUCTION_CAMPAIGN_LOCK.selected_candidate_spec(),
            next(
                item.to_dict()
                for item in CANDIDATE_SPECS
                if item.candidate_id == "c2_neutral"
            ),
        )
        self.assertEqual(
            PRODUCTION_CAMPAIGN_LOCK.plan_sha256,
            "fc17e1c504984984d6096553746b439767ccdf0a2072002dfd2f00e7256af60c",
        )
        self.assertEqual(
            PRODUCTION_CAMPAIGN_LOCK.plan_file_sha256,
            "fe991bc597392e49769e8b8c0b6a10f5576015a1fee61ae091482321f5563fb5",
        )
        self.assertEqual(
            PRODUCTION_CAMPAIGN_LOCK.selection_sha256,
            "e61f432a3952512210c33bb40d901950d7e90d8dc04352e5027b98fc61e27637",
        )
        self.assertEqual(
            PRODUCTION_CAMPAIGN_LOCK.selection_file_sha256,
            "8463b2fe1533c7a8153a8e8439f26465d480ea28000647bedac466737178f2e3",
        )
        for candidate_id in ("c1_no_bias", "c3_bounded_history_residual"):
            with self.subTest(candidate_id=candidate_id), tempfile.TemporaryDirectory() as directory:
                fixture = SyntheticAnchorFixture(
                    Path(directory), selected_candidate_id=candidate_id
                )
                with self.assertRaisesRegex(ValueError, "campaign lock mismatch"):
                    prepare_anchor_confirmation(
                        development_input=fixture.development_path,
                        regime_directory=fixture.regime_root,
                        output_directory=fixture.output_root,
                        verbose=False,
                    )
                self.assertFalse(fixture.output_root.exists())
                same_hashes_but_production_candidate = replace(
                    fixture.expected_campaign_lock,
                    selected_candidate_id=PRODUCTION_CAMPAIGN_LOCK.selected_candidate_id,
                    selected_candidate_spec_json=(
                        PRODUCTION_CAMPAIGN_LOCK.selected_candidate_spec_json
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError, "selected_candidate_id|selected_candidate_spec"
                ):
                    prepare_anchor_confirmation(
                        development_input=fixture.development_path,
                        regime_directory=fixture.regime_root,
                        output_directory=fixture.output_root,
                        verbose=False,
                        expected_campaign_lock=same_hashes_but_production_candidate,
                    )
                self.assertFalse(fixture.output_root.exists())

    def test_each_campaign_hash_mismatch_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            for field in (
                "plan_sha256",
                "plan_file_sha256",
                "selection_sha256",
                "selection_file_sha256",
            ):
                with self.subTest(field=field):
                    original = getattr(fixture.expected_campaign_lock, field)
                    replacement = ("0" if original[0] != "0" else "1") + original[1:]
                    mismatched = replace(
                        fixture.expected_campaign_lock,
                        **{field: replacement},
                    )
                    with self.assertRaisesRegex(ValueError, field):
                        prepare_anchor_confirmation(
                            development_input=fixture.development_path,
                            regime_directory=fixture.regime_root,
                            output_directory=fixture.output_root,
                            verbose=False,
                            expected_campaign_lock=mismatched,
                        )
                    self.assertFalse(fixture.output_root.exists())

            wrong_spec = fixture.expected_campaign_lock.selected_candidate_spec()
            wrong_spec["model_config"]["dropout"] = 0.16
            spec_mismatch = replace(
                fixture.expected_campaign_lock,
                selected_candidate_spec_json=json.dumps(
                    wrong_spec,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
            with self.assertRaisesRegex(ValueError, "selected_candidate_spec"):
                prepare_anchor_confirmation(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=spec_mismatch,
                )
            self.assertFalse(fixture.output_root.exists())

    def test_cli_and_core_require_exactly_one_phase_before_any_access(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            CLI_MODULE.parse_args([])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            CLI_MODULE.parse_args(["--confirm-only", "--resume"])
        parsed = CLI_MODULE.parse_args(["--prepare-only", "--resume"])
        self.assertTrue(parsed.prepare_only)
        self.assertTrue(parsed.resume)
        with (
            mock.patch.object(anchor_confirmation, "prepare_anchor_confirmation") as prepare,
            mock.patch.object(anchor_confirmation, "confirm_anchor") as confirm,
        ):
            with self.assertRaisesRegex(ValueError, "phase must be exactly one"):
                run_anchor_confirmation(
                    development_input="must-not-open-development.csv",
                    regime_directory="must-not-open-regime",
                    full_train_input="must-not-open-train.csv",
                    output_directory="must-not-create-output",
                    phase="",
                )
            with self.assertRaisesRegex(ValueError, "resume.*prepare"):
                run_anchor_confirmation(
                    development_input="must-not-open-development.csv",
                    regime_directory="must-not-open-regime",
                    full_train_input="must-not-open-train.csv",
                    output_directory="must-not-create-output",
                    phase="confirm",
                    resume=True,
                )
            prepare.assert_not_called()
            confirm.assert_not_called()

    def test_fixed_gates_include_strict_constant_and_every_regime_threshold(self) -> None:
        metrics = {
            "overall": {"brier": 0.2405, "baseline_brier": 0.2500},
            "game_type_2024": {
                "F": {"brier": 0.254, "absolute_bias": 0.05},
                "R": {"brier": 0.2485, "absolute_bias": 0.0},
            },
        }
        selected = _candidate_metrics(
            brier_2022=0.24,
            brier_2023=0.238,
            f_brier=0.249,
            f_bias=0.04,
            r_brier=0.248,
            parameters=1,
        )
        gates = evaluate_confirmation_gates(metrics, selected)
        self.assertTrue(gates["passed"])
        boundary_cases = (
            (
                "overall",
                "brier",
                0.25,
                "overall_brier_strictly_below_anchor_constant",
            ),
            (
                "overall",
                "brier",
                np.nextafter(0.2405, np.inf),
                "overall_brier_within_selected_2023_plus_0_0025",
            ),
            (
                "game_type_2024.R",
                "brier",
                np.nextafter(0.2485, np.inf),
                "game_type_R_brier_within_selected_2023_plus_0_0005",
            ),
            (
                "game_type_2024.F",
                "brier",
                np.nextafter(0.254, np.inf),
                "game_type_F_brier_within_selected_2023_plus_0_005",
            ),
            (
                "game_type_2024.F",
                "absolute_bias",
                np.nextafter(0.05, np.inf),
                "game_type_F_absolute_bias_at_most_0_05",
            ),
        )
        for location, name, value, failed_gate in boundary_cases:
            with self.subTest(failed_gate=failed_gate):
                changed = json.loads(json.dumps(metrics))
                destination = changed
                for part in location.split("."):
                    destination = destination[part]
                destination[name] = value
                observed = evaluate_confirmation_gates(changed, selected)
                self.assertFalse(observed["checks"][failed_gate]["passed"])

    def test_earliest_tie_replay_never_replaces_the_first_epoch(self) -> None:
        epoch, value = anchor_confirmation._replay_earliest_epoch(
            [
                {"epoch": 1, "inner_stop_brier": 0.2},
                {"epoch": 2, "inner_stop_brier": 0.2},
                {"epoch": 3, "inner_stop_brier": 0.1999999},
            ],
            min_delta=1e-6,
        )
        self.assertEqual(epoch, 1)
        self.assertEqual(value, 0.2)

    def test_synthetic_prepare_resume_confirm_and_one_shot_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            ready = prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                resume=False,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            self.assertFalse(ready["anchor_2024_accessed"])
            self.assertFalse((fixture.output_root / "access_started.json").exists())
            resumed = prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                resume=True,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            self.assertEqual(ready, resumed)
            with (
                mock.patch.object(
                    anchor_confirmation,
                    "_execute_decision_metrics_and_gates",
                    wraps=anchor_confirmation._execute_decision_metrics_and_gates,
                ) as decision_metrics,
                mock.patch.object(
                    anchor_confirmation,
                    "_audit_recompute_metrics_and_gates",
                    wraps=anchor_confirmation._audit_recompute_metrics_and_gates,
                ) as audit_metrics,
            ):
                terminal = confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.full_path,
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertEqual(decision_metrics.call_count, 1)
            self.assertGreaterEqual(audit_metrics.call_count, 1)
            self.assertIn(terminal["status"], {"confirmed", "not_confirmed"})
            self.assertTrue((fixture.output_root / "access_started.json").is_file())
            self.assertTrue((fixture.output_root / "anchor_predictions.csv").is_file())
            self.assertTrue((fixture.output_root / "predictions_committed.json").is_file())
            self.assertTrue((fixture.output_root / "metrics.json").is_file())
            access = json.loads(
                (fixture.output_root / "access_started.json").read_text(encoding="utf-8")
            )
            prediction_commit = json.loads(
                (fixture.output_root / "predictions_committed.json").read_text(
                    encoding="utf-8"
                )
            )
            metrics_commit = json.loads(
                (fixture.output_root / "metrics.json").read_text(encoding="utf-8")
            )
            metrics_started = json.loads(
                (fixture.output_root / "metrics_started.json").read_text(
                    encoding="utf-8"
                )
            )
            promotion_lock = json.loads(
                (fixture.output_root / "promotion_lock.json").read_text(
                    encoding="utf-8"
                )
            )
            with mock.patch.object(
                anchor_confirmation,
                "_metrics_from_prediction_table",
                side_effect=AssertionError("payload builder executed a metric"),
            ):
                rebuilt_payload = anchor_confirmation._metrics_payload(
                    root=fixture.output_root,
                    prediction_commit=prediction_commit,
                    metrics_started=metrics_started,
                    promotion_lock=promotion_lock,
                    metrics=metrics_commit["metrics"],
                    gates=metrics_commit["gates"],
                )
            self.assertEqual(rebuilt_payload, metrics_commit)
            self.assertEqual(access["one_shot_counters"], anchor_confirmation.ONE_SHOT_COUNTERS)
            self.assertNotIn("metric", access["one_shot_counters"])
            self.assertEqual(metrics_commit["decision_metric_execution_count"], 1)
            self.assertFalse(metrics_commit["audit_metric_recomputation_is_decision"])
            self.assertNotIn("metric_execution_count", metrics_commit)
            self.assertNotIn("metrics_recomputed_from_persisted_predictions", metrics_commit)
            self.assertEqual(
                metrics_commit["metric_execution_policy"],
                anchor_confirmation.METRIC_EXECUTION_POLICY,
            )
            self.assertEqual(
                prediction_commit["previous_access_ledger_sha256"],
                access["access_ledger_sha256"],
            )
            self.assertEqual(
                metrics_commit["previous_predictions_commit_sha256"],
                prediction_commit["predictions_commit_sha256"],
            )
            self.assertEqual(
                metrics_started["previous_predictions_commit_sha256"],
                prediction_commit["predictions_commit_sha256"],
            )
            self.assertEqual(
                metrics_commit["previous_metrics_started_sha256"],
                metrics_started["metrics_started_sha256"],
            )
            self.assertEqual(
                terminal["previous_metrics_sha256"], metrics_commit["metrics_sha256"]
            )
            # A committed terminal artifact must be returned without opening a
            # different/nonexistent full input and without a second inference.
            repeated = confirm_anchor(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                full_train_input=fixture.root / "must_not_be_opened.csv",
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            self.assertEqual(terminal, repeated)
            prediction_path = fixture.output_root / "anchor_predictions.csv"
            prediction_path.write_bytes(prediction_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash|differ"):
                confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.root / "still_must_not_be_opened.csv",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )

    def test_interrupted_access_closes_inconclusive_without_reopening_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            ready = prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            promotion_lock = json.loads(
                (fixture.output_root / "promotion_lock.json").read_text(encoding="utf-8")
            )
            access = anchor_confirmation._access_started_payload(
                ready=ready,
                ready_file_sha256=_sha256_file(
                    fixture.output_root / "anchor_ready.json"
                ),
                promotion_lock=promotion_lock,
                full_train_path=fixture.root / "never_open.csv",
            )
            anchor_confirmation._write_access_ledger_o_excl(
                fixture.output_root / "access_started.json", access
            )
            terminal = confirm_anchor(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                full_train_input=fixture.root / "does_not_exist.csv",
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            self.assertEqual(terminal["status"], "inconclusive")
            self.assertFalse(terminal["reinference_performed_during_recovery"])
            forged = dict(terminal)
            forged.pop("confirmation_sha256")
            forged["reason"] = "forged-but-rehashed-recovery-reason"
            forged = _payload_with_hash(forged, "confirmation_sha256")
            _write_json(fixture.output_root / "confirmation.json", forged)
            with self.assertRaisesRegex(ValueError, "predecessor chain differs"):
                confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.root / "still_must_not_be_opened.csv",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )

    def test_ready_byte_tamper_and_missing_full_path_do_not_claim_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            ready_path = fixture.output_root / "anchor_ready.json"
            canonical = ready_path.read_bytes()
            ready_path.write_bytes(canonical + b"\n")
            with self.assertRaisesRegex(ValueError, "not canonical"):
                confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.full_path,
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertFalse((fixture.output_root / "access_started.json").exists())
            ready_path.write_bytes(canonical)
            with self.assertRaises(FileNotFoundError):
                confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.root / "missing.csv",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertFalse((fixture.output_root / "access_started.json").exists())
            with self.assertRaisesRegex(ValueError, "outside"):
                confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.output_root / "checkpoint" / "manifest.json",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertFalse((fixture.output_root / "access_started.json").exists())

    def test_corrupt_access_ledger_closes_from_ready_without_source_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            (fixture.output_root / "access_started.json").write_bytes(b"{partial")
            terminal = confirm_anchor(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                full_train_input=fixture.root / "must_not_be_opened.csv",
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            self.assertEqual(terminal["status"], "inconclusive")
            self.assertEqual(terminal["last_valid_stage"], "00_anchor_ready")
            self.assertIn("access_ledger_invalid", terminal["reason"])
            self.assertNotIn("NameError", terminal["reason"])

    def test_stage02_and_stage03_commits_resume_without_source_or_reinference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            with mock.patch.object(
                anchor_confirmation,
                "_commit_metrics_and_terminal",
                side_effect=KeyboardInterrupt("synthetic crash after stage 02"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    confirm_anchor(
                        development_input=fixture.development_path,
                        regime_directory=fixture.regime_root,
                        full_train_input=fixture.full_path,
                        output_directory=fixture.output_root,
                        verbose=False,
                        expected_campaign_lock=fixture.expected_campaign_lock,
                    )
            self.assertTrue((fixture.output_root / "predictions_committed.json").is_file())
            self.assertFalse((fixture.output_root / "metrics.json").exists())
            fixture.full_path.unlink()
            real_checkpoint_reload = anchor_confirmation._load_anchor_checkpoint
            with (
                mock.patch.object(
                    anchor_confirmation,
                    "_load_anchor_checkpoint",
                    wraps=real_checkpoint_reload,
                ) as checkpoint_reload,
                mock.patch.object(
                    anchor_confirmation,
                    "_season_only_full_source_preflight",
                    side_effect=AssertionError("recovery reopened full source"),
                ),
                mock.patch.object(
                    anchor_confirmation,
                    "_extract_anchor_partition",
                    side_effect=AssertionError("recovery extracted source rows"),
                ),
                mock.patch.object(
                    anchor_confirmation.DeepTabularPreprocessor,
                    "transform",
                    side_effect=AssertionError("recovery transformed rows"),
                ),
                mock.patch.object(
                    anchor_confirmation.ContextGatedResidualModel,
                    "forward",
                    side_effect=AssertionError("recovery ran model.forward"),
                ),
                mock.patch.object(
                    anchor_confirmation,
                    "_predict",
                    side_effect=AssertionError("recovery predicted"),
                ),
                mock.patch.object(
                    anchor_confirmation,
                    "_execute_decision_metrics_and_gates",
                    wraps=anchor_confirmation._execute_decision_metrics_and_gates,
                ) as decision_metrics,
            ):
                recovered = confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.root / "must_not_reopen.csv",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertGreaterEqual(checkpoint_reload.call_count, 1)
            self.assertEqual(decision_metrics.call_count, 1)
            self.assertIn(recovered["status"], {"confirmed", "not_confirmed"})
            self.assertTrue((fixture.output_root / "metrics.json").is_file())

        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            real_write = anchor_confirmation._write_json_atomic_exclusive

            def crash_before_terminal(path: Path, payload: dict) -> None:
                if path.name == "confirmation.json":
                    raise KeyboardInterrupt("synthetic crash after stage 03")
                real_write(path, payload)

            with mock.patch.object(
                anchor_confirmation,
                "_write_json_atomic_exclusive",
                side_effect=crash_before_terminal,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    confirm_anchor(
                        development_input=fixture.development_path,
                        regime_directory=fixture.regime_root,
                        full_train_input=fixture.full_path,
                        output_directory=fixture.output_root,
                        verbose=False,
                        expected_campaign_lock=fixture.expected_campaign_lock,
                    )
            self.assertTrue((fixture.output_root / "metrics.json").is_file())
            self.assertFalse((fixture.output_root / "confirmation.json").exists())
            fixture.full_path.unlink()
            real_checkpoint_reload = anchor_confirmation._load_anchor_checkpoint
            with (
                mock.patch.object(
                    anchor_confirmation,
                    "_load_anchor_checkpoint",
                    wraps=real_checkpoint_reload,
                ) as checkpoint_reload,
                mock.patch.object(
                    anchor_confirmation,
                    "_season_only_full_source_preflight",
                    side_effect=AssertionError("recovery reopened full source"),
                ),
                mock.patch.object(
                    anchor_confirmation.DeepTabularPreprocessor,
                    "transform",
                    side_effect=AssertionError("recovery transformed rows"),
                ),
                mock.patch.object(
                    anchor_confirmation.ContextGatedResidualModel,
                    "forward",
                    side_effect=AssertionError("recovery ran model.forward"),
                ),
                mock.patch.object(
                    anchor_confirmation,
                    "_predict",
                    side_effect=AssertionError("recovery predicted"),
                ),
                mock.patch.object(
                    anchor_confirmation,
                    "_execute_decision_metrics_and_gates",
                    side_effect=AssertionError("stage 03 recovery made a new decision"),
                ),
            ):
                recovered = confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.root / "must_not_reopen.csv",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertGreaterEqual(checkpoint_reload.call_count, 1)
            self.assertIn(recovered["status"], {"confirmed", "not_confirmed"})

    def test_metric_intent_forbids_decision_reexecution_after_precommit_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            prepare_anchor_confirmation(
                development_input=fixture.development_path,
                regime_directory=fixture.regime_root,
                output_directory=fixture.output_root,
                verbose=False,
                expected_campaign_lock=fixture.expected_campaign_lock,
            )
            real_write = anchor_confirmation._write_json_atomic_exclusive

            def crash_before_metrics_commit(path: Path, payload: dict) -> None:
                if path.name == "metrics.json":
                    raise KeyboardInterrupt("synthetic crash after metric decision")
                real_write(path, payload)

            with (
                mock.patch.object(
                    anchor_confirmation,
                    "_execute_decision_metrics_and_gates",
                    wraps=anchor_confirmation._execute_decision_metrics_and_gates,
                ) as decision_metrics,
                mock.patch.object(
                    anchor_confirmation,
                    "_write_json_atomic_exclusive",
                    side_effect=crash_before_metrics_commit,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    confirm_anchor(
                        development_input=fixture.development_path,
                        regime_directory=fixture.regime_root,
                        full_train_input=fixture.full_path,
                        output_directory=fixture.output_root,
                        verbose=False,
                        expected_campaign_lock=fixture.expected_campaign_lock,
                    )
            self.assertEqual(decision_metrics.call_count, 1)
            metrics_started_path = fixture.output_root / "metrics_started.json"
            self.assertTrue(metrics_started_path.is_file())
            with self.assertRaises(FileExistsError):
                anchor_confirmation._write_metrics_started_o_excl(
                    metrics_started_path,
                    json.loads(metrics_started_path.read_text(encoding="utf-8")),
                )
            self.assertFalse((fixture.output_root / "metrics.json").exists())
            fixture.full_path.unlink()
            with (
                mock.patch.object(
                    anchor_confirmation,
                    "_execute_decision_metrics_and_gates",
                    side_effect=AssertionError("metric decision executed twice"),
                ) as forbidden_decision,
                mock.patch.object(
                    anchor_confirmation,
                    "_season_only_full_source_preflight",
                    side_effect=AssertionError("recovery reopened full source"),
                ),
                mock.patch.object(
                    anchor_confirmation.DeepTabularPreprocessor,
                    "transform",
                    side_effect=AssertionError("recovery transformed rows"),
                ),
                mock.patch.object(
                    anchor_confirmation.ContextGatedResidualModel,
                    "forward",
                    side_effect=AssertionError("recovery ran model.forward"),
                ),
                mock.patch.object(
                    anchor_confirmation,
                    "_predict",
                    side_effect=AssertionError("recovery predicted"),
                ),
            ):
                terminal = confirm_anchor(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    full_train_input=fixture.root / "must_not_reopen.csv",
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            forbidden_decision.assert_not_called()
            self.assertEqual(terminal["status"], "inconclusive")
            self.assertEqual(terminal["last_valid_stage"], "02_metrics_started")
            self.assertIn("decision_started", terminal["reason"])
            self.assertFalse((fixture.output_root / "metrics.json").exists())

    def test_regime_manifest_tamper_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticAnchorFixture(Path(directory))
            manifest_path = (
                fixture.regime_root
                / "candidates"
                / "c2_neutral"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metrics"]["fold_brier"]["2023"] = 0.0
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare_anchor_confirmation(
                    development_input=fixture.development_path,
                    regime_directory=fixture.regime_root,
                    output_directory=fixture.output_root,
                    verbose=False,
                    expected_campaign_lock=fixture.expected_campaign_lock,
                )
            self.assertFalse(fixture.output_root.exists())


if __name__ == "__main__":
    unittest.main()
