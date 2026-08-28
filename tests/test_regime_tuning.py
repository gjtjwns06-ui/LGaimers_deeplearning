from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lg_control import regime_tuning
from lg_control.regime_tuning import (
    CANDIDATE_SPECS,
    build_tuning_folds,
    candidate_ranking_key,
    choose_candidate,
    evaluate_hard_gates,
    load_tuning_frame,
    validate_tuning_seasons,
)
from tests.test_learning import make_learning_frame


def baseline_metrics() -> dict:
    return {
        "fold_brier": {"2022": 0.2443, "2023": 0.2550},
        "game_type_2023": {
            "F": {
                "brier": 0.3033,
                "target_rate": 0.473,
                "prediction_mean": 0.706,
                "absolute_bias": 0.233,
            },
            "R": {
                "brier": 0.2493,
                "target_rate": 0.503,
                "prediction_mean": 0.503,
                "absolute_bias": 0.0,
            },
        },
    }


def candidate_metrics(
    *,
    brier_2022: float,
    brier_2023: float,
    f_brier: float,
    f_bias: float,
    r_brier: float,
    parameter_count: int,
) -> dict:
    return {
        "fold_brier": {"2022": brier_2022, "2023": brier_2023},
        "game_type_2023": {
            "F": {
                "brier": f_brier,
                "target_rate": 0.473,
                "prediction_mean": 0.473 + f_bias,
                "absolute_bias": abs(f_bias),
            },
            "R": {
                "brier": r_brier,
                "target_rate": 0.503,
                "prediction_mean": 0.503,
                "absolute_bias": 0.0,
            },
        },
        "parameter_count": parameter_count,
    }


class RegimeTuningContractTests(unittest.TestCase):
    def test_candidate_registry_is_exact_and_maps_to_model_modes(self) -> None:
        self.assertEqual(
            [(item.candidate_id, item.game_type_mode) for item in CANDIDATE_SPECS],
            [
                ("c1_no_bias", "no_bias"),
                ("c2_neutral", "neutral"),
                ("c3_bounded_history_residual", "bounded_history_residual"),
            ],
        )
        for item in CANDIDATE_SPECS:
            self.assertEqual(item.model_config().game_type_mode, item.game_type_mode)

    def test_season_contract_rejects_anchor_and_requires_every_development_year(self) -> None:
        self.assertEqual(
            validate_tuning_seasons([2019, 2020, 2021, 2022, 2023]),
            (2019, 2020, 2021, 2022, 2023),
        )
        with self.assertRaisesRegex(ValueError, "sealed anchor"):
            validate_tuning_seasons([2019, 2020, 2021, 2022, 2023, 2024])
        with self.assertRaisesRegex(ValueError, "exactly seasons 2019..2023"):
            validate_tuning_seasons([2019, 2020, 2022, 2023])

    def test_csv_preflight_rejects_2024_before_official_schema_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contains_anchor.csv"
            pd.DataFrame({"season": [2019, 2020, 2021, 2022, 2023, 2024]}).to_csv(
                path, index=False
            )
            # If the full schema were checked first this would fail on missing
            # columns rather than on the sealed anchor.
            with self.assertRaisesRegex(ValueError, "sealed anchor"):
                load_tuning_frame(path)

    def test_anchor_free_validation_api_builds_only_2022_and_2023(self) -> None:
        frame = make_learning_frame()
        frame = frame[frame["season"].le(2023)].reset_index(drop=True)
        folds = build_tuning_folds(frame)
        self.assertEqual([fold.validation_season for fold in folds], [2022, 2023])
        self.assertEqual([fold.role for fold in folds], ["rolling", "rolling"])
        self.assertEqual(folds[0].train_seasons, (2019, 2020, 2021))
        self.assertEqual(folds[1].train_seasons, (2019, 2020, 2021, 2022))
        self.assertFalse(any(fold.validation_season == 2024 for fold in folds))

    def test_all_hard_gates_are_fixed_and_required(self) -> None:
        baseline = baseline_metrics()
        passing = candidate_metrics(
            brier_2022=0.2468,
            brier_2023=0.2490,
            f_brier=0.2800,
            f_bias=0.10,
            r_brier=0.2498,
            parameter_count=300_000,
        )
        result = evaluate_hard_gates(passing, baseline)
        self.assertTrue(result["passed"])
        self.assertTrue(all(item["passed"] for item in result["checks"].values()))

        failing = candidate_metrics(
            brier_2022=0.24681,  # baseline + 0.00251
            brier_2023=0.2490,
            f_brier=0.3033,  # strict improvement also fails on equality
            f_bias=0.233,
            r_brier=0.24981,
            parameter_count=300_000,
        )
        result = evaluate_hard_gates(failing, baseline)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["brier_2022_within_baseline_plus_0_0025"]["passed"]
        )
        self.assertFalse(result["checks"]["brier_2023_f_strictly_improves"]["passed"])
        self.assertFalse(
            result["checks"]["absolute_bias_2023_f_strictly_improves"]["passed"]
        )

    def test_lexicographic_selection_and_baseline_fallback(self) -> None:
        baseline = baseline_metrics()
        first = candidate_metrics(
            brier_2022=0.2465,
            brier_2023=0.2464,
            f_brier=0.285,
            f_bias=0.11,
            r_brier=0.2494,
            parameter_count=310_000,
        )
        second = candidate_metrics(
            brier_2022=0.2468,
            brier_2023=0.2450,
            f_brier=0.270,
            f_bias=0.08,
            r_brier=0.2494,
            parameter_count=290_000,
        )
        # The first candidate wins because max-fold Brier is the first key,
        # even though the second is better on every later key.
        self.assertLess(
            candidate_ranking_key("c1", first),
            candidate_ranking_key("c2", second),
        )
        manifests = [
            {
                "candidate": {"candidate_id": "c1"},
                "metrics": first,
                "hard_gates": evaluate_hard_gates(first, baseline),
            },
            {
                "candidate": {"candidate_id": "c2"},
                "metrics": second,
                "hard_gates": evaluate_hard_gates(second, baseline),
            },
        ]
        self.assertEqual(choose_candidate(manifests)["selected_candidate_id"], "c1")
        for manifest in manifests:
            manifest["hard_gates"] = {"passed": False, "checks": {}}
        fallback = choose_candidate(manifests)
        self.assertEqual(fallback["selected_candidate_id"], "baseline_v1")

    def test_embedded_plan_and_selection_hashes_detect_tampering(self) -> None:
        payload = regime_tuning._payload_with_hash(
            {"schema_version": 1, "value": "fixed"}, "plan_sha256"
        )
        observed = regime_tuning._verify_embedded_hash(payload, "plan_sha256")
        self.assertEqual(observed, payload["plan_sha256"])
        tampered = dict(payload)
        tampered["value"] = "changed"
        with self.assertRaisesRegex(ValueError, "embedded hash mismatch"):
            regime_tuning._verify_embedded_hash(tampered, "plan_sha256")

    def test_atomic_immutable_json_refuses_a_different_resume_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            regime_tuning._write_or_verify_immutable_json(path, {"value": 1})
            regime_tuning._write_or_verify_immutable_json(path, {"value": 1})
            with self.assertRaisesRegex(FileExistsError, "immutable artifact differs"):
                regime_tuning._write_or_verify_immutable_json(path, {"value": 2})

    def test_tiny_candidate_learning_evaluation_and_resume_round_trip(self) -> None:
        frame = make_learning_frame()
        frame = frame[frame["season"].le(2023)].reset_index(drop=True)
        alternate = frame.copy(deep=True)
        alternate["row_id"] = alternate["row_id"].astype(str) + "_ALT"
        alternate["game_type"] = alternate["game_type"].map({"F": "R", "R": "F"})
        frame = pd.concat([frame, alternate], ignore_index=True)
        folds = build_tuning_folds(frame)
        optimization = regime_tuning.OptimizationProcedure(
            max_selection_epochs=1,
            scheduler_horizon_epochs=1,
            batch_size=4,
            eval_batch_size=8,
            patience=1,
            num_workers=0,
            use_amp=False,
            cpu_threads=1,
        )
        evaluation_config = regime_tuning.EvaluationConfig(
            eval_batch_size=8,
            num_workers=0,
            cpu_threads=1,
        )
        candidate = CANDIDATE_SPECS[1]
        with tempfile.TemporaryDirectory() as directory:
            candidate_root = Path(directory) / candidate.candidate_id
            candidate_root.mkdir()
            learning = regime_tuning._run_candidate_learning(
                candidate_root=candidate_root,
                frame=frame,
                folds=folds,
                candidate=candidate,
                optimization=optimization,
                base_seed=2026,
                resume=False,
                verbose=False,
            )
            metrics = regime_tuning._run_candidate_evaluation(
                candidate_root=candidate_root,
                frame=frame,
                folds=folds,
                candidate_learning=learning,
                evaluation_config=evaluation_config,
                resume=False,
                verbose=False,
            )
            resumed_learning = regime_tuning._run_candidate_learning(
                candidate_root=candidate_root,
                frame=frame,
                folds=folds,
                candidate=candidate,
                optimization=optimization,
                base_seed=2026,
                resume=True,
                verbose=False,
            )
            resumed_metrics = regime_tuning._run_candidate_evaluation(
                candidate_root=candidate_root,
                frame=frame,
                folds=folds,
                candidate_learning=resumed_learning,
                evaluation_config=evaluation_config,
                resume=True,
                verbose=False,
            )
            self.assertTrue(regime_tuning._nested_values_match(metrics, resumed_metrics))
            self.assertFalse(list(candidate_root.glob(".evaluation.tmp-*")))
            plan_sha256 = "a" * 64
            baseline = regime_tuning.BaselineReference(
                learning_directory="baseline-learning",
                evaluation_directory="baseline-evaluation",
                learning_manifest_sha256="b" * 64,
                evaluation_manifest_sha256="c" * 64,
                source_input_sha256="d" * 64,
                checkpoint_manifest_sha256={},
                metrics=baseline_metrics(),
            )
            regime_tuning._atomic_write_json_exclusive(
                candidate_root / "candidate_spec.json",
                regime_tuning._candidate_spec_payload(
                    candidate=candidate, plan_sha256=plan_sha256
                ),
            )
            gates = evaluate_hard_gates(metrics, baseline.metrics)
            manifest = regime_tuning._payload_with_hash(
                {
                    "schema_version": 1,
                    "phase": "regime_candidate_complete",
                    "plan_sha256": plan_sha256,
                    "candidate": candidate.to_dict(),
                    "learning": learning,
                    "metrics": metrics,
                    "hard_gates": gates,
                    "ranking_key": list(
                        candidate_ranking_key(candidate.candidate_id, metrics)
                    ),
                    "anchor_2024_accessed": False,
                    "files_sha256": regime_tuning._tree_file_hashes(candidate_root),
                },
                "candidate_manifest_sha256",
            )
            regime_tuning._atomic_write_json_exclusive(
                candidate_root / "manifest.json", manifest
            )
            verified = regime_tuning._verify_candidate_directory(
                candidate_root,
                plan_sha256=plan_sha256,
                expected_candidate=candidate,
                folds=folds,
                baseline=baseline,
            )
            self.assertEqual(verified["candidate"], candidate.to_dict())
            with self.assertRaisesRegex(ValueError, "candidate directory name"):
                regime_tuning._verify_candidate_directory(
                    candidate_root,
                    plan_sha256=plan_sha256,
                    expected_candidate=CANDIDATE_SPECS[0],
                    folds=folds,
                    baseline=baseline,
                )


if __name__ == "__main__":
    unittest.main()
