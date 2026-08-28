"""One-shot 2024 anchor confirmation for a sealed regime-tuning winner.

The prepare phase is deliberately anchor-blind: it accepts only the sealed
2019--2023 development snapshot, re-verifies the complete regime-tuning hash
chain, selects an epoch on 2019--2022 -> 2023 once, discards that model, and
freshly refits on 2019--2023.  Only an ``anchor_ready`` commit marker permits
the explicit confirmation phase to claim the irreversible 2024 access ledger.

No existing development guard is relaxed here.  Anchor selection/refit and the
anchor-kind checkpoint live behind this separate module and separate artifact
contract.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import tempfile
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .constants import ID_COL, TARGET_COL, TRAIN_COLUMNS
from .evaluation import EvaluationConfig, _model_state_sha256, _predict, probability_metrics
from .learning import (
    EpochSelectionRecord,
    OptimizationProcedure,
    SeedSpec,
    _architecture_procedure_sha256,
    _dataset,
    _loader,
    _online_brier,
    _optimizer_and_scheduler,
    _set_deterministic_runtime,
    _train_epoch,
    _validate_training_arrays,
    row_id_sha256,
)
from .model import (
    ContextGatedModelConfig,
    ContextGatedResidualModel,
    assert_preprocessor_compatible,
    build_context_gated_model,
    validate_transformed_arrays,
)
from .preprocessing import DeepTabularPreprocessor
from .regime_tuning import (
    BASE_SEED,
    CANDIDATE_SPECS,
    DEVELOPMENT_SNAPSHOT_POLICY,
    TUNING_SEASONS,
    _candidate_spec_payload,
    _canonical_sha256,
    _payload_with_hash,
    _read_json,
    _serialize_json,
    _sha256_file,
    _tree_file_hashes,
    _verify_embedded_hash,
    _verify_file_inventory,
    candidate_ranking_key,
    choose_candidate,
    evaluate_hard_gates,
    load_tuning_frame,
)
from .schema import validate_input_frame


SCHEMA_VERSION = 1
ANCHOR_SEASON = 2024
ANCHOR_SELECTION_FIT_SEASONS = (2019, 2020, 2021, 2022)
ANCHOR_SELECTION_STOP_SEASON = 2023
ANCHOR_REFIT_SEASONS = (*ANCHOR_SELECTION_FIT_SEASONS, ANCHOR_SELECTION_STOP_SEASON)
ANCHOR_SEEDS = SeedSpec(
    initialization_seed=2226,
    training_seed=2227,
    loader_seed=2228,
)
ANCHOR_CHECKPOINT_KIND = "anchor_confirmation_pre_access_refit_v1"
ANCHOR_CHECKPOINT_FILES = frozenset(
    {
        "preprocessor.json",
        "model_config.json",
        "model_state.npz",
        "training_record.json",
    }
)
EXPECTED_CANDIDATE_IDS = tuple(item.candidate_id for item in CANDIDATE_SPECS)
REQUIRED_REGIME_SOURCE_FILES = frozenset(
    {
        "scripts/build_development_snapshot.py",
        "scripts/run_regime_tuning.py",
        "src/lg_control/constants.py",
        "src/lg_control/evaluation.py",
        "src/lg_control/learning.py",
        "src/lg_control/model.py",
        "src/lg_control/preprocessing.py",
        "src/lg_control/regime_tuning.py",
        "src/lg_control/schema.py",
        "src/lg_control/validation.py",
    }
)
FIXED_CONFIRMATION_GATES = {
    "overall_vs_anchor_constant": "anchor overall Brier < anchor target-rate constant Brier",
    "overall_vs_selected_2023": "anchor overall Brier <= selected 2023 Brier + 0.0025",
    "game_type_R_vs_selected_2023": "anchor R Brier <= selected 2023 R Brier + 0.0005",
    "game_type_F_vs_selected_2023": "anchor F Brier <= selected 2023 F Brier + 0.005",
    "game_type_F_absolute_bias": "anchor F absolute bias <= 0.05",
}
ANCHOR_PREDICTION_COLUMNS = (
    ID_COL,
    "season",
    "game_type",
    "y_true",
    "prediction",
)
ONE_SHOT_COUNTERS = {
    "candidate": 1,
    "checkpoint": 1,
    "prediction": 1,
}
METRIC_EXECUTION_POLICY = {
    "decision_input": "immutable persisted anchor_predictions.csv",
    "decision_intent_artifact": "metrics_started.json",
    "decision_intent_claim": "os.open(O_CREAT|O_EXCL|O_WRONLY)",
    "decision_execution_budget": 1,
    "decision_reexecution_after_intent": False,
    "audit_metric_recomputation_allowed": True,
    "audit_metric_recomputation_is_decision": False,
    "audit_metric_recomputation_count_recorded": False,
}
RECOVERY_POLICY = {
    "checkpoint_reload": "read_only_integrity_verification_allowed",
    "persisted_prediction_read": True,
    "metric_audit_recomputation_from_persisted_predictions": True,
    "metric_decision_reexecution_after_intent": False,
    "full_source_open": False,
    "preprocessor_transform": False,
    "model_forward": False,
    "predict": False,
    "reinference": False,
}


@dataclass(frozen=True)
class VerifiedRegimeSelection:
    regime_directory: str
    plan_sha256: str
    plan_file_sha256: str
    selection_sha256: str
    selection_file_sha256: str
    root_manifest_sha256: str
    candidate_manifest_sha256: dict[str, str]
    selected_candidate_id: str
    selected_candidate: dict[str, Any]
    selected_candidate_metrics: dict[str, Any]
    optimization: dict[str, Any]
    evaluation_config: dict[str, Any]
    expected_full_source_sha256: str

    def model_config(self) -> ContextGatedModelConfig:
        return ContextGatedModelConfig.from_dict(
            self.selected_candidate["model_config"]
        )

    def optimization_procedure(self) -> OptimizationProcedure:
        return OptimizationProcedure.from_dict(self.optimization)

    def evaluation_procedure(self) -> EvaluationConfig:
        return EvaluationConfig(**self.evaluation_config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpectedCampaignLock:
    """Externally trusted campaign identity; never inferred from artifacts."""

    selected_candidate_id: str
    selected_candidate_spec_json: str
    plan_sha256: str
    plan_file_sha256: str
    selection_sha256: str
    selection_file_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "plan_sha256",
            "plan_file_sha256",
            "selection_sha256",
            "selection_file_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"expected campaign lock has an invalid {name}")
        try:
            candidate = json.loads(self.selected_candidate_spec_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("expected campaign candidate spec is invalid JSON") from error
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if canonical != self.selected_candidate_spec_json:
            raise ValueError("expected campaign candidate spec JSON is not canonical")
        if (
            not isinstance(candidate, dict)
            or candidate.get("candidate_id") != self.selected_candidate_id
        ):
            raise ValueError("expected campaign candidate ID and spec differ")

    def selected_candidate_spec(self) -> dict[str, Any]:
        return dict(json.loads(self.selected_candidate_spec_json))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_spec": self.selected_candidate_spec(),
            "plan_sha256": self.plan_sha256,
            "plan_file_sha256": self.plan_file_sha256,
            "selection_sha256": self.selection_sha256,
            "selection_file_sha256": self.selection_file_sha256,
        }


@dataclass(frozen=True)
class LoadedAnchorCheckpoint:
    preprocessor: DeepTabularPreprocessor
    model: ContextGatedResidualModel
    training_record: dict[str, Any]
    manifest: dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


PRODUCTION_CAMPAIGN_LOCK = ExpectedCampaignLock(
    selected_candidate_id="c2_neutral",
    selected_candidate_spec_json=(
        '{"candidate_id":"c2_neutral","game_type_mode":"neutral",'
        '"label":"C2 neutral","model_config":{"dropout":0.15,'
        '"embedding_dim_cap":32,"entity_embedding_dropout":0.1,'
        '"ffn_dim":192,"game_type_mode":"neutral","hidden_dim":128,'
        '"num_blocks":3,"tower_dim":64}}'
    ),
    plan_sha256="fc17e1c504984984d6096553746b439767ccdf0a2072002dfd2f00e7256af60c",
    plan_file_sha256="fe991bc597392e49769e8b8c0b6a10f5576015a1fee61ae091482321f5563fb5",
    selection_sha256="e61f432a3952512210c33bb40d901950d7e90d8dc04352e5027b98fc61e27637",
    selection_file_sha256="8463b2fe1533c7a8153a8e8439f26465d480ea28000647bedac466737178f2e3",
)


def _preprocessor_state_sha256(preprocessor: DeepTabularPreprocessor) -> str:
    return _canonical_sha256(preprocessor.to_state_dict())


def _write_bytes_atomic_exclusive(path: Path, content: bytes) -> None:
    """Publish complete bytes atomically, refusing every overwrite."""

    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic create-if-absent publication.  Unlike
        # os.replace, it cannot overwrite an immutable ledger artifact.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic_exclusive(path, _serialize_json(payload))


def _write_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = dict(payload)
    if path.exists():
        if _read_json(path) != expected:
            raise FileExistsError(f"immutable artifact differs from this run: {path}")
        return
    _write_json_atomic_exclusive(path, expected)


def _write_access_ledger_o_excl(path: Path, payload: Mapping[str, Any]) -> None:
    """Claim irreversible access using the OS O_EXCL primitive."""

    content = _serialize_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0),
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("access ledger write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_metrics_started_o_excl(path: Path, payload: Mapping[str, Any]) -> None:
    """Irrevocably claim the sole metric-decision execution budget."""

    content = _serialize_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0),
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("metrics-started ledger write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_project_source(relative: str) -> Path:
    candidate = Path(relative)
    root = _project_root().resolve()
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe regime source inventory path: {relative!r}")
    resolved = (root / candidate).resolve()
    if root not in resolved.parents:
        raise ValueError(f"regime source inventory path escapes project: {relative!r}")
    return resolved


def _verify_recorded_regime_sources(inventory: Any) -> dict[str, str]:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("regime plan source inventory is missing")
    if not REQUIRED_REGIME_SOURCE_FILES.issubset(set(inventory)):
        missing = sorted(REQUIRED_REGIME_SOURCE_FILES.difference(inventory))
        raise ValueError(f"regime plan source inventory is incomplete: {missing}")
    verified: dict[str, str] = {}
    for relative, expected in inventory.items():
        if not isinstance(relative, str) or not _is_sha256(expected):
            raise ValueError("regime plan source inventory is malformed")
        path = _safe_project_source(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"recorded regime source hash differs: {relative}")
        verified[relative] = expected
    return verified


def _anchor_code_inventory() -> dict[str, str]:
    root = _project_root()
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "run_anchor_confirmation.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"anchor confirmation code inventory is incomplete: {missing}")
    return {
        path.relative_to(root).as_posix(): _sha256_file(path) for path in paths
    }


def _verify_development_provenance(
    plan_input: Any,
    observed: Mapping[str, Any],
) -> None:
    if not isinstance(plan_input, Mapping):
        raise ValueError("regime plan input provenance is missing")
    scalar_fields = ("sha256", "rows", "source_row_id_sha256")
    for name in scalar_fields:
        if plan_input.get(name) != observed.get(name):
            raise ValueError(f"development snapshot differs from regime plan: {name}")
    if plan_input.get("seasons") != list(TUNING_SEASONS):
        raise ValueError("regime plan development seasons are not exactly 2019..2023")
    planned_snapshot = plan_input.get("snapshot_manifest")
    observed_snapshot = observed.get("snapshot_manifest")
    if not isinstance(planned_snapshot, Mapping) or not isinstance(
        observed_snapshot, Mapping
    ):
        raise ValueError("development snapshot manifest provenance is missing")
    for name in (
        "artifact",
        "file_sha256",
        "contract_sha256",
        "source_sha256",
        "policy",
    ):
        if planned_snapshot.get(name) != observed_snapshot.get(name):
            raise ValueError(f"development snapshot manifest differs: {name}")
    if dict(observed_snapshot.get("policy", {})) != DEVELOPMENT_SNAPSHOT_POLICY:
        raise ValueError("development snapshot policy differs from the sealed contract")


def verify_regime_selection(
    regime_directory: str | Path,
    *,
    development_frame: pd.DataFrame,
    development_provenance: Mapping[str, Any],
    expected_campaign_lock: ExpectedCampaignLock = PRODUCTION_CAMPAIGN_LOCK,
) -> VerifiedRegimeSelection:
    """Re-verify the complete development selection without discovering 2024."""

    root = Path(regime_directory).resolve()
    plan_path = root / "plan.json"
    selection_path = root / "selection.json"
    root_manifest_path = root / "manifest.json"
    plan = _read_json(plan_path)
    selection = _read_json(selection_path)
    root_manifest = _read_json(root_manifest_path)
    plan_sha256 = _verify_embedded_hash(plan, "plan_sha256")
    selection_sha256 = _verify_embedded_hash(selection, "selection_sha256")
    plan_file_sha256 = _sha256_file(plan_path)
    selection_file_sha256 = _sha256_file(selection_path)
    if not isinstance(expected_campaign_lock, ExpectedCampaignLock):
        raise TypeError("expected_campaign_lock must be an ExpectedCampaignLock")
    observed_campaign_identity = {
        "plan_sha256": plan_sha256,
        "plan_file_sha256": plan_file_sha256,
        "selection_sha256": selection_sha256,
        "selection_file_sha256": selection_file_sha256,
    }
    expected_campaign_identity = {
        "plan_sha256": expected_campaign_lock.plan_sha256,
        "plan_file_sha256": expected_campaign_lock.plan_file_sha256,
        "selection_sha256": expected_campaign_lock.selection_sha256,
        "selection_file_sha256": expected_campaign_lock.selection_file_sha256,
    }
    for name, expected_value in expected_campaign_identity.items():
        if observed_campaign_identity[name] != expected_value:
            raise ValueError(f"production campaign lock mismatch: {name}")

    if plan.get("allowed_input_seasons") != list(TUNING_SEASONS):
        raise ValueError("regime plan is not development-only 2019..2023")
    if plan.get("development_validation_seasons") != [2022, 2023]:
        raise ValueError("regime plan validation seasons differ from 2022/2023")
    boundary = plan.get("stage_boundary")
    if not isinstance(boundary, Mapping) or boundary != {
        "anchor_2024_allowed": False,
        "calibration_or_ensemble_allowed": False,
        "test_inference_or_submission_allowed": False,
    }:
        raise ValueError("regime plan stage boundary is not sealed")
    _verify_development_provenance(plan.get("input"), development_provenance)
    _verify_recorded_regime_sources(plan.get("source_files_sha256"))

    if plan.get("candidates") != [item.to_dict() for item in CANDIDATE_SPECS]:
        raise ValueError("regime plan candidates differ from the fixed registry")
    if selection.get("plan_sha256") != plan_sha256:
        raise ValueError("regime selection belongs to a different plan")
    for name in (
        "anchor_2024_accessed",
        "calibration_or_ensemble_performed",
        "test_inference_or_submission_performed",
    ):
        if selection.get(name) is not False:
            raise ValueError(f"regime selection boundary flag must be false: {name}")

    expected_candidate_ids = set(EXPECTED_CANDIDATE_IDS)
    root_hashes = root_manifest.get("candidate_manifest_sha256")
    selection_hashes = selection.get("candidate_manifest_sha256")
    if (
        not isinstance(root_hashes, Mapping)
        or not isinstance(selection_hashes, Mapping)
        or set(root_hashes) != expected_candidate_ids
        or dict(root_hashes) != dict(selection_hashes)
    ):
        raise ValueError("regime candidate manifest hash maps differ from C1/C2/C3")
    if (
        root_manifest.get("phase") != "regime_tuning_complete"
        or root_manifest.get("anchor_2024_accessed") is not False
        or root_manifest.get("plan_sha256") != plan_sha256
        or root_manifest.get("plan_file_sha256") != plan_file_sha256
        or root_manifest.get("selection_sha256") != selection_sha256
        or root_manifest.get("selection_file_sha256") != selection_file_sha256
    ):
        raise ValueError("regime tuning root manifest does not reconcile")

    baseline = plan.get("baseline_reference")
    if not isinstance(baseline, Mapping) or selection.get("baseline_reference") != baseline:
        raise ValueError("regime selection baseline reference differs from plan")
    baseline_metrics = baseline.get("metrics")
    if not isinstance(baseline_metrics, Mapping):
        raise ValueError("regime baseline metrics are missing")

    manifests: list[dict[str, Any]] = []
    for candidate in CANDIDATE_SPECS:
        candidate_root = root / "candidates" / candidate.candidate_id
        manifest_path = candidate_root / "manifest.json"
        expected_hash = str(root_hashes[candidate.candidate_id])
        if _sha256_file(manifest_path) != expected_hash:
            raise ValueError(f"regime candidate manifest file hash differs: {candidate.candidate_id}")
        manifest = _read_json(manifest_path)
        _verify_embedded_hash(manifest, "candidate_manifest_sha256")
        if (
            manifest.get("phase") != "regime_candidate_complete"
            or manifest.get("plan_sha256") != plan_sha256
            or manifest.get("anchor_2024_accessed") is not False
            or manifest.get("candidate") != candidate.to_dict()
        ):
            raise ValueError(f"regime candidate manifest differs: {candidate.candidate_id}")
        _verify_file_inventory(candidate_root, manifest.get("files_sha256"))
        spec = _read_json(candidate_root / "candidate_spec.json")
        if spec != _candidate_spec_payload(candidate=candidate, plan_sha256=plan_sha256):
            raise ValueError(f"regime candidate spec differs: {candidate.candidate_id}")
        metrics = manifest.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"regime candidate metrics are missing: {candidate.candidate_id}")
        gates = evaluate_hard_gates(metrics, baseline_metrics)
        if manifest.get("hard_gates") != gates:
            raise ValueError(f"regime candidate gates do not recompute: {candidate.candidate_id}")
        ranking = list(candidate_ranking_key(candidate.candidate_id, metrics))
        if manifest.get("ranking_key") != ranking:
            raise ValueError(f"regime candidate ranking does not recompute: {candidate.candidate_id}")
        manifests.append(manifest)

    recomputed_decision = choose_candidate(manifests)
    decision = selection.get("decision")
    if decision != recomputed_decision:
        raise ValueError("regime selection decision does not recompute")
    if not isinstance(decision, Mapping) or decision.get("outcome") != "candidate_selected":
        raise ValueError("anchor confirmation requires a selected non-baseline candidate")
    selected_id = decision.get("selected_candidate_id")
    if not isinstance(selected_id, str) or selected_id not in expected_candidate_ids:
        raise ValueError("anchor confirmation cannot promote baseline or an unknown candidate")
    if selected_id != expected_campaign_lock.selected_candidate_id:
        raise ValueError("production campaign lock mismatch: selected_candidate_id")
    if root_manifest.get("selected_candidate_id") != selected_id:
        raise ValueError("regime root and selection disagree on the winner")
    selected_manifest = next(
        item for item in manifests if item["candidate"]["candidate_id"] == selected_id
    )
    if selected_manifest.get("hard_gates", {}).get("passed") is not True:
        raise ValueError("selected regime candidate did not pass every development gate")
    if selected_manifest.get("candidate") != (
        expected_campaign_lock.selected_candidate_spec()
    ):
        raise ValueError("production campaign lock mismatch: selected_candidate_spec")

    optimization = OptimizationProcedure.from_dict(plan.get("optimization", {}))
    if optimization.use_amp or optimization.num_workers != 0:
        raise ValueError("anchor confirmation requires CPU/no-AMP/num_workers=0 optimization")
    evaluation = EvaluationConfig(**dict(plan.get("evaluation_config", {})))
    expected_source = development_provenance["snapshot_manifest"]["source_sha256"]
    if baseline.get("source_input_sha256") != expected_source:
        raise ValueError("regime baseline and development snapshot source hashes differ")
    if plan.get("seed_policy", {}).get("base_seed") != BASE_SEED:
        raise ValueError("regime tuning base seed differs from its fixed contract")

    return VerifiedRegimeSelection(
        regime_directory=str(root),
        plan_sha256=plan_sha256,
        plan_file_sha256=plan_file_sha256,
        selection_sha256=selection_sha256,
        selection_file_sha256=selection_file_sha256,
        root_manifest_sha256=_sha256_file(root_manifest_path),
        candidate_manifest_sha256={
            key: str(value) for key, value in sorted(root_hashes.items())
        },
        selected_candidate_id=selected_id,
        selected_candidate=dict(selected_manifest["candidate"]),
        selected_candidate_metrics=dict(selected_manifest["metrics"]),
        optimization=optimization.to_dict(),
        evaluation_config=evaluation.to_dict(),
        expected_full_source_sha256=str(expected_source),
    )


def _selection_partitions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = pd.to_numeric(frame["season"], errors="raise").astype(int)
    if tuple(sorted(seasons.unique())) != ANCHOR_REFIT_SEASONS:
        raise ValueError("anchor preparation input must contain exactly 2019..2023")
    fit = frame[seasons.isin(ANCHOR_SELECTION_FIT_SEASONS)].copy()
    stop = frame[seasons.eq(ANCHOR_SELECTION_STOP_SEASON)].copy()
    if fit.empty or stop.empty or len(fit) + len(stop) != len(frame):
        raise ValueError("anchor selection partitions do not cover development rows")
    if tuple(sorted(pd.to_numeric(fit["season"]).astype(int).unique())) != (
        ANCHOR_SELECTION_FIT_SEASONS
    ):
        raise ValueError("anchor selection fit seasons differ from 2019..2022")
    return fit, stop


def _replay_earliest_epoch(
    history: Sequence[Mapping[str, Any]],
    *,
    min_delta: float,
) -> tuple[int, float]:
    """Replay the fixed strict-improvement rule; ties retain the first epoch."""

    best_brier = float("inf")
    best_epoch = 0
    for expected_epoch, item in enumerate(history, start=1):
        if item.get("epoch") != expected_epoch:
            raise ValueError("selection history epochs are not contiguous")
        value = float(item.get("inner_stop_brier"))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("selection history contains invalid Brier")
        if value < best_brier - min_delta:
            best_brier = value
            best_epoch = expected_epoch
    if best_epoch < 1:
        raise ValueError("selection history has no finite epoch")
    return best_epoch, best_brier


def _select_anchor_epoch(
    frame: pd.DataFrame,
    *,
    model_config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    verbose: bool,
) -> EpochSelectionRecord:
    fit_frame, stop_frame = _selection_partitions(frame)
    preprocessor = DeepTabularPreprocessor(min_category_frequency=1).fit(fit_frame)
    fit_arrays = preprocessor.transform(fit_frame)
    stop_arrays = preprocessor.transform(stop_frame)
    model = build_context_gated_model(
        preprocessor,
        config=model_config,
        initialization_seed=ANCHOR_SEEDS.initialization_seed,
    )
    fit_cat, fit_num, fit_target = _validate_training_arrays(
        model,
        fit_arrays,
        expected_row_ids=fit_frame[ID_COL].to_numpy(dtype=object, copy=False),
    )
    stop_cat, stop_num, stop_target = _validate_training_arrays(
        model,
        stop_arrays,
        expected_row_ids=stop_frame[ID_COL].to_numpy(dtype=object, copy=False),
    )
    device = torch.device("cpu")
    _set_deterministic_runtime(
        ANCHOR_SEEDS.training_seed,
        cpu_threads=optimization.cpu_threads,
    )
    model.to(device)
    fit_loader = _loader(
        _dataset(fit_cat, fit_num, fit_target),
        batch_size=optimization.batch_size,
        shuffle=True,
        loader_seed=ANCHOR_SEEDS.loader_seed,
        num_workers=optimization.num_workers,
        device=device,
    )
    stop_loader = _loader(
        _dataset(stop_cat, stop_num, stop_target),
        batch_size=optimization.eval_batch_size,
        shuffle=False,
        loader_seed=ANCHOR_SEEDS.loader_seed,
        num_workers=optimization.num_workers,
        device=device,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, optimization)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_brier = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, optimization.max_selection_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            fit_loader,
            optimizer,
            scaler,
            device=device,
            procedure=optimization,
            amp_enabled=False,
        )
        stop_brier = _online_brier(model, stop_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "inner_stop_brier": stop_brier,
            }
        )
        if verbose:
            print(
                "[anchor_confirmation][selection] "
                f"epoch={epoch} train_loss={train_loss:.8f} "
                f"stop_brier={stop_brier:.8f} lr={learning_rate:.3g}",
                flush=True,
            )
        # Strict improvement plus no replacement on equality fixes the tie to
        # the earliest epoch.
        if stop_brier < best_brier - optimization.min_delta:
            best_brier = stop_brier
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= optimization.patience:
            break
    replay_epoch, replay_brier = _replay_earliest_epoch(
        history, min_delta=optimization.min_delta
    )
    if replay_epoch != best_epoch or replay_brier != best_brier:
        raise RuntimeError("anchor epoch selection does not replay deterministically")
    fingerprint = model.feature_layout.preprocessor_fingerprint
    model.to("cpu")
    del model, optimizer, scheduler, scaler
    return EpochSelectionRecord(
        outer_fold_name="anchor_confirmation_2024",
        outer_validation_season=ANCHOR_SEASON,
        inner_partition_name="anchor_confirmation_2024__inner_stop_2023",
        inner_train_seasons=ANCHOR_SELECTION_FIT_SEASONS,
        inner_stop_season=ANCHOR_SELECTION_STOP_SEASON,
        fit_rows=len(fit_frame),
        stop_rows=len(stop_frame),
        fit_row_id_sha256=row_id_sha256(
            fit_frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
        stop_row_id_sha256=row_id_sha256(
            stop_frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
        selected_epoch=best_epoch,
        stopped_epoch=len(history),
        best_inner_brier=best_brier,
        scheduler_horizon_epochs=optimization.scheduler_horizon_epochs,
        criterion="brier",
        architecture_procedure_sha256=_architecture_procedure_sha256(model_config),
        inner_preprocessor_fingerprint=fingerprint,
        optimization=optimization.to_dict(),
        seeds=ANCHOR_SEEDS.to_dict(),
        history=tuple(history),
    )


def _selection_payload(
    record: EpochSelectionRecord,
    *,
    promotion_lock_sha256: str,
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_pre_access_epoch_selection",
            "promotion_lock_sha256": promotion_lock_sha256,
            "fit_seasons": list(ANCHOR_SELECTION_FIT_SEASONS),
            "stop_season": ANCHOR_SELECTION_STOP_SEASON,
            "outer_anchor_season_reserved": ANCHOR_SEASON,
            "tie_break": "earliest epoch under strict min_delta improvement",
            "selection_model_discarded": True,
            "anchor_2024_accessed": False,
            "record": record.to_dict(),
        },
        "epoch_selection_sha256",
    )


def _verify_selection_payload(
    payload: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    model_config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    promotion_lock_sha256: str,
) -> EpochSelectionRecord:
    _verify_embedded_hash(payload, "epoch_selection_sha256")
    if (
        payload.get("phase") != "anchor_pre_access_epoch_selection"
        or payload.get("promotion_lock_sha256") != promotion_lock_sha256
        or payload.get("fit_seasons") != list(ANCHOR_SELECTION_FIT_SEASONS)
        or payload.get("stop_season") != ANCHOR_SELECTION_STOP_SEASON
        or payload.get("outer_anchor_season_reserved") != ANCHOR_SEASON
        or payload.get("selection_model_discarded") is not True
        or payload.get("anchor_2024_accessed") is not False
    ):
        raise ValueError("anchor epoch selection envelope differs from contract")
    record = EpochSelectionRecord.from_dict(payload.get("record", {}))
    fit_frame, stop_frame = _selection_partitions(frame)
    expected = {
        "outer_fold_name": "anchor_confirmation_2024",
        "outer_validation_season": ANCHOR_SEASON,
        "inner_partition_name": "anchor_confirmation_2024__inner_stop_2023",
        "inner_train_seasons": ANCHOR_SELECTION_FIT_SEASONS,
        "inner_stop_season": ANCHOR_SELECTION_STOP_SEASON,
        "fit_rows": len(fit_frame),
        "stop_rows": len(stop_frame),
        "fit_row_id_sha256": row_id_sha256(
            fit_frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
        "stop_row_id_sha256": row_id_sha256(
            stop_frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
        "architecture_procedure_sha256": _architecture_procedure_sha256(model_config),
        "optimization": optimization.to_dict(),
        "seeds": ANCHOR_SEEDS.to_dict(),
    }
    for name, value in expected.items():
        if getattr(record, name) != value:
            raise ValueError(f"anchor epoch selection differs: {name}")
    replay_epoch, replay_brier = _replay_earliest_epoch(
        record.history, min_delta=optimization.min_delta
    )
    if record.selected_epoch != replay_epoch or record.best_inner_brier != replay_brier:
        raise ValueError("anchor selection does not preserve earliest-epoch tie break")
    return record


def _fresh_refit_on_development(
    frame: pd.DataFrame,
    *,
    model_config: ContextGatedModelConfig,
    optimization: OptimizationProcedure,
    selection: EpochSelectionRecord,
    selection_sha256: str,
    promotion_lock_sha256: str,
    development_sha256: str,
    verbose: bool,
) -> tuple[DeepTabularPreprocessor, ContextGatedResidualModel, dict[str, Any]]:
    if tuple(sorted(pd.to_numeric(frame["season"]).astype(int).unique())) != (
        ANCHOR_REFIT_SEASONS
    ):
        raise ValueError("anchor refit must use exactly 2019..2023")
    preprocessor = DeepTabularPreprocessor(min_category_frequency=1).fit(frame)
    arrays = preprocessor.transform(frame)
    model = build_context_gated_model(
        preprocessor,
        config=model_config,
        initialization_seed=ANCHOR_SEEDS.initialization_seed,
    )
    categorical, numerical, target = _validate_training_arrays(
        model,
        arrays,
        expected_row_ids=frame[ID_COL].to_numpy(dtype=object, copy=False),
    )
    device = torch.device("cpu")
    _set_deterministic_runtime(
        ANCHOR_SEEDS.training_seed,
        cpu_threads=optimization.cpu_threads,
    )
    model.to(device)
    loader = _loader(
        _dataset(categorical, numerical, target),
        batch_size=optimization.batch_size,
        shuffle=True,
        loader_seed=ANCHOR_SEEDS.loader_seed,
        num_workers=optimization.num_workers,
        device=device,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, optimization)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history: list[dict[str, Any]] = []
    for epoch in range(1, selection.selected_epoch + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device=device,
            procedure=optimization,
            amp_enabled=False,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
            }
        )
        if verbose:
            print(
                "[anchor_confirmation][fresh_refit] "
                f"epoch={epoch} train_loss={train_loss:.8f} lr={learning_rate:.3g}",
                flush=True,
            )
        scheduler.step()
    model.to("cpu").eval()
    training_record = {
        "record_version": 1,
        "phase": "anchor_confirmation_pre_access_refit",
        "checkpoint_kind": ANCHOR_CHECKPOINT_KIND,
        "reserved_anchor_season": ANCHOR_SEASON,
        "train_seasons": list(ANCHOR_REFIT_SEASONS),
        "train_rows": len(frame),
        "train_row_id_sha256": row_id_sha256(
            frame[ID_COL].to_numpy(dtype=object, copy=False)
        ),
        "development_snapshot_sha256": development_sha256,
        "selected_epoch": selection.selected_epoch,
        "epoch_selection_sha256": selection_sha256,
        "promotion_lock_sha256": promotion_lock_sha256,
        "model_config": model.get_config(),
        "architecture_procedure_sha256": _architecture_procedure_sha256(model_config),
        "parameter_report": model.parameter_report(),
        "preprocessor_fingerprint": model.feature_layout.preprocessor_fingerprint,
        "optimization": optimization.to_dict(),
        "seeds": ANCHOR_SEEDS.to_dict(),
        "history": history,
        "selection_model_reused": False,
        "fresh_model_refit": True,
        "target_removed_before_anchor_prediction": None,
        "anchor_access": {"transform_calls": 0, "predict_calls": 0, "metric_calls": 0},
        "anchor_2024_accessed": False,
        "contains_anchor_predictions": False,
        "contains_anchor_metrics": False,
        "promotion_allowed": False,
        "evaluation_status": "not_evaluated",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "amp_enabled": False,
        },
    }
    return preprocessor, model, training_record


def _save_anchor_checkpoint(
    directory: Path,
    *,
    preprocessor: DeepTabularPreprocessor,
    model: ContextGatedResidualModel,
    training_record: Mapping[str, Any],
) -> None:
    destination = directory.resolve()
    if destination.exists():
        raise FileExistsError(f"anchor checkpoint already exists: {destination}")
    assert_preprocessor_compatible(model.feature_layout, preprocessor)
    if training_record.get("phase") != "anchor_confirmation_pre_access_refit":
        raise ValueError("anchor checkpoint training phase differs")
    if training_record.get("anchor_2024_accessed") is not False:
        raise ValueError("anchor checkpoint cannot contain anchor access")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".checkpoint.staging-", dir=destination.parent)
    ).resolve()
    try:
        preprocessor.save(staging / "preprocessor.json")
        (staging / "model_config.json").write_bytes(
            _serialize_json(model.get_config())
        )
        (staging / "training_record.json").write_bytes(
            _serialize_json(training_record)
        )
        state_arrays = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in sorted(model.state_dict().items())
        }
        np.savez_compressed(staging / "model_state.npz", **state_arrays)
        manifest = _payload_with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_kind": ANCHOR_CHECKPOINT_KIND,
                "files_sha256": {
                    name: _sha256_file(staging / name)
                    for name in sorted(ANCHOR_CHECKPOINT_FILES)
                },
                "model_state_schema": {
                    name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                    for name, array in state_arrays.items()
                },
                "preprocessor_fingerprint": model.feature_layout.preprocessor_fingerprint,
                "epoch_selection_sha256": training_record["epoch_selection_sha256"],
                "promotion_lock_sha256": training_record["promotion_lock_sha256"],
                "anchor_2024_accessed": False,
                "contains_anchor_predictions": False,
                "contains_anchor_metrics": False,
            },
            "checkpoint_manifest_sha256",
        )
        (staging / "manifest.json").write_bytes(_serialize_json(manifest))
        os.replace(staging, destination)
    finally:
        if staging.exists() and staging.parent == destination.parent:
            for path in sorted(
                staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def _load_anchor_checkpoint(
    directory: Path,
    *,
    expected_model_config: ContextGatedModelConfig,
    expected_optimization: OptimizationProcedure,
    expected_selection: EpochSelectionRecord,
    expected_selection_sha256: str,
    expected_promotion_lock_sha256: str,
    expected_development_sha256: str,
    expected_train_row_ids: np.ndarray,
) -> LoadedAnchorCheckpoint:
    root = directory.resolve()
    manifest = _read_json(root / "manifest.json")
    _verify_embedded_hash(manifest, "checkpoint_manifest_sha256")
    if (
        manifest.get("checkpoint_kind") != ANCHOR_CHECKPOINT_KIND
        or manifest.get("epoch_selection_sha256") != expected_selection_sha256
        or manifest.get("promotion_lock_sha256") != expected_promotion_lock_sha256
        or manifest.get("anchor_2024_accessed") is not False
        or manifest.get("contains_anchor_predictions") is not False
        or manifest.get("contains_anchor_metrics") is not False
    ):
        raise ValueError("anchor checkpoint manifest differs from pre-access contract")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != ANCHOR_CHECKPOINT_FILES:
        raise ValueError("anchor checkpoint file inventory differs")
    for name, expected in hashes.items():
        if not _is_sha256(expected) or _sha256_file(root / name) != expected:
            raise ValueError(f"anchor checkpoint file hash differs: {name}")
    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != ANCHOR_CHECKPOINT_FILES:
        raise ValueError("anchor checkpoint contains unexpected files")

    preprocessor = DeepTabularPreprocessor.load(root / "preprocessor.json")
    model_payload = _read_json(root / "model_config.json")
    model = ContextGatedResidualModel.from_config(model_payload)
    assert_preprocessor_compatible(model.feature_layout, preprocessor)
    if model.config.to_dict() != expected_model_config.to_dict():
        raise ValueError("anchor checkpoint model differs from selected candidate")
    if manifest.get("preprocessor_fingerprint") != (
        model.feature_layout.preprocessor_fingerprint
    ):
        raise ValueError("anchor checkpoint preprocessor fingerprint differs")

    state_schema = manifest.get("model_state_schema")
    expected_state = model.state_dict()
    if not isinstance(state_schema, Mapping) or set(state_schema) != set(expected_state):
        raise ValueError("anchor checkpoint tensor schema differs from model")
    with np.load(root / "model_state.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected_state):
            raise ValueError("anchor checkpoint state keys differ from model")
        state: dict[str, torch.Tensor] = {}
        for name in archive.files:
            array = archive[name]
            schema = state_schema[name]
            architecture = expected_state[name].detach().cpu().numpy()
            if (
                not isinstance(schema, Mapping)
                or schema.get("shape") != list(array.shape)
                or schema.get("dtype") != str(array.dtype)
                or array.shape != architecture.shape
                or array.dtype != architecture.dtype
            ):
                raise ValueError(f"anchor checkpoint tensor differs: {name}")
            state[name] = torch.from_numpy(array.copy())
    model.load_state_dict(state, strict=True)
    model.to("cpu").eval()

    record = _read_json(root / "training_record.json")
    expected = {
        "phase": "anchor_confirmation_pre_access_refit",
        "checkpoint_kind": ANCHOR_CHECKPOINT_KIND,
        "reserved_anchor_season": ANCHOR_SEASON,
        "train_seasons": list(ANCHOR_REFIT_SEASONS),
        "train_rows": len(expected_train_row_ids),
        "train_row_id_sha256": row_id_sha256(expected_train_row_ids),
        "development_snapshot_sha256": expected_development_sha256,
        "selected_epoch": expected_selection.selected_epoch,
        "epoch_selection_sha256": expected_selection_sha256,
        "promotion_lock_sha256": expected_promotion_lock_sha256,
        "architecture_procedure_sha256": _architecture_procedure_sha256(
            expected_model_config
        ),
        "optimization": expected_optimization.to_dict(),
        "seeds": ANCHOR_SEEDS.to_dict(),
        "selection_model_reused": False,
        "fresh_model_refit": True,
        "anchor_2024_accessed": False,
        "contains_anchor_predictions": False,
        "contains_anchor_metrics": False,
        "promotion_allowed": False,
        "evaluation_status": "not_evaluated",
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f"anchor checkpoint training record differs: {name}")
    if record.get("model_config") != model.get_config():
        raise ValueError("anchor checkpoint model config and record differ")
    if record.get("preprocessor_fingerprint") != (
        model.feature_layout.preprocessor_fingerprint
    ):
        raise ValueError("anchor checkpoint record fingerprint differs")
    if record.get("anchor_access") != {
        "transform_calls": 0,
        "predict_calls": 0,
        "metric_calls": 0,
    }:
        raise ValueError("anchor checkpoint records pre-confirmation access")
    history = record.get("history")
    if not isinstance(history, list) or len(history) != expected_selection.selected_epoch:
        raise ValueError("anchor checkpoint refit history differs from selected epoch")
    return LoadedAnchorCheckpoint(
        preprocessor=preprocessor,
        model=model,
        training_record=record,
        manifest=manifest,
    )


def _promotion_lock_payload(
    *,
    regime: VerifiedRegimeSelection,
    development_provenance: Mapping[str, Any],
    expected_campaign_lock: ExpectedCampaignLock,
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_confirmation_promotion_lock",
            "expected_campaign_lock": expected_campaign_lock.to_dict(),
            "regime_selection": regime.to_dict(),
            "development_input": dict(development_provenance),
            "selected_candidate_id": regime.selected_candidate_id,
            "selected_model_config": regime.selected_candidate["model_config"],
            "optimization": regime.optimization,
            "evaluation_config": regime.evaluation_config,
            "anchor_seed_policy": ANCHOR_SEEDS.to_dict(),
            "epoch_selection": {
                "fit_seasons": list(ANCHOR_SELECTION_FIT_SEASONS),
                "stop_season": ANCHOR_SELECTION_STOP_SEASON,
                "tie_break": "earliest epoch under strict min_delta improvement",
                "runs": 1,
            },
            "fresh_refit_seasons": list(ANCHOR_REFIT_SEASONS),
            "fixed_confirmation_gates": FIXED_CONFIRMATION_GATES,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
            "anchor_code_files_sha256": _anchor_code_inventory(),
            "expected_full_source_sha256": regime.expected_full_source_sha256,
            "alternate_candidate_allowed": False,
            "anchor_2024_accessed": False,
        },
        "promotion_lock_sha256",
    )


def _verify_anchor_code_inventory(inventory: Any) -> None:
    if not isinstance(inventory, Mapping) or dict(inventory) != _anchor_code_inventory():
        raise ValueError("anchor confirmation code differs from promotion lock")


def _anchor_ready_payload(
    root: Path,
    *,
    promotion_lock: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    tracked = {
        "promotion_lock.json": _sha256_file(root / "promotion_lock.json"),
        "epoch_selection.json": _sha256_file(root / "epoch_selection.json"),
        **{
            f"checkpoint/{relative}": digest
            for relative, digest in _tree_file_hashes(root / "checkpoint").items()
        },
    }
    checkpoint_manifest = _read_json(root / "checkpoint" / "manifest.json")
    _verify_embedded_hash(checkpoint_manifest, "checkpoint_manifest_sha256")
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_confirmation_ready_pre_access",
            "ledger_stage": "00_anchor_ready",
            "promotion_lock_sha256": promotion_lock["promotion_lock_sha256"],
            "epoch_selection_sha256": selection_payload["epoch_selection_sha256"],
            "checkpoint_manifest_sha256": checkpoint_manifest[
                "checkpoint_manifest_sha256"
            ],
            "checkpoint_manifest_file_sha256": _sha256_file(
                root / "checkpoint" / "manifest.json"
            ),
            "selected_candidate_id": promotion_lock["selected_candidate_id"],
            "selected_epoch": selection_payload["record"]["selected_epoch"],
            "files_sha256": tracked,
            "anchor_2024_accessed": False,
            "access_ledger_created": False,
        },
        "anchor_ready_sha256",
    )


def _verify_anchor_ready(
    root: Path,
    *,
    frame: pd.DataFrame,
    development_provenance: Mapping[str, Any],
    expected_campaign_lock: ExpectedCampaignLock,
) -> tuple[dict[str, Any], dict[str, Any], EpochSelectionRecord, LoadedAnchorCheckpoint]:
    promotion_lock = _read_json(root / "promotion_lock.json")
    promotion_lock_sha256 = _verify_embedded_hash(
        promotion_lock, "promotion_lock_sha256"
    )
    if (
        promotion_lock.get("phase") != "anchor_confirmation_promotion_lock"
        or promotion_lock.get("expected_campaign_lock")
        != expected_campaign_lock.to_dict()
        or promotion_lock.get("anchor_2024_accessed") is not False
        or promotion_lock.get("alternate_candidate_allowed") is not False
        or promotion_lock.get("development_input") != dict(development_provenance)
        or promotion_lock.get("anchor_seed_policy") != ANCHOR_SEEDS.to_dict()
        or promotion_lock.get("fixed_confirmation_gates")
        != FIXED_CONFIRMATION_GATES
        or promotion_lock.get("metric_execution_policy")
        != METRIC_EXECUTION_POLICY
        or promotion_lock.get("recovery_policy") != RECOVERY_POLICY
    ):
        raise ValueError("promotion lock differs from anchor confirmation contract")
    _verify_anchor_code_inventory(promotion_lock.get("anchor_code_files_sha256"))
    selected_candidate = promotion_lock.get("regime_selection", {}).get(
        "selected_candidate"
    )
    if not isinstance(selected_candidate, Mapping):
        raise ValueError("promotion lock selected candidate is missing")
    model_config = ContextGatedModelConfig.from_dict(
        selected_candidate.get("model_config", {})
    )
    optimization = OptimizationProcedure.from_dict(
        promotion_lock.get("optimization", {})
    )
    if optimization.use_amp or optimization.num_workers != 0:
        raise ValueError("promotion lock optimization is not deterministic CPU")

    selection_payload = _read_json(root / "epoch_selection.json")
    selection = _verify_selection_payload(
        selection_payload,
        frame=frame,
        model_config=model_config,
        optimization=optimization,
        promotion_lock_sha256=promotion_lock_sha256,
    )
    selection_sha256 = selection_payload["epoch_selection_sha256"]
    checkpoint = _load_anchor_checkpoint(
        root / "checkpoint",
        expected_model_config=model_config,
        expected_optimization=optimization,
        expected_selection=selection,
        expected_selection_sha256=selection_sha256,
        expected_promotion_lock_sha256=promotion_lock_sha256,
        expected_development_sha256=str(development_provenance["sha256"]),
        expected_train_row_ids=frame[ID_COL].to_numpy(dtype=object, copy=False),
    )
    ready_path = root / "anchor_ready.json"
    ready = _read_json(ready_path)
    if ready_path.read_bytes() != _serialize_json(ready):
        raise ValueError("anchor_ready file bytes are not canonical")
    _verify_embedded_hash(ready, "anchor_ready_sha256")
    expected_ready = _anchor_ready_payload(
        root,
        promotion_lock=promotion_lock,
        selection_payload=selection_payload,
    )
    if ready != expected_ready:
        raise ValueError("anchor_ready seal differs from prepared artifacts")
    return ready, promotion_lock, selection, checkpoint


def prepare_anchor_confirmation(
    *,
    development_input: str | Path,
    regime_directory: str | Path,
    output_directory: str | Path,
    resume: bool = False,
    verbose: bool = True,
    expected_campaign_lock: ExpectedCampaignLock = PRODUCTION_CAMPAIGN_LOCK,
) -> dict[str, Any]:
    """Create and hash-seal the selected model without opening full train."""

    frame, development_provenance = load_tuning_frame(development_input)
    regime = verify_regime_selection(
        regime_directory,
        development_frame=frame,
        development_provenance=development_provenance,
        expected_campaign_lock=expected_campaign_lock,
    )
    root = Path(output_directory).resolve()
    if root.exists() and not resume:
        raise FileExistsError(f"anchor confirmation output exists; use resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "access_started.json").exists() or (root / "confirmation.json").exists():
        raise RuntimeError("anchor access has started; preparation can never run again")
    allowed = {
        "promotion_lock.json",
        "epoch_selection.json",
        "checkpoint",
        "anchor_ready.json",
    }
    interrupted = sorted(root.glob(".checkpoint.staging-*"))
    allowed_names = allowed.union(path.name for path in interrupted)
    unexpected = {path.name for path in root.iterdir()}.difference(allowed_names)
    if unexpected:
        raise FileExistsError(f"unexpected anchor preparation artifacts: {sorted(unexpected)}")

    promotion_lock = _promotion_lock_payload(
        regime=regime,
        development_provenance=development_provenance,
        expected_campaign_lock=expected_campaign_lock,
    )
    lock_path = root / "promotion_lock.json"
    _write_or_verify_json(lock_path, promotion_lock)
    promotion_lock_sha256 = promotion_lock["promotion_lock_sha256"]
    model_config = regime.model_config()
    optimization = regime.optimization_procedure()

    selection_path = root / "epoch_selection.json"
    if selection_path.exists():
        if not resume:
            raise FileExistsError("anchor epoch selection already exists; use resume")
        selection_payload = _read_json(selection_path)
        selection = _verify_selection_payload(
            selection_payload,
            frame=frame,
            model_config=model_config,
            optimization=optimization,
            promotion_lock_sha256=promotion_lock_sha256,
        )
    else:
        if (root / "checkpoint").exists() or (root / "anchor_ready.json").exists():
            raise FileExistsError("anchor checkpoint/ready exists without epoch selection")
        selection = _select_anchor_epoch(
            frame,
            model_config=model_config,
            optimization=optimization,
            verbose=verbose,
        )
        selection_payload = _selection_payload(
            selection,
            promotion_lock_sha256=promotion_lock_sha256,
        )
        _write_json_atomic_exclusive(selection_path, selection_payload)
    selection_sha256 = selection_payload["epoch_selection_sha256"]

    checkpoint_root = root / "checkpoint"
    if interrupted:
        if not resume:
            raise FileExistsError("interrupted anchor checkpoint exists; use resume")
        for staging in interrupted:
            if staging.resolve().parent != root or not staging.name.startswith(
                ".checkpoint.staging-"
            ):
                raise RuntimeError("refusing unknown checkpoint staging path")
            for path in sorted(
                staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
    if checkpoint_root.exists():
        if not resume:
            raise FileExistsError("anchor checkpoint already exists; use resume")
        checkpoint = _load_anchor_checkpoint(
            checkpoint_root,
            expected_model_config=model_config,
            expected_optimization=optimization,
            expected_selection=selection,
            expected_selection_sha256=selection_sha256,
            expected_promotion_lock_sha256=promotion_lock_sha256,
            expected_development_sha256=str(development_provenance["sha256"]),
            expected_train_row_ids=frame[ID_COL].to_numpy(dtype=object, copy=False),
        )
    else:
        preprocessor, model, training_record = _fresh_refit_on_development(
            frame,
            model_config=model_config,
            optimization=optimization,
            selection=selection,
            selection_sha256=selection_sha256,
            promotion_lock_sha256=promotion_lock_sha256,
            development_sha256=str(development_provenance["sha256"]),
            verbose=verbose,
        )
        _save_anchor_checkpoint(
            checkpoint_root,
            preprocessor=preprocessor,
            model=model,
            training_record=training_record,
        )
        checkpoint = _load_anchor_checkpoint(
            checkpoint_root,
            expected_model_config=model_config,
            expected_optimization=optimization,
            expected_selection=selection,
            expected_selection_sha256=selection_sha256,
            expected_promotion_lock_sha256=promotion_lock_sha256,
            expected_development_sha256=str(development_provenance["sha256"]),
            expected_train_row_ids=frame[ID_COL].to_numpy(dtype=object, copy=False),
        )
    del checkpoint

    ready_path = root / "anchor_ready.json"
    expected_ready = _anchor_ready_payload(
        root,
        promotion_lock=promotion_lock,
        selection_payload=selection_payload,
    )
    if ready_path.exists():
        if not resume:
            raise FileExistsError("anchor_ready already exists; use resume")
        if _read_json(ready_path) != expected_ready:
            raise ValueError("existing anchor_ready differs from prepared artifacts")
    else:
        _write_json_atomic_exclusive(ready_path, expected_ready)
    ready, _, _, _ = _verify_anchor_ready(
        root,
        frame=frame,
        development_provenance=development_provenance,
        expected_campaign_lock=expected_campaign_lock,
    )
    return ready


def _parse_identity_prefix(raw_line: bytes, *, line_number: int) -> tuple[str, int]:
    first_comma = raw_line.find(b",")
    second_comma = raw_line.find(b",", first_comma + 1)
    if first_comma <= 0 or second_comma <= first_comma + 1:
        raise ValueError(f"line {line_number} has a malformed identity prefix")
    raw_row_id = raw_line[:first_comma]
    raw_season = raw_line[first_comma + 1 : second_comma]
    if b'"' in raw_row_id or b'"' in raw_season:
        raise ValueError(f"line {line_number} quotes an identity prefix field")
    try:
        row_id = raw_row_id.decode("ascii")
        season_text = raw_season.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"line {line_number} has a non-ASCII identity prefix") from error
    if not row_id:
        raise ValueError(f"line {line_number} has an empty row_id")
    try:
        season = int(season_text)
    except ValueError as error:
        raise ValueError(f"line {line_number} has a non-integer season") from error
    if str(season) != season_text.strip():
        raise ValueError(f"line {line_number} has a non-canonical season")
    return row_id, season


def _read_and_validate_header(source: Any) -> bytes:
    raw_header = source.readline()
    if not raw_header:
        raise ValueError("full training CSV is empty")
    try:
        parsed = next(csv.reader([raw_header.decode("utf-8-sig")], strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("full training CSV header is malformed") from error
    if parsed != TRAIN_COLUMNS:
        raise ValueError("full training CSV columns differ from the official schema")
    return raw_header


def _season_only_full_source_preflight(path: Path) -> dict[str, Any]:
    """Inspect only row_id/season prefixes; every suffix remains uninterpreted."""

    digest = hashlib.sha256()
    counts: Counter[int] = Counter()
    rows = 0
    with path.open("rb") as source:
        raw_header = _read_and_validate_header(source)
        digest.update(raw_header)
        for line_number, raw_line in enumerate(source, start=2):
            digest.update(raw_line)
            _, season = _parse_identity_prefix(raw_line, line_number=line_number)
            if season not in (*ANCHOR_REFIT_SEASONS, ANCHOR_SEASON):
                raise ValueError(f"unexpected season {season} in full training CSV")
            counts[season] += 1
            rows += 1
    expected = (*ANCHOR_REFIT_SEASONS, ANCHOR_SEASON)
    if tuple(sorted(counts)) != expected or any(counts[season] <= 0 for season in expected):
        raise ValueError("full training CSV must contain exactly positive 2019..2024 partitions")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": rows,
        "season_counts": {str(season): counts[season] for season in expected},
        "lexically_parsed_fields": ["row_id", "season"],
        "non_anchor_suffix_semantically_parsed": False,
        "target_semantically_parsed_during_preflight": False,
    }


def _extract_anchor_partition(
    path: Path,
    *,
    preflight: Mapping[str, Any],
) -> pd.DataFrame:
    """Second pass: semantically parse only rows already identified as 2024."""

    buffer = io.BytesIO()
    digest = hashlib.sha256()
    counts: Counter[int] = Counter()
    with path.open("rb") as source:
        raw_header = _read_and_validate_header(source)
        digest.update(raw_header)
        buffer.write(raw_header)
        for line_number, raw_line in enumerate(source, start=2):
            digest.update(raw_line)
            _, season = _parse_identity_prefix(raw_line, line_number=line_number)
            counts[season] += 1
            if season == ANCHOR_SEASON:
                buffer.write(raw_line)
                if not raw_line.endswith((b"\n", b"\r")):
                    buffer.write(b"\n")
    if digest.hexdigest() != preflight.get("sha256"):
        raise RuntimeError("full training CSV changed after season-only preflight")
    if {str(key): value for key, value in sorted(counts.items())} != preflight.get(
        "season_counts"
    ):
        raise RuntimeError("full training season counts changed between passes")
    buffer.seek(0)
    anchor = pd.read_csv(buffer, low_memory=False)
    if list(anchor.columns) != TRAIN_COLUMNS:
        raise ValueError("extracted anchor columns differ from official training schema")
    if anchor.empty:
        raise ValueError("extracted anchor partition is empty")
    seasons = pd.to_numeric(anchor["season"], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.equal(seasons, float(ANCHOR_SEASON)).all():
        raise ValueError("extracted anchor partition contains a non-2024 row")
    validate_input_frame(anchor, require_target=True, allow_target=True)
    if not anchor[ID_COL].is_unique:
        raise ValueError("anchor row IDs must be unique")
    if len(anchor) != int(preflight["season_counts"][str(ANCHOR_SEASON)]):
        raise ValueError("anchor row count differs from season-only preflight")
    return anchor


def evaluate_confirmation_gates(
    metrics: Mapping[str, Any],
    selected_candidate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    overall = metrics["overall"]
    f_metrics = metrics["game_type_2024"]["F"]
    r_metrics = metrics["game_type_2024"]["R"]
    selected_2023 = selected_candidate_metrics["fold_brier"]["2023"]
    selected_f = selected_candidate_metrics["game_type_2023"]["F"]["brier"]
    selected_r = selected_candidate_metrics["game_type_2023"]["R"]["brier"]
    checks = {
        "overall_brier_strictly_below_anchor_constant": {
            "value": float(overall["brier"]),
            "threshold": float(overall["baseline_brier"]),
            "operator": "<",
        },
        "overall_brier_within_selected_2023_plus_0_0025": {
            "value": float(overall["brier"]),
            "threshold": float(selected_2023) + 0.0025,
            "operator": "<=",
        },
        "game_type_R_brier_within_selected_2023_plus_0_0005": {
            "value": float(r_metrics["brier"]),
            "threshold": float(selected_r) + 0.0005,
            "operator": "<=",
        },
        "game_type_F_brier_within_selected_2023_plus_0_005": {
            "value": float(f_metrics["brier"]),
            "threshold": float(selected_f) + 0.005,
            "operator": "<=",
        },
        "game_type_F_absolute_bias_at_most_0_05": {
            "value": float(f_metrics["absolute_bias"]),
            "threshold": 0.05,
            "operator": "<=",
        },
    }
    for check in checks.values():
        value = float(check["value"])
        threshold = float(check["threshold"])
        if not np.isfinite(value) or not np.isfinite(threshold):
            raise ValueError("anchor gate contains a non-finite metric")
        check["passed"] = (
            value < threshold if check["operator"] == "<" else value <= threshold
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def _anchor_metrics(
    anchor: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    target = pd.to_numeric(anchor[TARGET_COL], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    game_type = anchor["game_type"].astype(str)
    if set(game_type.unique()) != {"F", "R"}:
        raise ValueError("anchor game_type partition must contain exactly F and R")
    result: dict[str, Any] = {
        "overall": probability_metrics(target, probabilities),
        "game_type_2024": {},
    }
    for value in ("F", "R"):
        mask = game_type.eq(value).to_numpy()
        slice_metrics = probability_metrics(target[mask], probabilities[mask])
        slice_metrics["absolute_bias"] = abs(
            float(slice_metrics["prediction_mean"])
            - float(slice_metrics["target_rate"])
        )
        result["game_type_2024"][value] = slice_metrics
    return result


def _access_started_payload(
    *,
    ready: Mapping[str, Any],
    ready_file_sha256: str,
    promotion_lock: Mapping[str, Any],
    full_train_path: Path,
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_confirmation_access_started",
            "ledger_stage": "01_access_started",
            "previous_anchor_ready_sha256": ready["anchor_ready_sha256"],
            "previous_anchor_ready_file_sha256": ready_file_sha256,
            "promotion_lock_sha256": promotion_lock["promotion_lock_sha256"],
            "selected_candidate_id": promotion_lock["selected_candidate_id"],
            "declared_full_train_path": str(full_train_path),
            "expected_full_source_sha256": promotion_lock[
                "expected_full_source_sha256"
            ],
            "access_claim_primitive": "os.open(O_CREAT|O_EXCL|O_WRONLY)",
            "one_shot": True,
            "one_shot_counters": ONE_SHOT_COUNTERS,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
            "reinference_allowed_after_this_record": False,
        },
        "access_ledger_sha256",
    )


def _write_prediction_table_atomic_exclusive(path: Path, table: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            table.to_csv(
                handle,
                index=False,
                lineterminator="\n",
                float_format="%.17g",
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_prediction_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    if list(table.columns) != list(ANCHOR_PREDICTION_COLUMNS):
        raise ValueError("persisted anchor prediction columns differ from contract")
    if table.empty or not table[ID_COL].is_unique or table[ID_COL].isna().any():
        raise ValueError("persisted anchor prediction IDs must be nonempty and unique")
    seasons = pd.to_numeric(table["season"], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.equal(seasons, float(ANCHOR_SEASON)).all():
        raise ValueError("persisted predictions contain a non-anchor season")
    if set(table["game_type"].astype(str).unique()) != {"F", "R"}:
        raise ValueError("persisted predictions must contain exactly F/R game types")
    target = pd.to_numeric(table["y_true"], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    probability = pd.to_numeric(table["prediction"], errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("persisted anchor targets are not finite binary labels")
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("persisted anchor probabilities are invalid")
    return table


def _metrics_from_prediction_table(table: pd.DataFrame) -> dict[str, Any]:
    pseudo_anchor = table[[ID_COL, "season", "game_type", "y_true"]].rename(
        columns={"y_true": TARGET_COL}
    )
    probabilities = table["prediction"].to_numpy(dtype=np.float64, copy=True)
    return _anchor_metrics(pseudo_anchor, probabilities)


def _verify_access_ledger(
    path: Path,
    *,
    ready: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path)
    if path.read_bytes() != _serialize_json(payload):
        raise ValueError("anchor access ledger bytes are not canonical")
    _verify_embedded_hash(payload, "access_ledger_sha256")
    if (
        payload.get("phase") != "anchor_confirmation_access_started"
        or payload.get("ledger_stage") != "01_access_started"
        or payload.get("previous_anchor_ready_sha256")
        != ready.get("anchor_ready_sha256")
        or payload.get("previous_anchor_ready_file_sha256")
        != _sha256_file(path.parent / "anchor_ready.json")
        or payload.get("promotion_lock_sha256")
        != promotion_lock.get("promotion_lock_sha256")
        or payload.get("selected_candidate_id")
        != promotion_lock.get("selected_candidate_id")
        or payload.get("expected_full_source_sha256")
        != promotion_lock.get("expected_full_source_sha256")
        or not isinstance(payload.get("declared_full_train_path"), str)
        or not Path(payload["declared_full_train_path"]).is_absolute()
        or payload.get("access_claim_primitive")
        != "os.open(O_CREAT|O_EXCL|O_WRONLY)"
        or payload.get("one_shot") is not True
        or payload.get("one_shot_counters") != ONE_SHOT_COUNTERS
        or payload.get("metric_execution_policy") != METRIC_EXECUTION_POLICY
        or payload.get("recovery_policy") != RECOVERY_POLICY
        or payload.get("reinference_allowed_after_this_record") is not False
    ):
        raise ValueError("anchor access ledger differs from one-shot contract")
    return payload


def _prediction_commit_payload(
    *,
    root: Path,
    ready: Mapping[str, Any],
    access: Mapping[str, Any],
    regime: VerifiedRegimeSelection,
    checkpoint: LoadedAnchorCheckpoint,
    prediction_table: pd.DataFrame,
    source_preflight: Mapping[str, Any],
    model_state_before: str,
    model_state_after: str,
    preprocessor_state_before: str,
    preprocessor_state_after: str,
) -> dict[str, Any]:
    prediction_path = root / "anchor_predictions.csv"
    checkpoint_manifest = _read_json(root / "checkpoint" / "manifest.json")
    _verify_embedded_hash(checkpoint_manifest, "checkpoint_manifest_sha256")
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_predictions_committed",
            "ledger_stage": "02_predictions_committed",
            "previous_anchor_ready_sha256": ready["anchor_ready_sha256"],
            "previous_access_ledger_sha256": access["access_ledger_sha256"],
            "previous_access_file_sha256": _sha256_file(root / "access_started.json"),
            "selected_candidate_id": regime.selected_candidate_id,
            "checkpoint_manifest_sha256": checkpoint_manifest[
                "checkpoint_manifest_sha256"
            ],
            "checkpoint_manifest_file_sha256": _sha256_file(
                root / "checkpoint" / "manifest.json"
            ),
            "prediction_file": "anchor_predictions.csv",
            "prediction_file_sha256": _sha256_file(prediction_path),
            "prediction_rows": len(prediction_table),
            "prediction_row_id_sha256": row_id_sha256(
                prediction_table[ID_COL].to_numpy(dtype=object, copy=False)
            ),
            "prediction_columns": list(ANCHOR_PREDICTION_COLUMNS),
            "semantic_partition": [ANCHOR_SEASON],
            "source_preflight": dict(source_preflight),
            "target_removed_before_model": True,
            "target_consumed_by_model": False,
            "anchor_target_used_for_metrics_only": True,
            "non_anchor_target_semantically_parsed": False,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "prepared_model_state_sha256": _model_state_sha256(checkpoint.model),
            "preprocessor_state_sha256_before": preprocessor_state_before,
            "preprocessor_state_sha256_after": preprocessor_state_after,
            "prepared_preprocessor_state_sha256": _preprocessor_state_sha256(
                checkpoint.preprocessor
            ),
            "one_shot_counters": ONE_SHOT_COUNTERS,
            "prediction_execution_count": 1,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
            "alternate_candidate_attempted": False,
        },
        "predictions_commit_sha256",
    )


def _verify_prediction_commit(
    root: Path,
    *,
    ready: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    checkpoint: LoadedAnchorCheckpoint,
) -> tuple[dict[str, Any], pd.DataFrame]:
    access = _verify_access_ledger(
        root / "access_started.json",
        ready=ready,
        promotion_lock=promotion_lock,
    )
    commit_path = root / "predictions_committed.json"
    commit = _read_json(commit_path)
    if commit_path.read_bytes() != _serialize_json(commit):
        raise ValueError("anchor prediction commit bytes are not canonical")
    _verify_embedded_hash(commit, "predictions_commit_sha256")
    prediction_path = root / "anchor_predictions.csv"
    checkpoint_manifest = _read_json(root / "checkpoint" / "manifest.json")
    _verify_embedded_hash(checkpoint_manifest, "checkpoint_manifest_sha256")
    if (
        commit.get("phase") != "anchor_predictions_committed"
        or commit.get("ledger_stage") != "02_predictions_committed"
        or commit.get("previous_anchor_ready_sha256")
        != ready.get("anchor_ready_sha256")
        or commit.get("previous_access_ledger_sha256")
        != access.get("access_ledger_sha256")
        or commit.get("previous_access_file_sha256")
        != _sha256_file(root / "access_started.json")
        or commit.get("selected_candidate_id")
        != promotion_lock.get("selected_candidate_id")
        or commit.get("checkpoint_manifest_sha256")
        != checkpoint_manifest.get("checkpoint_manifest_sha256")
        or commit.get("checkpoint_manifest_file_sha256")
        != _sha256_file(root / "checkpoint" / "manifest.json")
        or commit.get("prediction_file") != "anchor_predictions.csv"
        or commit.get("prediction_file_sha256") != _sha256_file(prediction_path)
        or commit.get("prediction_columns") != list(ANCHOR_PREDICTION_COLUMNS)
        or commit.get("semantic_partition") != [ANCHOR_SEASON]
        or commit.get("target_removed_before_model") is not True
        or commit.get("target_consumed_by_model") is not False
        or commit.get("anchor_target_used_for_metrics_only") is not True
        or commit.get("non_anchor_target_semantically_parsed") is not False
        or commit.get("model_state_sha256_before")
        != commit.get("model_state_sha256_after")
        or commit.get("model_state_sha256_after")
        != _model_state_sha256(checkpoint.model)
        or commit.get("prepared_model_state_sha256")
        != _model_state_sha256(checkpoint.model)
        or commit.get("preprocessor_state_sha256_before")
        != commit.get("preprocessor_state_sha256_after")
        or commit.get("preprocessor_state_sha256_after")
        != _preprocessor_state_sha256(checkpoint.preprocessor)
        or commit.get("prepared_preprocessor_state_sha256")
        != _preprocessor_state_sha256(checkpoint.preprocessor)
        or commit.get("one_shot_counters") != ONE_SHOT_COUNTERS
        or commit.get("prediction_execution_count") != 1
        or commit.get("metric_execution_policy") != METRIC_EXECUTION_POLICY
        or commit.get("recovery_policy") != RECOVERY_POLICY
        or commit.get("alternate_candidate_attempted") is not False
    ):
        raise ValueError("anchor prediction commit differs from hash-chain contract")
    source_preflight = commit.get("source_preflight")
    expected_season_keys = {
        str(season) for season in (*ANCHOR_REFIT_SEASONS, ANCHOR_SEASON)
    }
    season_counts = (
        source_preflight.get("season_counts")
        if isinstance(source_preflight, Mapping)
        else None
    )
    if (
        not isinstance(source_preflight, Mapping)
        or source_preflight.get("sha256")
        != promotion_lock.get("expected_full_source_sha256")
        or source_preflight.get("path") != access.get("declared_full_train_path")
        or source_preflight.get("lexically_parsed_fields") != ["row_id", "season"]
        or source_preflight.get("non_anchor_suffix_semantically_parsed") is not False
        or source_preflight.get("target_semantically_parsed_during_preflight")
        is not False
        or not isinstance(season_counts, Mapping)
        or set(season_counts) != expected_season_keys
        or any(type(value) is not int or value <= 0 for value in season_counts.values())
        or source_preflight.get("rows") != sum(season_counts.values())
    ):
        raise ValueError("anchor prediction source preflight differs")
    table = _read_prediction_table(prediction_path)
    if (
        commit.get("prediction_rows") != len(table)
        or len(table) != season_counts[str(ANCHOR_SEASON)]
        or commit.get("prediction_row_id_sha256")
        != row_id_sha256(table[ID_COL].to_numpy(dtype=object, copy=False))
    ):
        raise ValueError("anchor prediction commit row identity differs")
    return commit, table


def _metrics_started_payload(
    *,
    root: Path,
    prediction_commit: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_metric_decision_started",
            "ledger_stage": "02_metrics_started",
            "previous_predictions_commit_sha256": prediction_commit[
                "predictions_commit_sha256"
            ],
            "previous_predictions_commit_file_sha256": _sha256_file(
                root / "predictions_committed.json"
            ),
            "prediction_file_sha256": _sha256_file(
                root / "anchor_predictions.csv"
            ),
            "selected_candidate_id": promotion_lock["selected_candidate_id"],
            "decision_execution_budget": 1,
            "decision_reexecution_allowed_after_this_record": False,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
            "one_shot_counters": ONE_SHOT_COUNTERS,
        },
        "metrics_started_sha256",
    )


def _verify_metrics_started(
    root: Path,
    *,
    prediction_commit: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / "metrics_started.json"
    payload = _read_json(path)
    if path.read_bytes() != _serialize_json(payload):
        raise ValueError("metrics-started ledger bytes are not canonical")
    _verify_embedded_hash(payload, "metrics_started_sha256")
    expected = _metrics_started_payload(
        root=root,
        prediction_commit=prediction_commit,
        promotion_lock=promotion_lock,
    )
    if payload != expected:
        raise ValueError("metrics-started ledger differs from decision contract")
    return payload


def _execute_decision_metrics_and_gates(
    prediction_table: pd.DataFrame,
    selected_candidate_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the single gate decision from the immutable prediction table."""

    metrics = _metrics_from_prediction_table(prediction_table)
    gates = evaluate_confirmation_gates(metrics, selected_candidate_metrics)
    return metrics, gates


def _audit_recompute_metrics_and_gates(
    prediction_table: pd.DataFrame,
    selected_candidate_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically verify a decision; this is never another decision."""

    metrics = _metrics_from_prediction_table(prediction_table)
    gates = evaluate_confirmation_gates(metrics, selected_candidate_metrics)
    return metrics, gates


def _metrics_payload(
    *,
    root: Path,
    prediction_commit: Mapping[str, Any],
    metrics_started: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize already-decided metrics/gates without executing metrics."""

    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_metrics_committed",
            "ledger_stage": "03_metrics_committed",
            "previous_predictions_commit_sha256": prediction_commit[
                "predictions_commit_sha256"
            ],
            "previous_predictions_commit_file_sha256": _sha256_file(
                root / "predictions_committed.json"
            ),
            "previous_metrics_started_sha256": metrics_started[
                "metrics_started_sha256"
            ],
            "previous_metrics_started_file_sha256": _sha256_file(
                root / "metrics_started.json"
            ),
            "prediction_file_sha256": _sha256_file(
                root / "anchor_predictions.csv"
            ),
            "selected_candidate_id": promotion_lock["selected_candidate_id"],
            "metrics": dict(metrics),
            "gates": dict(gates),
            "one_shot_counters": ONE_SHOT_COUNTERS,
            "decision_metric_execution_count": 1,
            "audit_metric_recomputation_is_decision": False,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
        },
        "metrics_sha256",
    )


def _verify_metrics_commit(
    root: Path,
    *,
    ready: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    checkpoint: LoadedAnchorCheckpoint,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    prediction_commit, table = _verify_prediction_commit(
        root,
        ready=ready,
        promotion_lock=promotion_lock,
        checkpoint=checkpoint,
    )
    metrics_path = root / "metrics.json"
    payload = _read_json(metrics_path)
    if metrics_path.read_bytes() != _serialize_json(payload):
        raise ValueError("anchor metrics commit bytes are not canonical")
    _verify_embedded_hash(payload, "metrics_sha256")
    metrics_started = _verify_metrics_started(
        root,
        prediction_commit=prediction_commit,
        promotion_lock=promotion_lock,
    )
    audit_metrics, audit_gates = _audit_recompute_metrics_and_gates(
        table,
        promotion_lock["regime_selection"]["selected_candidate_metrics"],
    )
    expected = _metrics_payload(
        root=root,
        prediction_commit=prediction_commit,
        metrics_started=metrics_started,
        promotion_lock=promotion_lock,
        metrics=audit_metrics,
        gates=audit_gates,
    )
    if payload != expected:
        raise ValueError("anchor metrics do not recompute from persisted predictions")
    return payload, prediction_commit, table


def _terminal_payload(
    *,
    root: Path,
    ready: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
) -> dict[str, Any]:
    status = "confirmed" if metrics_payload["gates"]["passed"] else "not_confirmed"
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_confirmation_terminal",
            "ledger_stage": "04_confirmation",
            "status": status,
            "previous_anchor_ready_sha256": ready["anchor_ready_sha256"],
            "previous_metrics_sha256": metrics_payload["metrics_sha256"],
            "previous_metrics_file_sha256": _sha256_file(root / "metrics.json"),
            "one_shot_counters": ONE_SHOT_COUNTERS,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
            "alternate_candidate_attempted": False,
        },
        "confirmation_sha256",
    )


def _inconclusive_payload(
    *,
    ready: Mapping[str, Any],
    access_file_sha256: str,
    access_ledger_sha256: str | None,
    reason: str,
    last_valid_stage: str,
    last_valid_stage_sha256: str | None,
) -> dict[str, Any]:
    return _payload_with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "anchor_confirmation_terminal",
            "ledger_stage": "04_confirmation",
            "status": "inconclusive",
            "previous_anchor_ready_sha256": ready["anchor_ready_sha256"],
            "previous_access_file_sha256": access_file_sha256,
            "previous_access_ledger_sha256": access_ledger_sha256,
            "last_valid_stage": last_valid_stage,
            "last_valid_stage_sha256": last_valid_stage_sha256,
            "reason": reason,
            "metrics": None,
            "gates": None,
            "one_shot_counters": ONE_SHOT_COUNTERS,
            "metric_execution_policy": METRIC_EXECUTION_POLICY,
            "recovery_policy": RECOVERY_POLICY,
            "full_source_reopened_for_recovery": False,
            "reinference_performed_during_recovery": False,
            "alternate_candidate_attempted": False,
        },
        "confirmation_sha256",
    )


def _derive_inconclusive_payload(
    root: Path,
    *,
    ready: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    checkpoint: LoadedAnchorCheckpoint,
) -> dict[str, Any]:
    """Derive the sole valid terminal state from the last verifiable ledger."""

    access_path = root / "access_started.json"
    access_file_sha256 = _sha256_file(access_path)
    try:
        access = _verify_access_ledger(
            access_path,
            ready=ready,
            promotion_lock=promotion_lock,
        )
    except Exception as error:
        return _inconclusive_payload(
            ready=ready,
            access_file_sha256=access_file_sha256,
            access_ledger_sha256=None,
            reason=f"access_ledger_invalid: {type(error).__name__}: {error}",
            last_valid_stage="00_anchor_ready",
            last_valid_stage_sha256=ready["anchor_ready_sha256"],
        )

    access_sha256 = access["access_ledger_sha256"]
    prediction_commit_path = root / "predictions_committed.json"
    if not prediction_commit_path.is_file():
        suffix = (
            "_with_uncommitted_post_access_files"
            if (root / "anchor_predictions.csv").exists()
            or (root / "metrics_started.json").exists()
            or (root / "metrics.json").exists()
            else ""
        )
        return _inconclusive_payload(
            ready=ready,
            access_file_sha256=access_file_sha256,
            access_ledger_sha256=access_sha256,
            reason=f"access_started_before_valid_predictions_commit{suffix}",
            last_valid_stage="01_access_started",
            last_valid_stage_sha256=access_sha256,
        )
    try:
        prediction_commit, _ = _verify_prediction_commit(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
    except Exception as error:
        return _inconclusive_payload(
            ready=ready,
            access_file_sha256=access_file_sha256,
            access_ledger_sha256=access_sha256,
            reason=f"predictions_commit_invalid: {type(error).__name__}: {error}",
            last_valid_stage="01_access_started",
            last_valid_stage_sha256=access_sha256,
        )

    metrics_started_path = root / "metrics_started.json"
    metrics_path = root / "metrics.json"
    if not metrics_started_path.is_file():
        if metrics_path.exists():
            return _inconclusive_payload(
                ready=ready,
                access_file_sha256=access_file_sha256,
                access_ledger_sha256=access_sha256,
                reason="metrics_commit_exists_without_decision_intent",
                last_valid_stage="02_predictions_committed",
                last_valid_stage_sha256=prediction_commit[
                    "predictions_commit_sha256"
                ],
            )
        raise RuntimeError(
            "a valid predictions commit before metric intent is recoverable"
        )
    try:
        metrics_started = _verify_metrics_started(
            root,
            prediction_commit=prediction_commit,
            promotion_lock=promotion_lock,
        )
    except Exception as error:
        return _inconclusive_payload(
            ready=ready,
            access_file_sha256=access_file_sha256,
            access_ledger_sha256=access_sha256,
            reason=f"metrics_started_invalid: {type(error).__name__}: {error}",
            last_valid_stage="02_predictions_committed",
            last_valid_stage_sha256=prediction_commit[
                "predictions_commit_sha256"
            ],
        )
    if not metrics_path.is_file():
        return _inconclusive_payload(
            ready=ready,
            access_file_sha256=access_file_sha256,
            access_ledger_sha256=access_sha256,
            reason="metric_decision_started_before_valid_metrics_commit",
            last_valid_stage="02_metrics_started",
            last_valid_stage_sha256=metrics_started["metrics_started_sha256"],
        )
    try:
        _verify_metrics_commit(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
    except Exception as error:
        return _inconclusive_payload(
            ready=ready,
            access_file_sha256=access_file_sha256,
            access_ledger_sha256=access_sha256,
            reason=f"metrics_commit_invalid: {type(error).__name__}: {error}",
            last_valid_stage="02_metrics_started",
            last_valid_stage_sha256=metrics_started["metrics_started_sha256"],
        )
    raise RuntimeError("valid committed metrics require terminal recovery")


def _verify_terminal_confirmation(
    payload: Mapping[str, Any],
    *,
    root: Path,
    ready: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    checkpoint: LoadedAnchorCheckpoint,
) -> dict[str, Any]:
    confirmation_path = root / "confirmation.json"
    if confirmation_path.is_file() and confirmation_path.read_bytes() != _serialize_json(
        payload
    ):
        raise ValueError("terminal confirmation bytes are not canonical")
    _verify_embedded_hash(payload, "confirmation_sha256")
    access_path = root / "access_started.json"
    if (
        payload.get("phase") != "anchor_confirmation_terminal"
        or payload.get("ledger_stage") != "04_confirmation"
        or payload.get("previous_anchor_ready_sha256")
        != ready.get("anchor_ready_sha256")
        or payload.get("one_shot_counters") != ONE_SHOT_COUNTERS
        or payload.get("metric_execution_policy") != METRIC_EXECUTION_POLICY
        or payload.get("recovery_policy") != RECOVERY_POLICY
        or payload.get("alternate_candidate_attempted") is not False
    ):
        raise ValueError("terminal anchor confirmation hash chain differs")
    status = payload.get("status")
    if status not in {"confirmed", "not_confirmed", "inconclusive"}:
        raise ValueError("terminal anchor confirmation has an invalid status")
    if status == "inconclusive":
        if (
            payload.get("metrics") is not None
            or payload.get("gates") is not None
            or payload.get("previous_access_file_sha256")
            != _sha256_file(access_path)
            or payload.get("full_source_reopened_for_recovery") is not False
            or payload.get("reinference_performed_during_recovery") is not False
        ):
            raise ValueError("inconclusive confirmation audit differs")
        expected = _derive_inconclusive_payload(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
        if payload != expected:
            raise ValueError("inconclusive confirmation predecessor chain differs")
        return dict(payload)
    metrics_payload, _, _ = _verify_metrics_commit(
        root,
        ready=ready,
        promotion_lock=promotion_lock,
        checkpoint=checkpoint,
    )
    expected = _terminal_payload(
        root=root,
        ready=ready,
        metrics_payload=metrics_payload,
    )
    if payload != expected:
        raise ValueError("terminal anchor confirmation differs from persisted metrics")
    return dict(payload)


def _commit_metrics_and_terminal(
    root: Path,
    *,
    ready: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    checkpoint: LoadedAnchorCheckpoint,
) -> dict[str, Any]:
    metrics_path = root / "metrics.json"
    if metrics_path.exists():
        metrics_payload, _, _ = _verify_metrics_commit(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
    else:
        prediction_commit, table = _verify_prediction_commit(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
        metrics_started_path = root / "metrics_started.json"
        if metrics_started_path.exists():
            raise RuntimeError(
                "metric decision intent already exists; decision reexecution is forbidden"
            )
        metrics_started = _metrics_started_payload(
            root=root,
            prediction_commit=prediction_commit,
            promotion_lock=promotion_lock,
        )
        _write_metrics_started_o_excl(metrics_started_path, metrics_started)
        persisted_metrics_started = _verify_metrics_started(
            root,
            prediction_commit=prediction_commit,
            promotion_lock=promotion_lock,
        )
        if persisted_metrics_started != metrics_started:
            raise ValueError("persisted metric decision intent differs")
        decision_metrics, decision_gates = _execute_decision_metrics_and_gates(
            table,
            promotion_lock["regime_selection"]["selected_candidate_metrics"],
        )
        metrics_payload = _metrics_payload(
            root=root,
            prediction_commit=prediction_commit,
            metrics_started=metrics_started,
            promotion_lock=promotion_lock,
            metrics=decision_metrics,
            gates=decision_gates,
        )
        _write_json_atomic_exclusive(metrics_path, metrics_payload)
        metrics_payload, _, _ = _verify_metrics_commit(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
    terminal = _terminal_payload(
        root=root,
        ready=ready,
        metrics_payload=metrics_payload,
    )
    _write_json_atomic_exclusive(root / "confirmation.json", terminal)
    return _verify_terminal_confirmation(
        terminal,
        root=root,
        ready=ready,
        promotion_lock=promotion_lock,
        checkpoint=checkpoint,
    )


def confirm_anchor(
    *,
    development_input: str | Path,
    regime_directory: str | Path,
    full_train_input: str | Path,
    output_directory: str | Path,
    verbose: bool = True,
    expected_campaign_lock: ExpectedCampaignLock = PRODUCTION_CAMPAIGN_LOCK,
) -> dict[str, Any]:
    """Consume the 2024 anchor exactly once and return a terminal decision."""

    # Re-verify every pre-access input before even resolving/opening full train.
    frame, development_provenance = load_tuning_frame(development_input)
    regime = verify_regime_selection(
        regime_directory,
        development_frame=frame,
        development_provenance=development_provenance,
        expected_campaign_lock=expected_campaign_lock,
    )
    root = Path(output_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError("anchor preparation directory does not exist")
    ready, promotion_lock, _, checkpoint = _verify_anchor_ready(
        root,
        frame=frame,
        development_provenance=development_provenance,
        expected_campaign_lock=expected_campaign_lock,
    )
    expected_lock = _promotion_lock_payload(
        regime=regime,
        development_provenance=development_provenance,
        expected_campaign_lock=expected_campaign_lock,
    )
    if promotion_lock != expected_lock:
        raise ValueError("promotion lock no longer matches sealed regime selection")
    access_path = root / "access_started.json"
    confirmation_path = root / "confirmation.json"
    prediction_path = root / "anchor_predictions.csv"
    prediction_commit_path = root / "predictions_committed.json"

    if confirmation_path.exists():
        if not access_path.is_file():
            raise ValueError("terminal confirmation exists without an access ledger")
        return _verify_terminal_confirmation(
            _read_json(confirmation_path),
            root=root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )

    if access_path.exists():
        # A prior process crossed the irreversible boundary.  Recovery may
        # reload the checkpoint read-only for integrity checks and read sealed
        # predictions/metrics.  It may never reopen full train, transform rows,
        # run model.forward/_predict, or perform any reinference.
        try:
            _verify_access_ledger(
                access_path,
                ready=ready,
                promotion_lock=promotion_lock,
            )
            if prediction_commit_path.is_file():
                # A valid stage 02 permits stage 03/04 completion from disk.
                _verify_prediction_commit(
                    root,
                    ready=ready,
                    promotion_lock=promotion_lock,
                    checkpoint=checkpoint,
                )
                return _commit_metrics_and_terminal(
                    root,
                    ready=ready,
                    promotion_lock=promotion_lock,
                    checkpoint=checkpoint,
                )
        except Exception:
            pass
        terminal = _derive_inconclusive_payload(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
        _write_json_atomic_exclusive(confirmation_path, terminal)
        return _verify_terminal_confirmation(
            terminal,
            root=root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )

    full_path = Path(full_train_input).resolve()
    development_path = Path(development_input).resolve()
    regime_root = Path(regime_directory).resolve()
    if (
        full_path == development_path
        or full_path == development_path.with_suffix(".manifest.json")
        or full_path == root
        or root in full_path.parents
        or full_path == regime_root
        or regime_root in full_path.parents
    ):
        raise ValueError(
            "full train input must be outside development, regime, and output artifacts"
        )
    if not full_path.is_file():
        raise FileNotFoundError(f"full train input is not a regular file: {full_path}")
    access = _access_started_payload(
        ready=ready,
        ready_file_sha256=_sha256_file(root / "anchor_ready.json"),
        promotion_lock=promotion_lock,
        full_train_path=full_path,
    )
    _write_access_ledger_o_excl(access_path, access)
    access_file_sha256 = _sha256_file(access_path)
    try:
        persisted_access = _verify_access_ledger(
            access_path,
            ready=ready,
            promotion_lock=promotion_lock,
        )
        if persisted_access != access:
            raise ValueError("persisted access ledger differs from claimed access")

        preflight = _season_only_full_source_preflight(full_path)
        if preflight["sha256"] != promotion_lock["expected_full_source_sha256"]:
            raise ValueError("full train source hash differs from development provenance")
        anchor = _extract_anchor_partition(full_path, preflight=preflight)

        target = pd.to_numeric(anchor[TARGET_COL], errors="coerce").to_numpy(
            dtype=np.float64, copy=True
        )
        if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
            raise ValueError("anchor target must contain finite binary 0/1 values")
        feature_frame = anchor.drop(columns=[TARGET_COL])
        preprocessor_state_before = _preprocessor_state_sha256(
            checkpoint.preprocessor
        )
        transformed = checkpoint.preprocessor.transform(feature_frame)
        preprocessor_state_after = _preprocessor_state_sha256(
            checkpoint.preprocessor
        )
        if preprocessor_state_before != preprocessor_state_after:
            raise RuntimeError("anchor transform mutated the prepared preprocessor")
        if "target" in transformed:
            raise RuntimeError("anchor target entered preprocessing output")
        validate_transformed_arrays(
            checkpoint.model.feature_layout,
            transformed,
            expected_row_ids=anchor[ID_COL].to_numpy(dtype=object, copy=False),
        )
        evaluation_config = regime.evaluation_procedure()
        before_state = _model_state_sha256(checkpoint.model)
        probabilities = _predict(
            checkpoint.model,
            transformed["categorical"],
            transformed["numerical"],
            config=evaluation_config,
            device=torch.device("cpu"),
        )
        after_state = _model_state_sha256(checkpoint.model)
        if before_state != after_state:
            raise RuntimeError("anchor prediction mutated the prepared checkpoint")
        prediction_table = pd.DataFrame(
            {
                ID_COL: anchor[ID_COL].to_numpy(copy=True),
                "season": np.full(len(anchor), ANCHOR_SEASON, dtype=np.int16),
                "game_type": anchor["game_type"].astype(str).to_numpy(copy=True),
                "y_true": target.astype(np.int8),
                "prediction": probabilities,
            },
            columns=list(ANCHOR_PREDICTION_COLUMNS),
        )
        if list(prediction_table.columns) != list(ANCHOR_PREDICTION_COLUMNS):
            raise RuntimeError("anchor prediction table columns changed before commit")
        _write_prediction_table_atomic_exclusive(prediction_path, prediction_table)
        persisted_table = _read_prediction_table(prediction_path)
        prediction_commit = _prediction_commit_payload(
            root=root,
            ready=ready,
            access=access,
            regime=regime,
            checkpoint=checkpoint,
            prediction_table=persisted_table,
            source_preflight=preflight,
            model_state_before=before_state,
            model_state_after=after_state,
            preprocessor_state_before=preprocessor_state_before,
            preprocessor_state_after=preprocessor_state_after,
        )
        _write_json_atomic_exclusive(prediction_commit_path, prediction_commit)
        _verify_prediction_commit(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
        return _commit_metrics_and_terminal(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
    except Exception as error:
        # A valid stage 02 before the immutable metric intent may still claim
        # its one decision.  Once metrics_started exists, a missing metrics
        # commit can never execute the decision function again.
        if prediction_commit_path.is_file():
            try:
                _verify_prediction_commit(
                    root,
                    ready=ready,
                    promotion_lock=promotion_lock,
                    checkpoint=checkpoint,
                )
                if not (root / "metrics_started.json").exists() or (
                    root / "metrics.json"
                ).is_file():
                    return _commit_metrics_and_terminal(
                        root,
                        ready=ready,
                        promotion_lock=promotion_lock,
                        checkpoint=checkpoint,
                    )
            except Exception:
                pass
        terminal = _derive_inconclusive_payload(
            root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
        _write_json_atomic_exclusive(confirmation_path, terminal)
        result = _verify_terminal_confirmation(
            terminal,
            root=root,
            ready=ready,
            promotion_lock=promotion_lock,
            checkpoint=checkpoint,
        )
    if verbose:
        print(
            f"[anchor_confirmation] terminal_status={result['status']}",
            flush=True,
        )
    return result


def run_anchor_confirmation(
    *,
    development_input: str | Path,
    regime_directory: str | Path,
    full_train_input: str | Path,
    output_directory: str | Path,
    phase: str,
    resume: bool = False,
    verbose: bool = True,
    expected_campaign_lock: ExpectedCampaignLock = PRODUCTION_CAMPAIGN_LOCK,
) -> dict[str, Any]:
    """Run exactly one explicit phase; there is no combined campaign action."""

    if phase not in {"prepare", "confirm"}:
        raise ValueError("phase must be exactly one of: prepare, confirm")
    if resume and phase != "prepare":
        raise ValueError("resume is permitted only for the prepare phase")
    if phase == "confirm":
        confirmation = confirm_anchor(
            development_input=development_input,
            regime_directory=regime_directory,
            full_train_input=full_train_input,
            output_directory=output_directory,
            verbose=verbose,
            expected_campaign_lock=expected_campaign_lock,
        )
        return {"phase": "confirm", "confirmation": confirmation}
    ready = prepare_anchor_confirmation(
        development_input=development_input,
        regime_directory=regime_directory,
        output_directory=output_directory,
        resume=resume,
        verbose=verbose,
        expected_campaign_lock=expected_campaign_lock,
    )
    return {"phase": "prepare", "anchor_ready": ready}


__all__ = [
    "SCHEMA_VERSION",
    "ANCHOR_SEASON",
    "ANCHOR_SEEDS",
    "FIXED_CONFIRMATION_GATES",
    "METRIC_EXECUTION_POLICY",
    "RECOVERY_POLICY",
    "ExpectedCampaignLock",
    "PRODUCTION_CAMPAIGN_LOCK",
    "VerifiedRegimeSelection",
    "verify_regime_selection",
    "evaluate_confirmation_gates",
    "prepare_anchor_confirmation",
    "confirm_anchor",
    "run_anchor_confirmation",
]
