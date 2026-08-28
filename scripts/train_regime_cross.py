#!/usr/bin/env python3
"""Train the regime-aware DCNv2/MoE candidate on a forward season fold."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import ID_COL, TARGET_COL
from lg_control.combined_signal_features import CombinedSignalAugmentedPreprocessor
from lg_control.auxiliary_targets import recover_fold_train_auxiliary_targets
from lg_control.exclusive_outcome import (
    OUTCOME_NAMES,
    ExclusiveOutcomeGatedCrossModel,
    ExclusiveOutcomeHeadConfig,
    ExclusiveOutcomeLossConfig,
    build_exclusive_outcome_model,
    recover_fold_train_exclusive_targets,
    train_exclusive_outcome_model,
)
from lg_control.metrics import metric_dict
from lg_control.multitask_training import MultiTaskLossConfig, train_multitask_model
from lg_control.numerical_embedding_model import (
    PeriodicPiecewiseModelConfig,
    build_periodic_piecewise_model,
)
from lg_control.preprocessing import DeepTabularPreprocessor
from lg_control.rate_basis_features import RateBasisAugmentedPreprocessor
from lg_control.regime_split_head import build_regime_split_model
from lg_control.regime_model import (
    AUXILIARY_TARGET_NAMES,
    GatedCrossModelConfig,
    GatedCrossRegimeModel,
    MultiTaskGatedCrossRegimeModel,
    MultiTaskHeadConfig,
    StructuredMultiTaskGatedCrossRegimeModel,
    StructuredResidualConfig,
    build_regime_model,
    build_multitask_regime_model,
    build_structured_multitask_regime_model,
)
from lg_control.regime_pipeline import RegimeAugmentedPreprocessor
from lg_control.recent_regime import (
    RecentRegimePolicy,
    recent_regime_audit,
    recent_regime_weights,
    select_recent_regime_inner_partitions,
    select_recent_regime_training_rows,
)
from lg_control.temporal_target_features import TemporalTargetAugmentedPreprocessor
from lg_control.temporal_aux_features import TemporalAuxAugmentedPreprocessor
from lg_control.targeted_prior_features import (
    TARGETED_PRIOR_FEATURE_COLUMNS,
    TargetedPriorAugmentedPreprocessor,
)
from lg_control.strict_trackman import StrictTrackmanFeatureBuilder
from lg_control.trackman_pipeline import StrictTrackmanAugmentedPreprocessor
from lg_control.schema import validate_input_frame
from lg_control.training import TrainConfig, train_model


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def decay(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("decay must be in (0, 1]")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument(
        "--min-train-year",
        type=int,
        default=None,
        help="Optional inclusive lower season bound for recent-only ablations",
    )
    parser.add_argument(
        "--training-policy",
        choices=("legacy_all", "recent_only", "stable_r_plus_recent"),
        default="legacy_all",
        help=(
            "Training-row contract. Recent policies hard-exclude game_type=F "
            "before --regime-start-year."
        ),
    )
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument(
        "--inner-validation-start-month",
        type=int,
        default=None,
        help=(
            "Optional within-regime chronological stop split. With 2023 and 7, "
            "fit months 4--6 and validate months 7--10 without opening 2024."
        ),
    )
    parser.add_argument(
        "--recent-decay",
        type=decay,
        default=0.85,
        help="Per-year weight decay inside the 2023+ regime.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/target1200_cross")
    )
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument(
        "--scheduler-epochs",
        type=positive_int,
        default=None,
        help="Optional cosine-schedule horizon kept fixed during nested refits",
    )
    parser.add_argument("--patience", type=positive_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=8192)
    parser.add_argument("--eval-batch-size", type=positive_int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=positive_int, default=8)
    parser.add_argument("--min-category-frequency", type=positive_int, default=2)
    parser.add_argument(
        "--architecture",
        choices=("cross", "periodic", "split"),
        default="cross",
        help="Cross/MoE backbone or nonlinear numerical-embedding residual tower",
    )
    parser.add_argument(
        "--feature-set",
        choices=(
            "base",
            "rate_basis",
            "combined",
            "temporal_aux",
            "targeted_prior",
            "v2",
            "temporal_te",
            "trackman_te",
        ),
        default="v2",
        help="canonical inputs or one leakage-safe augmented feature family",
    )
    parser.add_argument("--r-decay", type=decay, default=0.60)
    parser.add_argument("--f-decay", type=decay, default=0.08)
    parser.add_argument("--embedding-dim-cap", type=positive_int, default=16)
    parser.add_argument("--interaction-dim", type=positive_int, default=96)
    parser.add_argument("--cross-layers", type=positive_int, default=3)
    parser.add_argument("--num-experts", type=positive_int, default=4)
    parser.add_argument("--low-rank", type=positive_int, default=24)
    parser.add_argument("--deep-hidden-dim", type=positive_int, default=128)
    parser.add_argument("--deep-layers", type=positive_int, default=2)
    parser.add_argument("--dropout", type=probability, default=0.10)
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument(
        "--exclusive-outcome",
        action="store_true",
        help="Train an exhaustive five-class success/failure-state softmax",
    )
    parser.add_argument("--ball-strike-weight", type=probability, default=0.10)
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--structured-correction-cap", type=float, default=0.25)
    parser.add_argument("--split-correction-cap", type=float, default=0.35)
    parser.add_argument("--split-hidden-dim", type=positive_int, default=32)
    parser.add_argument("--auxiliary-hidden-dim", type=positive_int, default=64)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--consistency-weight", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-mode",
        choices=("best", "last"),
        default="best",
        help="Use validation-best checkpoint or the final preselected epoch",
    )
    parser.add_argument(
        "--save-head-probabilities",
        action="store_true",
        help="Persist auxiliary/outcome probabilities for residual meta-models",
    )
    parser.add_argument(
        "--initial-model-state",
        type=Path,
        default=None,
        help="Optional strict NPZ checkpoint used to continue a compatible run",
    )
    parser.add_argument(
        "--trackman-state",
        type=Path,
        default=None,
        help="StrictTrackmanFeatureBuilder JSON, required by feature-set=trackman_te",
    )
    parser.add_argument("--max-train-rows", type=positive_int, default=None)
    parser.add_argument("--max-val-rows", type=positive_int, default=None)
    return parser.parse_args()


def deterministic_limit(
    frame: pd.DataFrame, limit: int | None, *, seed: int
) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.copy()
    generator = np.random.default_rng(seed)
    positions = np.sort(generator.choice(len(frame), size=limit, replace=False))
    return frame.iloc[positions].copy()


def regime_weights(
    frame: pd.DataFrame,
    *,
    validation_year: int,
    r_decay: float,
    f_decay: float,
) -> np.ndarray:
    seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int64)
    age = (validation_year - 1) - seasons
    if (age < 0).any():
        raise ValueError("training rows must precede the validation season")
    is_f = frame["game_type"].astype("string").str.upper().eq("F").to_numpy()
    per_row_decay = np.where(is_f, f_decay, r_decay)
    weights = np.power(per_row_decay, age).astype(np.float32)
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("regime weights must be finite and positive")
    return weights / float(weights.mean())


def sliced_metrics(
    validation: pd.DataFrame, probabilities: np.ndarray
) -> dict[str, dict[str, float | int]]:
    target = validation[TARGET_COL].to_numpy(dtype=np.float64)
    game_type = validation["game_type"].astype("string").str.upper()
    masks = {
        "all": np.ones(len(validation), dtype=bool),
        "game_type_F": game_type.eq("F").to_numpy(),
        "game_type_R": game_type.eq("R").to_numpy(),
        "pitcher_cold_start": validation["asof_pitcher_n"].eq(0).to_numpy(),
        "pitcher_recent_missing": validation[
            "asof_pitcher_prev5_game_success_rate"
        ].isna().to_numpy(),
    }
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        result[name] = {
            "rows": int(mask.sum()),
            "target_rate": float(target[mask].mean()),
            "prediction_mean": float(probabilities[mask].mean()),
            **metric_dict(target[mask], probabilities[mask]),
        }
    return result


def save_state_npz(model: torch.nn.Module, path: Path) -> None:
    state = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    np.savez_compressed(path, **state)


def load_state_npz(model: torch.nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"initial model state does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    incompatibilities = model.load_state_dict(state, strict=True)
    if incompatibilities.missing_keys or incompatibilities.unexpected_keys:
        raise ValueError("initial model state is incompatible with the requested model")


def predict_head_probabilities(
    model: torch.nn.Module,
    categorical: np.ndarray,
    numerical: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Return row-aligned auxiliary probabilities from the selected checkpoint."""

    if not isinstance(
        model, (ExclusiveOutcomeGatedCrossModel, MultiTaskGatedCrossRegimeModel)
    ):
        return {}
    target_device = torch.device(device)
    model.to(target_device)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        torch.from_numpy(np.asarray(numerical, dtype=np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pieces: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.inference_mode():
        for cat_batch, num_batch in loader:
            outputs = model.forward_with_aux(
                cat_batch.to(target_device), num_batch.to(target_device)
            )
            if isinstance(model, ExclusiveOutcomeGatedCrossModel):
                outcome = outputs["outcome_probabilities"].float().cpu().numpy()
                for position, name in enumerate(OUTCOME_NAMES):
                    pieces.setdefault(f"head_outcome_{name}", []).append(
                        outcome[:, position]
                    )
                if "ball_strike_logits" in outputs:
                    auxiliary = (
                        torch.sigmoid(outputs["ball_strike_logits"])
                        .float()
                        .cpu()
                        .numpy()
                    )
                    for position, name in enumerate(("ball", "strike")):
                        pieces.setdefault(f"head_aux_{name}", []).append(
                            auxiliary[:, position]
                        )
            else:
                auxiliary = (
                    torch.sigmoid(outputs["auxiliary_logits"])
                    .float()
                    .cpu()
                    .numpy()
                )
                for position, name in enumerate(AUXILIARY_TARGET_NAMES):
                    pieces.setdefault(f"head_aux_{name}", []).append(
                        auxiliary[:, position]
                    )
    return {
        name: np.concatenate(values).astype(np.float32, copy=False)
        for name, values in pieces.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    data = pd.read_csv(args.input, low_memory=False)
    validate_input_frame(data, require_target=True, allow_target=True)
    policy = None
    weight_cutoff_year = args.validation_year
    if args.inner_validation_start_month is not None:
        if args.training_policy == "legacy_all":
            raise ValueError("inner validation requires a recent training policy")
        if args.min_train_year is not None:
            raise ValueError("--min-train-year cannot be combined with inner validation")
        policy = RecentRegimePolicy(
            regime_start_year=args.regime_start_year,
            history_policy=args.training_policy,
            historical_r_decay=args.r_decay,
            recent_decay=args.recent_decay,
        )
        train, validation = select_recent_regime_inner_partitions(
            data,
            validation_year=args.validation_year,
            validation_start_month=args.inner_validation_start_month,
            policy=policy,
        )
        # Makes the current new-regime season age zero while historical R is
        # decayed exactly as it will be for the 2024 outer refit.
        weight_cutoff_year = args.validation_year + 1
    elif args.training_policy == "legacy_all":
        train = data.loc[data["season"] < args.validation_year]
        if args.min_train_year is not None:
            train = train.loc[train["season"] >= args.min_train_year]
    else:
        if args.min_train_year is not None:
            raise ValueError("--min-train-year cannot be combined with --training-policy")
        policy = RecentRegimePolicy(
            regime_start_year=args.regime_start_year,
            history_policy=args.training_policy,
            historical_r_decay=args.r_decay,
            recent_decay=args.recent_decay,
        )
        train = select_recent_regime_training_rows(
            data, cutoff_year=args.validation_year, policy=policy
        )
    if args.inner_validation_start_month is None:
        validation = data.loc[data["season"] == args.validation_year]
    if train.empty or validation.empty:
        raise ValueError("forward split produced an empty train or validation partition")
    train = deterministic_limit(train, args.max_train_rows, seed=args.seed)
    validation = deterministic_limit(
        validation, args.max_val_rows, seed=args.seed + 1
    )
    print(
        f"fold={args.validation_year} train_rows={len(train):,} "
        f"validation_rows={len(validation):,}",
        flush=True,
    )

    if args.feature_set == "targeted_prior":
        preprocessor = TargetedPriorAugmentedPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train, partition="fold_train")
    elif args.feature_set == "temporal_aux":
        preprocessor = TemporalAuxAugmentedPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train)
    elif args.feature_set == "combined":
        preprocessor = CombinedSignalAugmentedPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train)
    elif args.feature_set == "rate_basis":
        preprocessor = RateBasisAugmentedPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train)
    elif args.feature_set == "v2":
        preprocessor = RegimeAugmentedPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train)
    elif args.feature_set == "temporal_te":
        preprocessor = TemporalTargetAugmentedPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train)
    elif args.feature_set == "trackman_te":
        if args.trackman_state is None:
            raise ValueError("--trackman-state is required for feature-set=trackman_te")
        preprocessor = StrictTrackmanAugmentedPreprocessor(
            StrictTrackmanFeatureBuilder.load(args.trackman_state),
            min_category_frequency=args.min_category_frequency,
        ).fit(train)
    else:
        preprocessor = DeepTabularPreprocessor(
            min_category_frequency=args.min_category_frequency
        ).fit(train)
    train_arrays = preprocessor.transform(train)
    validation_arrays = preprocessor.transform(validation)
    model_config = GatedCrossModelConfig(
        embedding_dim_cap=args.embedding_dim_cap,
        interaction_dim=args.interaction_dim,
        cross_layers=args.cross_layers,
        num_experts=args.num_experts,
        low_rank=args.low_rank,
        deep_hidden_dim=args.deep_hidden_dim,
        deep_layers=args.deep_layers,
        dropout=args.dropout,
    )
    periodic_config = PeriodicPiecewiseModelConfig(
        embedding_dim_cap=args.embedding_dim_cap,
        hidden_dim=args.deep_hidden_dim,
        ffn_dim=max(args.deep_hidden_dim, 128),
        num_blocks=args.deep_layers,
        dropout=args.dropout,
    )
    multitask_config = MultiTaskHeadConfig(
        hidden_dim=args.auxiliary_hidden_dim, dropout=args.dropout
    )
    exclusive_head_config = ExclusiveOutcomeHeadConfig(
        hidden_dim=args.auxiliary_hidden_dim,
        dropout=args.dropout,
        use_ball_strike_auxiliary=True,
    )
    if args.structured and not args.multitask:
        raise ValueError("--structured requires --multitask")
    if args.exclusive_outcome and (args.multitask or args.structured):
        raise ValueError("--exclusive-outcome cannot be combined with multi-task modes")
    if args.architecture in {"periodic", "split"} and (
        args.multitask or args.structured or args.exclusive_outcome
    ):
        raise ValueError(f"{args.architecture} architecture supports single-task training")
    structured_config = StructuredResidualConfig(
        correction_cap=args.structured_correction_cap
    )
    if args.architecture == "periodic":
        layout_or_preprocessor = (
            preprocessor.feature_layout()
            if hasattr(preprocessor, "feature_layout")
            else preprocessor
        )
        model = build_periodic_piecewise_model(
            layout_or_preprocessor,
            config=periodic_config,
            initialization_seed=args.seed,
        )
    elif args.architecture == "split":
        targeted_names = (
            TARGETED_PRIOR_FEATURE_COLUMNS
            if args.feature_set == "targeted_prior"
            else ()
        )
        model = build_regime_split_model(
            preprocessor,
            config=model_config,
            hidden_dim=args.split_hidden_dim,
            dropout=min(args.dropout, 0.10),
            correction_cap=args.split_correction_cap,
            targeted_numerical_names=targeted_names,
            initialization_seed=args.seed,
        )
    elif isinstance(
        preprocessor,
        (
            RegimeAugmentedPreprocessor,
            RateBasisAugmentedPreprocessor,
            CombinedSignalAugmentedPreprocessor,
            TemporalAuxAugmentedPreprocessor,
            TargetedPriorAugmentedPreprocessor,
            TemporalTargetAugmentedPreprocessor,
            StrictTrackmanAugmentedPreprocessor,
        ),
    ):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            if args.exclusive_outcome:
                model = ExclusiveOutcomeGatedCrossModel(
                    preprocessor.feature_layout(), model_config, exclusive_head_config
                )
            elif args.structured:
                model = StructuredMultiTaskGatedCrossRegimeModel(
                    preprocessor.feature_layout(),
                    model_config,
                    multitask_config,
                    structured_config,
                )
            elif args.multitask:
                model = MultiTaskGatedCrossRegimeModel(
                    preprocessor.feature_layout(), model_config, multitask_config
                )
            else:
                model = GatedCrossRegimeModel(preprocessor.feature_layout(), model_config)
    else:
        if args.exclusive_outcome:
            model = build_exclusive_outcome_model(
                preprocessor,
                config=model_config,
                head_config=exclusive_head_config,
                initialization_seed=args.seed,
            )
        elif args.structured:
            model = build_structured_multitask_regime_model(
                preprocessor,
                config=model_config,
                multitask_config=multitask_config,
                structured_config=structured_config,
                initialization_seed=args.seed,
            )
        elif args.multitask:
            model = build_multitask_regime_model(
                preprocessor,
                config=model_config,
                multitask_config=multitask_config,
                initialization_seed=args.seed,
            )
        else:
            model = build_regime_model(
                preprocessor, config=model_config, initialization_seed=args.seed
            )
    if policy is None:
        weights = regime_weights(
            train,
            validation_year=args.validation_year,
            r_decay=args.r_decay,
            f_decay=args.f_decay,
        )
        training_audit = None
    else:
        weights = recent_regime_weights(
            train, cutoff_year=weight_cutoff_year, policy=policy
        )
        training_audit = recent_regime_audit(train, weights, policy=policy)
    weighted_rate = float(
        np.average(train[TARGET_COL].to_numpy(dtype=np.float64), weights=weights)
    )
    weighted_rate = float(np.clip(weighted_rate, 1e-5, 1.0 - 1e-5))
    with torch.no_grad():
        model.output_bias.fill_(math.log(weighted_rate / (1.0 - weighted_rate)))
    auxiliary_frame = None
    exclusive_targets = None
    if args.exclusive_outcome:
        assert isinstance(model, ExclusiveOutcomeGatedCrossModel)
        exclusive_targets = recover_fold_train_exclusive_targets(
            train, partition="fold_train"
        )
        observed = exclusive_targets.outcome_mask
        class_weight = weights[observed]
        class_indices = exclusive_targets.class_indices[observed]
        class_counts = np.bincount(
            class_indices,
            weights=class_weight,
            minlength=len(OUTCOME_NAMES),
        ).astype(np.float64)
        class_rates = np.clip(class_counts / class_counts.sum(), 1e-6, 1.0)
        ball_strike_rates = np.array(
            [
                np.average(
                    exclusive_targets.ball_strike_targets[observed, position],
                    weights=class_weight,
                )
                for position in range(2)
            ],
            dtype=np.float32,
        )
        ball_strike_rates = np.clip(ball_strike_rates, 1e-5, 1.0 - 1e-5)
        with torch.no_grad():
            # The five-class bias, rather than a binary base bias, defines the
            # initial exhaustive simplex.
            model.output_bias.zero_()
            model.outcome_head.bias.copy_(
                torch.from_numpy(np.log(class_rates).astype(np.float32))
            )
            if model.ball_strike_head is not None:
                model.ball_strike_head.bias.copy_(
                    torch.from_numpy(
                        np.log(ball_strike_rates / (1.0 - ball_strike_rates))
                    )
                )
    elif args.multitask:
        auxiliary_frame = recover_fold_train_auxiliary_targets(
            train, partition="fold_train"
        )
        auxiliary_columns = [
            "aux_reverse",
            "aux_middle",
            "aux_ball",
            "aux_strike",
        ]
        auxiliary_target = auxiliary_frame[auxiliary_columns].to_numpy(dtype=np.float32)
        auxiliary_mask = auxiliary_frame["auxiliary_valid_mask"].to_numpy(
            dtype=np.float32
        )
        weighted_auxiliary_rate = np.array(
            [
                np.average(
                    np.nan_to_num(auxiliary_target[:, position], nan=0.0),
                    weights=weights * auxiliary_mask,
                )
                for position in range(auxiliary_target.shape[1])
            ],
            dtype=np.float32,
        )
        weighted_auxiliary_rate = np.clip(weighted_auxiliary_rate, 1e-5, 1.0 - 1e-5)
        assert isinstance(model, MultiTaskGatedCrossRegimeModel)
        with torch.no_grad():
            model.auxiliary_head.bias.copy_(
                torch.from_numpy(
                    np.log(weighted_auxiliary_rate / (1.0 - weighted_auxiliary_rate))
                )
            )

    if args.initial_model_state is not None:
        load_state_npz(model, args.initial_model_state)
        print(f"loaded_initial_state={args.initial_model_state}", flush=True)

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        brier_weight=args.brier_weight,
        patience=args.patience,
        seed=args.seed,
        num_workers=0,
        use_amp=True,
        verbose=True,
        scheduler_epochs=args.scheduler_epochs,
    )
    if args.exclusive_outcome:
        assert exclusive_targets is not None
        assert isinstance(model, ExclusiveOutcomeGatedCrossModel)
        result = train_exclusive_outcome_model(
            model,
            train_arrays["categorical"],
            train_arrays["numerical"],
            exclusive_targets,
            validation_arrays["categorical"],
            validation_arrays["numerical"],
            validation[TARGET_COL].to_numpy(dtype=np.float32),
            config=train_config,
            loss_config=ExclusiveOutcomeLossConfig(
                ball_strike_weight=args.ball_strike_weight
            ),
            device=args.device,
            sample_weights=weights,
            checkpoint_mode=args.checkpoint_mode,
        )
    elif args.multitask:
        assert auxiliary_frame is not None
        assert isinstance(model, MultiTaskGatedCrossRegimeModel)
        result = train_multitask_model(
            model,
            train_arrays["categorical"],
            train_arrays["numerical"],
            train[TARGET_COL].to_numpy(dtype=np.float32),
            auxiliary_target,
            auxiliary_mask,
            validation_arrays["categorical"],
            validation_arrays["numerical"],
            validation[TARGET_COL].to_numpy(dtype=np.float32),
            config=train_config,
            loss_config=MultiTaskLossConfig(
                auxiliary_weight=args.auxiliary_weight,
                consistency_weight=args.consistency_weight,
            ),
            device=args.device,
            sample_weights=weights,
            checkpoint_mode=args.checkpoint_mode,
        )
    else:
        result = train_model(
            model,
            train_arrays["categorical"],
            train_arrays["numerical"],
            train[TARGET_COL].to_numpy(dtype=np.float32),
            validation_arrays["categorical"],
            validation_arrays["numerical"],
            validation[TARGET_COL].to_numpy(dtype=np.float32),
            config=train_config,
            device=args.device,
            sample_weights=weights,
            checkpoint_mode=args.checkpoint_mode,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            ID_COL: validation[ID_COL].to_numpy(copy=True),
            "season": validation["season"].to_numpy(copy=True),
            "game_type": validation["game_type"].to_numpy(copy=True),
            "y_true": validation[TARGET_COL].to_numpy(dtype=np.int8),
            "prediction": result.probabilities,
        }
    )
    if args.save_head_probabilities:
        head_probabilities = predict_head_probabilities(
            model,
            validation_arrays["categorical"],
            validation_arrays["numerical"],
            batch_size=args.eval_batch_size,
            device=args.device,
        )
        for name, values in head_probabilities.items():
            if len(values) != len(predictions):
                raise RuntimeError(f"head probability rows differ for {name}")
            predictions[name] = values
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "preprocessor.json").write_text(
        json.dumps(preprocessor.to_state_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "model_config.json").write_text(
        json.dumps(model.get_config(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_state_npz(model, args.output_dir / "model_state.npz")
    payload: dict[str, object] = {
        "validation_year": args.validation_year,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_seasons": sorted(int(value) for value in train["season"].unique()),
        "best_epoch": result.best_epoch,
        "checkpoint_mode": args.checkpoint_mode,
        "metrics": result.best_metrics,
        "slices": sliced_metrics(validation, result.probabilities),
        "history": result.history,
        "weighted_training_target_rate": weighted_rate,
        "regime_weighting": {"r_decay": args.r_decay, "f_decay": args.f_decay},
        "training_policy": args.training_policy,
        "inner_validation_start_month": args.inner_validation_start_month,
        "weight_cutoff_year": weight_cutoff_year,
        "recent_regime_audit": training_audit,
        "feature_set": args.feature_set,
        "architecture": args.architecture,
        "multitask": args.multitask,
        "exclusive_outcome": args.exclusive_outcome,
        "structured": args.structured,
        "exclusive_outcome_loss": (
            {
                "ball_strike_weight": args.ball_strike_weight,
                "valid_rows": int(exclusive_targets.outcome_mask.sum()),
            }
            if exclusive_targets is not None
            else None
        ),
        "multitask_loss": (
            {
                "auxiliary_weight": args.auxiliary_weight,
                "consistency_weight": args.consistency_weight,
                "valid_rows": int(auxiliary_frame["auxiliary_valid_mask"].sum()),
            }
            if auxiliary_frame is not None
            else None
        ),
        "model_config": model.get_config(),
        "parameter_report": model.parameter_report(),
        "train_config": train_config.to_dict(),
        "initial_model_state": (
            str(args.initial_model_state) if args.initial_model_state is not None else None
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best_epoch={result.best_epoch} "
        f"brier={result.best_metrics['brier']:.8f} "
        f"score={result.best_metrics['competition_score']:.2f}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run(parse_args())
