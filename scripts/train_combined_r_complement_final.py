#!/usr/bin/env python3
"""Fixed-epoch final refit of the combined-signal DCNv2 complement."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_control.auxiliary_targets import recover_fold_train_auxiliary_targets  # noqa: E402
from lg_control.combined_signal_features import CombinedSignalAugmentedPreprocessor  # noqa: E402
from lg_control.multitask_training import MultiTaskLossConfig, train_multitask_model  # noqa: E402
from lg_control.regime_model import (  # noqa: E402
    GatedCrossModelConfig,
    MultiTaskGatedCrossRegimeModel,
    MultiTaskHeadConfig,
)
from lg_control.training import TrainConfig  # noqa: E402


OUTPUT = ROOT / "artifacts/final_combined_r_complement"
SEED = 42
EPOCHS = 3


def save_state(model: torch.nn.Module, path: Path) -> None:
    np.savez_compressed(
        path,
        **{name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()},
    )


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    print(f"final combined complement rows={len(data):,}", flush=True)
    preprocessor = CombinedSignalAugmentedPreprocessor(min_category_frequency=2).fit(data)
    arrays = preprocessor.transform(data)
    model_config = GatedCrossModelConfig(
        embedding_dim_cap=16,
        interaction_dim=64,
        cross_layers=3,
        num_experts=4,
        low_rank=16,
        deep_hidden_dim=96,
        deep_layers=2,
        dropout=0.10,
    )
    head_config = MultiTaskHeadConfig(hidden_dim=64, dropout=0.10)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = MultiTaskGatedCrossRegimeModel(
            preprocessor.feature_layout(), model_config, head_config
        )
    age = 2024 - data.season.to_numpy(np.int64)
    is_f = data.game_type.astype(str).str.upper().eq("F").to_numpy()
    decay = np.where(is_f, 0.08, 0.60)
    weights = np.power(decay, age).astype(np.float32)
    weights /= float(weights.mean())
    target = data.control_success.to_numpy(np.float32)
    rate = float(np.average(target, weights=weights))
    with torch.no_grad():
        model.output_bias.fill_(math.log(rate / (1.0 - rate)))
    auxiliary = recover_fold_train_auxiliary_targets(data, partition="fold_train")
    auxiliary_target = auxiliary[
        ["aux_reverse", "aux_middle", "aux_ball", "aux_strike"]
    ].to_numpy(np.float32)
    auxiliary_mask = auxiliary.auxiliary_valid_mask.to_numpy(np.float32)
    auxiliary_rate = np.array(
        [
            np.average(
                np.nan_to_num(auxiliary_target[:, position], nan=0.0),
                weights=weights * auxiliary_mask,
            )
            for position in range(4)
        ],
        dtype=np.float32,
    )
    auxiliary_rate = np.clip(auxiliary_rate, 1e-5, 1.0 - 1e-5)
    with torch.no_grad():
        model.auxiliary_head.bias.copy_(
            torch.from_numpy(np.log(auxiliary_rate / (1.0 - auxiliary_rate)))
        )
    config = TrainConfig(
        epochs=EPOCHS,
        scheduler_epochs=4,
        batch_size=8192,
        eval_batch_size=32768,
        learning_rate=8e-4,
        weight_decay=2e-5,
        brier_weight=1.0,
        patience=3,
        seed=SEED,
        num_workers=0,
        use_amp=True,
        verbose=True,
    )
    # The small in-training evaluation slice is diagnostic only.  `last`
    # forces the preselected third epoch and cannot select on these rows.
    diagnostic = np.arange(min(4096, len(data)), dtype=np.int64)
    result = train_multitask_model(
        model,
        arrays["categorical"],
        arrays["numerical"],
        target,
        auxiliary_target,
        auxiliary_mask,
        arrays["categorical"][diagnostic],
        arrays["numerical"][diagnostic],
        target[diagnostic],
        config=config,
        loss_config=MultiTaskLossConfig(auxiliary_weight=0.15, consistency_weight=0.02),
        device="cpu",
        sample_weights=weights,
        checkpoint_mode="last",
    )
    (OUTPUT / "preprocessor.json").write_text(
        json.dumps(preprocessor.to_state_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "model_config.json").write_text(
        json.dumps(model.get_config(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_state(model, OUTPUT / "model_state.npz")
    manifest = {
        "kind": "combined_signal_dcnv2_multitask_complement_v1",
        "training_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "training_rows": int(len(data)),
        "epochs": EPOCHS,
        "scheduler_epochs": 4,
        "checkpoint_mode": "last",
        "seed": SEED,
        "r_decay": 0.60,
        "f_decay": 0.08,
        "weighted_target_rate": rate,
        "history": result.history,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
