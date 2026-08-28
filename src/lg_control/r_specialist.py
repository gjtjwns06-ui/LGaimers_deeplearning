"""Row-local R specialist with a parameter-efficient TabM-style ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .exclusive_outcome import BALL_STRIKE_NAMES, OUTCOME_NAMES, ExclusiveOutcomeTargets
from .model import TabularFeatureLayout
from .rate_basis_features import RateBasisAugmentedPreprocessor


@dataclass(frozen=True)
class RSpecialistConfig:
    embedding_dim_cap: int = 12
    hidden_dim: int = 96
    ensemble_size: int = 4
    hidden_layers: int = 2
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if min(
            self.embedding_dim_cap,
            self.hidden_dim,
            self.ensemble_size,
            self.hidden_layers,
        ) < 1:
            raise ValueError("R specialist dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("R specialist dropout must be in [0, 1)")


@dataclass(frozen=True)
class RSpecialistTrainConfig:
    epochs: int = 5
    batch_size: int = 4096
    eval_batch_size: int = 16384
    learning_rate: float = 8e-4
    weight_decay: float = 2e-4
    brier_weight: float = 2.0
    auxiliary_weight: float = 0.10
    seed: int = 42
    gradient_clip_norm: float = 3.0
    num_workers: int = 0
    verbose: bool = True


def _embedding_dim(cardinality: int, cap: int) -> int:
    return min(cap, max(4, int(math.ceil(math.log2(max(2, cardinality))))))


class RSpecialistTabM(nn.Module):
    """Shared nonlinear trunk with independently modulated prediction members."""

    architecture_name = "r_specialist_tabm_exclusive_v1"

    def __init__(
        self,
        layout: TabularFeatureLayout,
        config: RSpecialistConfig | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config or RSpecialistConfig()
        dimensions = [
            _embedding_dim(value, self.config.embedding_dim_cap)
            for value in layout.categorical_cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, width) for cardinality, width in zip(layout.categorical_cardinalities, dimensions)]
        )
        input_dim = sum(dimensions) + len(layout.numerical_names)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, self.config.hidden_dim)
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.member_scale = nn.Parameter(
            torch.ones(self.config.ensemble_size, self.config.hidden_dim)
        )
        self.member_bias = nn.Parameter(
            torch.zeros(self.config.ensemble_size, self.config.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.config.hidden_dim),
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                    nn.Dropout(self.config.dropout),
                )
                for _ in range(self.config.hidden_layers)
            ]
        )
        k, h = self.config.ensemble_size, self.config.hidden_dim
        self.outcome_weight = nn.Parameter(torch.empty(k, h, len(OUTCOME_NAMES)))
        self.outcome_bias = nn.Parameter(torch.zeros(k, len(OUTCOME_NAMES)))
        self.auxiliary_weight = nn.Parameter(torch.empty(k, h, len(BALL_STRIKE_NAMES)))
        self.auxiliary_bias = nn.Parameter(torch.zeros(k, len(BALL_STRIKE_NAMES)))
        self.numeric_wide = nn.Linear(len(layout.numerical_names), len(OUTCOME_NAMES), bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.ones_(self.member_scale)
        with torch.no_grad():
            self.member_scale.add_(0.02 * torch.randn_like(self.member_scale))
        nn.init.zeros_(self.member_bias)
        for block in self.blocks:
            nn.init.ones_(block[0].weight)
            nn.init.zeros_(block[0].bias)
            nn.init.xavier_uniform_(block[1].weight)
            nn.init.zeros_(block[1].bias)
            nn.init.xavier_uniform_(block[4].weight, gain=0.5)
            nn.init.zeros_(block[4].bias)
        nn.init.normal_(self.outcome_weight, std=0.01)
        nn.init.zeros_(self.outcome_bias)
        nn.init.normal_(self.auxiliary_weight, std=0.01)
        nn.init.zeros_(self.auxiliary_bias)
        nn.init.normal_(self.numeric_wide.weight, std=0.005)

    def forward_with_aux(self, categorical: Tensor, numerical: Tensor) -> dict[str, Tensor]:
        if categorical.ndim != 2 or numerical.ndim != 2:
            raise ValueError("R specialist inputs must be two-dimensional")
        if categorical.shape[1] != len(self.embeddings):
            raise ValueError("R specialist categorical width differs")
        if numerical.shape[1] != len(self.layout.numerical_names):
            raise ValueError("R specialist numerical width differs")
        embedded = [layer(categorical[:, index]) for index, layer in enumerate(self.embeddings)]
        combined = torch.cat([*embedded, numerical], dim=1)
        shared = F.silu(self.input_projection(self.input_norm(combined)))
        shared = self.input_dropout(shared)
        hidden = shared[:, None, :] * self.member_scale[None, :, :] + self.member_bias[None, :, :]
        for block in self.blocks:
            hidden = hidden + block(hidden)
        outcome_logits = torch.einsum("bkh,kho->bko", hidden, self.outcome_weight)
        outcome_logits = outcome_logits + self.outcome_bias[None, :, :]
        outcome_logits = outcome_logits + self.numeric_wide(numerical)[:, None, :]
        member_outcome = torch.softmax(outcome_logits, dim=-1)
        outcome_probability = member_outcome.mean(dim=1)
        log_outcome_probability = torch.log(outcome_probability.clamp_min(1e-7))
        auxiliary_logits = torch.einsum("bkh,kha->bka", hidden, self.auxiliary_weight)
        auxiliary_logits = auxiliary_logits + self.auxiliary_bias[None, :, :]
        auxiliary_probability = torch.sigmoid(auxiliary_logits).mean(dim=1)
        return {
            "outcome_logits": log_outcome_probability,
            "outcome_probabilities": outcome_probability,
            "success_probability": outcome_probability[:, 0],
            "auxiliary_probabilities": auxiliary_probability,
        }

    def forward(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        probability = self.forward_with_aux(categorical, numerical)["success_probability"]
        return torch.logit(probability.clamp(1e-7, 1.0 - 1e-7))

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture_name,
            "layout": self.layout.to_dict(),
            "model": asdict(self.config),
        }

    @classmethod
    def from_config_dict(cls, payload: Mapping[str, Any]) -> "RSpecialistTabM":
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError("unsupported R specialist architecture")
        return cls(
            TabularFeatureLayout.from_dict(payload["layout"]),
            RSpecialistConfig(**payload["model"]),
        )


def initialize_priors(
    model: RSpecialistTabM,
    targets: ExclusiveOutcomeTargets,
    weights: np.ndarray,
) -> None:
    observed = np.asarray(targets.outcome_mask, dtype=bool)
    classes = np.asarray(targets.class_indices, dtype=np.int64)[observed]
    sample_weight = np.asarray(weights, dtype=np.float64)[observed]
    counts = np.bincount(classes, weights=sample_weight, minlength=len(OUTCOME_NAMES))
    rates = np.clip(counts / counts.sum(), 1e-6, 1.0)
    aux = np.array(
        [
            np.average(targets.ball_strike_targets[observed, index], weights=sample_weight)
            for index in range(len(BALL_STRIKE_NAMES))
        ]
    )
    with torch.no_grad():
        model.outcome_bias.copy_(
            torch.from_numpy(np.tile(np.log(rates), (model.config.ensemble_size, 1))).float()
        )
        model.auxiliary_bias.copy_(
            torch.from_numpy(
                np.tile(np.log(aux / (1.0 - aux)), (model.config.ensemble_size, 1))
            ).float()
        )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_r_specialist_fixed_epochs(
    model: RSpecialistTabM,
    categorical: np.ndarray,
    numerical: np.ndarray,
    targets: ExclusiveOutcomeTargets,
    weights: np.ndarray,
    *,
    config: RSpecialistTrainConfig,
    device: str = "cpu",
) -> list[dict[str, float]]:
    rows = len(categorical)
    sample_weight = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(sample_weight) != rows or np.any(sample_weight <= 0.0):
        raise ValueError("R specialist weights must be positive and row-aligned")
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        torch.from_numpy(np.asarray(numerical, dtype=np.float32)),
        torch.from_numpy(targets.binary_success.astype(np.float32)),
        torch.from_numpy(targets.class_indices.astype(np.int64)),
        torch.from_numpy(targets.outcome_mask.astype(bool)),
        torch.from_numpy(targets.ball_strike_targets.astype(np.float32)),
        torch.from_numpy(targets.ball_strike_mask.astype(bool)),
        torch.from_numpy(sample_weight),
    )
    _seed(config.seed)
    target_device = torch.device(device)
    model.to(target_device)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.05
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        total = 0.0
        for batch in loader:
            cat, num, success, classes, observed, auxiliary, aux_mask, weight = (
                value.to(target_device) for value in batch
            )
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_with_aux(cat, num)
            safe_class = torch.clamp(classes, min=0)
            ce = F.nll_loss(output["outcome_logits"], safe_class, reduction="none")
            fallback = F.binary_cross_entropy(
                output["success_probability"].clamp(1e-7, 1 - 1e-7), success, reduction="none"
            )
            classification = torch.where(observed, ce, fallback)
            denominator = weight.sum().clamp_min(1e-8)
            classification_loss = (classification * weight).sum() / denominator
            brier_loss = (torch.square(output["success_probability"] - success) * weight).sum() / denominator
            aux_element = F.binary_cross_entropy(
                output["auxiliary_probabilities"].clamp(1e-7, 1 - 1e-7), auxiliary, reduction="none"
            )
            aux_weight = aux_mask.to(weight.dtype) * weight[:, None]
            auxiliary_loss = (aux_element * aux_weight).sum() / aux_weight.sum().clamp_min(1.0)
            loss = classification_loss + config.brier_weight * brier_loss + config.auxiliary_weight * auxiliary_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_weight = float(weight.sum().detach().cpu())
            total += batch_weight
            sums += batch_weight * np.array(
                [float(loss.detach()), float(classification_loss.detach()), float(brier_loss.detach()), float(auxiliary_loss.detach())]
            )
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "loss": float(sums[0] / total),
            "classification_loss": float(sums[1] / total),
            "brier": float(sums[2] / total),
            "auxiliary_loss": float(sums[3] / total),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if config.verbose:
            print(
                f"r_specialist epoch={epoch} loss={record['loss']:.7f} "
                f"train_brier={record['brier']:.7f}",
                flush=True,
            )
    return history


def predict_r_specialist(
    model: RSpecialistTabM,
    categorical: np.ndarray,
    numerical: np.ndarray,
    *,
    batch_size: int = 16384,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        torch.from_numpy(np.asarray(numerical, dtype=np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    outcomes: list[np.ndarray] = []
    auxiliaries: list[np.ndarray] = []
    with torch.inference_mode():
        for cat, num in loader:
            output = model.forward_with_aux(cat.to(device), num.to(device))
            outcomes.append(output["outcome_probabilities"].cpu().numpy())
            auxiliaries.append(output["auxiliary_probabilities"].cpu().numpy())
    outcome = np.concatenate(outcomes) if outcomes else np.empty((0, len(OUTCOME_NAMES)))
    auxiliary = np.concatenate(auxiliaries) if auxiliaries else np.empty((0, len(BALL_STRIKE_NAMES)))
    result = {f"outcome_{name}": outcome[:, index] for index, name in enumerate(OUTCOME_NAMES)}
    result.update({f"aux_{name}": auxiliary[:, index] for index, name in enumerate(BALL_STRIKE_NAMES)})
    return result


def save_model_npz(model: nn.Module, path: Path) -> None:
    np.savez_compressed(path, **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})


def load_model_npz(model: nn.Module, path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    model.load_state_dict(state, strict=True)


@dataclass(frozen=True)
class LoadedRSpecialist:
    preprocessor: RateBasisAugmentedPreprocessor
    models: tuple[RSpecialistTabM, ...]
    manifest: dict[str, Any]


def load_r_specialist(directory: Path, *, device: str = "cpu") -> LoadedRSpecialist:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "r_specialist_tabm_ensemble_v1":
        raise ValueError("unexpected R specialist artifact kind")
    preprocessor = RateBasisAugmentedPreprocessor.from_state_dict(
        json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
    )
    model_config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
    models: list[RSpecialistTabM] = []
    for filename in manifest["model_files"]:
        model = RSpecialistTabM.from_config_dict(model_config)
        load_model_npz(model, directory / filename)
        models.append(model.to(device).eval())
    return LoadedRSpecialist(preprocessor, tuple(models), manifest)


def predict_r_specialist_artifact(
    artifact: LoadedRSpecialist,
    frame: pd.DataFrame,
    *,
    batch_size: int = 16384,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    arrays = artifact.preprocessor.transform(frame)
    outputs = [
        predict_r_specialist(
            model,
            arrays["categorical"],
            arrays["numerical"],
            batch_size=batch_size,
            device=device,
        )
        for model in artifact.models
    ]
    if not outputs:
        raise ValueError("R specialist artifact has no models")
    return {
        name: np.mean(np.stack([output[name] for output in outputs]), axis=0)
        for name in outputs[0]
    }


__all__ = [
    "LoadedRSpecialist",
    "RSpecialistConfig",
    "RSpecialistTabM",
    "RSpecialistTrainConfig",
    "initialize_priors",
    "load_r_specialist",
    "predict_r_specialist",
    "predict_r_specialist_artifact",
    "save_model_npz",
    "train_r_specialist_fixed_epochs",
]
