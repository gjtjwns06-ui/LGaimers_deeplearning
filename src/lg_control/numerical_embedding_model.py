"""Compact tabular network with learned nonlinear numerical embeddings.

This candidate is deliberately different from the residual MLP and DCNv2
models in this project.  Each standardized numerical value is first expanded
into a target-free basis consisting of clipped raw values, fixed piecewise
linear ramps, and learned-frequency sine/cosine terms.  A feature-local learned
projection compresses that basis before a small residual tower sees it.

The fixed knots are architecture constants, not fitted quantiles.  Therefore
the module cannot consume validation or target statistics.  All normalization
and computation is along a row's feature axis; inference is independent of
batch composition and is compatible with ``training.predict_arrays`` because
``forward`` returns one logit per row.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import TabularFeatureLayout


_RESERVED_CATEGORY_COUNT = 2


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return result


@dataclass(frozen=True)
class PeriodicPiecewiseModelConfig:
    """Capacity and target-free basis choices for the numerical model."""

    embedding_dim_cap: int = 12
    numerical_embedding_dim: int = 3
    periodic_frequencies: int = 2
    piecewise_knots: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    raw_value_clip: float = 6.0
    hidden_dim: int = 96
    ffn_dim: int = 128
    num_blocks: int = 2
    dropout: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "embedding_dim_cap",
            "numerical_embedding_dim",
            "periodic_frequencies",
            "hidden_dim",
            "ffn_dim",
            "num_blocks",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.embedding_dim_cap < 2:
            raise ValueError("embedding_dim_cap must be at least 2")
        object.__setattr__(self, "dropout", _probability(self.dropout, "dropout"))

        if isinstance(self.piecewise_knots, (str, bytes)):
            raise TypeError("piecewise_knots must be a sequence of real numbers")
        try:
            knots = tuple(float(value) for value in self.piecewise_knots)
        except (TypeError, ValueError) as error:
            raise TypeError("piecewise_knots must be a sequence of real numbers") from error
        if len(knots) < 3 or not all(math.isfinite(value) for value in knots):
            raise ValueError("piecewise_knots needs at least three finite values")
        if any(right <= left for left, right in zip(knots, knots[1:])):
            raise ValueError("piecewise_knots must be strictly increasing")
        object.__setattr__(self, "piecewise_knots", knots)

        if isinstance(self.raw_value_clip, bool):
            raise TypeError("raw_value_clip must be a real number, not bool")
        try:
            clip = float(self.raw_value_clip)
        except (TypeError, ValueError) as error:
            raise TypeError("raw_value_clip must be a real number") from error
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("raw_value_clip must be finite and positive")
        object.__setattr__(self, "raw_value_clip", clip)

    def to_dict(self) -> dict[str, object]:
        return {
            "embedding_dim_cap": self.embedding_dim_cap,
            "numerical_embedding_dim": self.numerical_embedding_dim,
            "periodic_frequencies": self.periodic_frequencies,
            "piecewise_knots": list(self.piecewise_knots),
            "raw_value_clip": self.raw_value_clip,
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeriodicPiecewiseModelConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("model config payload must be a mapping")
        expected = {
            "embedding_dim_cap",
            "numerical_embedding_dim",
            "periodic_frequencies",
            "piecewise_knots",
            "raw_value_clip",
            "hidden_dim",
            "ffn_dim",
            "num_blocks",
            "dropout",
        }
        if set(payload) != expected:
            raise ValueError("model config keys differ from the strict contract")
        values = dict(payload)
        values["piecewise_knots"] = tuple(values["piecewise_knots"])
        return cls(**values)


class _FeatureLocalNumericalEmbedding(nn.Module):
    """Per-feature spline/Fourier basis followed by local projections."""

    def __init__(self, num_features: int, config: PeriodicPiecewiseModelConfig) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_frequencies = config.periodic_frequencies
        self.embedding_dim = config.numerical_embedding_dim
        self.raw_value_clip = config.raw_value_clip
        knots = torch.tensor(config.piecewise_knots, dtype=torch.float32)
        self.register_buffer("knots", knots, persistent=True)
        self.register_buffer("interval_widths", knots[1:] - knots[:-1], persistent=True)

        self.basis_dim = 1 + (len(config.piecewise_knots) - 1) + 2 * self.num_frequencies
        initial_frequency = torch.logspace(
            math.log10(0.5),
            math.log10(2.0),
            self.num_frequencies,
            dtype=torch.float32,
        )
        # Frequencies stay positive through softplus but remain fully learned.
        inverse_softplus = torch.log(torch.expm1(initial_frequency))
        self.register_buffer(
            "_initial_raw_frequencies",
            inverse_softplus.repeat(num_features, 1),
            persistent=False,
        )
        self.raw_frequencies = nn.Parameter(
            inverse_softplus.repeat(num_features, 1)
        )
        self.phases = nn.Parameter(torch.zeros(num_features, self.num_frequencies))
        self.projection_weight = nn.Parameter(
            torch.empty(num_features, self.basis_dim, self.embedding_dim)
        )
        self.projection_bias = nn.Parameter(torch.zeros(num_features, self.embedding_dim))
        self.feature_scale = nn.Parameter(torch.ones(num_features, self.embedding_dim))

    def reset_parameters(self) -> None:
        # Each numerical feature owns a basis->embedding map.  Initializing the
        # whole rank-3 tensor at once would incorrectly make fan-out depend on
        # the number of unrelated features and shrink the basis signal.
        for feature_weight in self.projection_weight:
            nn.init.xavier_uniform_(feature_weight)
        with torch.no_grad():
            self.raw_frequencies.copy_(self._initial_raw_frequencies)
            self.phases.zero_()
            self.projection_bias.zero_()
            self.feature_scale.fill_(1.0)

    def basis(self, numerical: Tensor) -> Tensor:
        clipped = numerical.clamp(-self.raw_value_clip, self.raw_value_clip)
        piecewise = (
            (clipped.unsqueeze(2) - self.knots[:-1]) / self.interval_widths
        ).clamp(0.0, 1.0)
        frequencies = F.softplus(self.raw_frequencies).unsqueeze(0)
        angles = clipped.unsqueeze(2) * frequencies + self.phases.unsqueeze(0)
        return torch.cat(
            [clipped.unsqueeze(2), piecewise, torch.sin(angles), torch.cos(angles)],
            dim=2,
        )

    def forward(self, numerical: Tensor) -> Tensor:
        basis = self.basis(numerical)
        projected = torch.einsum("bnf,nfd->bnd", basis, self.projection_weight)
        projected = projected + self.projection_bias.unsqueeze(0)
        return F.gelu(projected) * self.feature_scale.unsqueeze(0)


class _CompactResidualBlock(nn.Module):
    def __init__(self, width: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.value_gate = nn.Linear(width, ffn_dim * 2)
        self.output = nn.Linear(ffn_dim, width)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), 0.05))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_uniform_(self.value_gate.weight)
        nn.init.zeros_(self.value_gate.bias)
        nn.init.xavier_uniform_(self.output.weight, gain=0.5)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.layer_scale.fill_(0.05)

    def forward(self, inputs: Tensor) -> Tensor:
        value, gate = self.value_gate(self.norm(inputs)).chunk(2, dim=1)
        update = F.gelu(value) * torch.sigmoid(gate)
        update = self.output(self.dropout(update))
        return inputs + self.layer_scale * self.dropout(update)


class PeriodicPiecewiseResidualModel(nn.Module):
    """Categorical embeddings + learned local numerical basis + residual MLP."""

    architecture_name = "periodic_piecewise_residual_v1"

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: PeriodicPiecewiseModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_layout, TabularFeatureLayout):
            raise TypeError("feature_layout must be a TabularFeatureLayout")
        if config is None:
            config = PeriodicPiecewiseModelConfig()
        if not isinstance(config, PeriodicPiecewiseModelConfig):
            raise TypeError("config must be a PeriodicPiecewiseModelConfig")
        self.feature_layout = feature_layout
        self.config = config

        self.embedding_dims = tuple(
            min(config.embedding_dim_cap, max(4, int(round(1.6 * card**0.56))))
            for card in feature_layout.categorical_cardinalities
        )
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension)
            for cardinality, dimension in zip(
                feature_layout.categorical_cardinalities,
                self.embedding_dims,
                strict=True,
            )
        )
        self.register_buffer(
            "_cardinality_limits",
            torch.tensor(feature_layout.categorical_cardinalities, dtype=torch.long),
            persistent=True,
        )
        num_numerical = len(feature_layout.numerical_names)
        self.numerical_embedding = _FeatureLocalNumericalEmbedding(num_numerical, config)

        input_width = sum(self.embedding_dims) + num_numerical * config.numerical_embedding_dim
        self.input_norm = nn.LayerNorm(input_width)
        self.input_projection = nn.Linear(input_width, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            _CompactResidualBlock(config.hidden_dim, config.ffn_dim, config.dropout)
            for _ in range(config.num_blocks)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.deep_head = nn.Linear(config.hidden_dim, 1, bias=False)
        self.numeric_wide_head = nn.Linear(num_numerical, 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[:_RESERVED_CATEGORY_COUNT].zero_()
        self.numerical_embedding.reset_parameters()
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)
        nn.init.normal_(self.deep_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.numeric_wide_head.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.output_bias.zero_()

    def _validate_inputs(self, categorical: Tensor, numerical: Tensor) -> None:
        if not isinstance(categorical, Tensor) or not isinstance(numerical, Tensor):
            raise TypeError("categorical and numerical inputs must be torch tensors")
        expected_cat = len(self.feature_layout.categorical_names)
        expected_num = len(self.feature_layout.numerical_names)
        if categorical.ndim != 2 or categorical.shape[1] != expected_cat:
            raise ValueError(f"categorical must have shape [batch, {expected_cat}]")
        if numerical.ndim != 2 or numerical.shape[1] != expected_num:
            raise ValueError(f"numerical must have shape [batch, {expected_num}]")
        if categorical.shape[0] == 0:
            raise ValueError("input batch must contain at least one row")
        if categorical.shape[0] != numerical.shape[0]:
            raise ValueError("categorical and numerical batch sizes must match")
        if categorical.dtype != torch.long:
            raise TypeError("categorical must have dtype torch.long")
        if numerical.dtype != torch.float32:
            raise TypeError("numerical must have dtype torch.float32")
        if categorical.device != numerical.device:
            raise ValueError("categorical and numerical must be on the same device")
        if categorical.device != self._cardinality_limits.device:
            raise ValueError("model and inputs must be on the same device")
        if torch.any(categorical < 0).item() or torch.any(
            categorical >= self._cardinality_limits.unsqueeze(0)
        ).item():
            raise ValueError("categorical input contains an index outside its vocabulary")
        if not torch.isfinite(numerical).all().item():
            raise ValueError("numerical input contains NaN or infinity")

    def _raw_features(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        parts: list[Tensor] = []
        for index, embedding in enumerate(self.embeddings):
            values = categorical[:, index]
            represented = embedding(values)
            known = values.ge(_RESERVED_CATEGORY_COUNT).unsqueeze(1)
            parts.append(represented * known.to(represented.dtype))
        embedded_numerical = self.numerical_embedding(numerical)
        parts.append(embedded_numerical.flatten(start_dim=1))
        return torch.cat(parts, dim=1)

    def forward_with_details(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_inputs(categorical, numerical)
        hidden = self.input_projection(self.input_norm(self._raw_features(categorical, numerical)))
        hidden = self.input_dropout(F.gelu(hidden))
        for block in self.blocks:
            hidden = block(hidden)
        deep_logit = self.deep_head(self.output_norm(hidden)).squeeze(1)
        numeric_wide_logit = self.numeric_wide_head(numerical).squeeze(1)
        logits = self.output_bias + deep_logit + numeric_wide_logit
        return {
            "logits": logits,
            "deep_logit": deep_logit,
            "numeric_wide_logit": numeric_wide_logit,
        }

    def forward(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        return self.forward_with_details(categorical, numerical)["logits"]

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> "PeriodicPiecewiseResidualModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        expected = {"architecture", "feature_layout", "model"}
        if set(payload) != expected:
            raise ValueError("checkpoint config keys differ from the strict contract")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            PeriodicPiecewiseModelConfig.from_dict(payload["model"]),
        )

    def parameter_report(self) -> dict[str, object]:
        groups: dict[str, int] = {}
        total = 0
        trainable = 0
        for name, parameter in self.named_parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
            group = name.split(".", 1)[0]
            groups[group] = groups.get(group, 0) + count
        return {
            "architecture": self.architecture_name,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "fp32_size_mib": total * 4 / (1024**2),
            "by_top_level_module": dict(sorted(groups.items())),
        }

    def compute_report(self, *, batch_size: int = 8192) -> dict[str, int | float]:
        """Return a conservative dense multiply-add/activation estimate."""

        batch_size = _positive_int(batch_size, "batch_size")
        num_numerical = len(self.feature_layout.numerical_names)
        embedded_width = (
            sum(self.embedding_dims)
            + num_numerical * self.config.numerical_embedding_dim
        )
        basis_dim = self.numerical_embedding.basis_dim
        macs = num_numerical * basis_dim * self.config.numerical_embedding_dim
        macs += embedded_width * self.config.hidden_dim
        macs += self.config.num_blocks * 3 * self.config.hidden_dim * self.config.ffn_dim
        macs += self.config.hidden_dim + num_numerical
        # Dominant simultaneously materialized tensors; excludes allocator overhead.
        activation_values = batch_size * (
            num_numerical * basis_dim
            + num_numerical * self.config.numerical_embedding_dim
            + embedded_width
            + self.config.hidden_dim
            + 2 * self.config.ffn_dim
        )
        return {
            "estimated_multiply_adds_per_row": int(macs),
            "estimated_fp32_activation_mib": activation_values * 4 / (1024**2),
            "batch_size": batch_size,
            "numerical_basis_dim": basis_dim,
            "embedded_input_width": embedded_width,
        }


def build_periodic_piecewise_model(
    preprocessor_or_layout: Any,
    *,
    config: PeriodicPiecewiseModelConfig | None = None,
    initialization_seed: int = 42,
) -> PeriodicPiecewiseResidualModel:
    """Build deterministically on CPU without changing the ambient RNG state."""

    if isinstance(initialization_seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    try:
        seed = operator.index(initialization_seed)
    except TypeError as error:
        raise TypeError("initialization_seed must be an integer") from error
    if seed < 0:
        raise ValueError("initialization_seed must be non-negative")
    if isinstance(preprocessor_or_layout, TabularFeatureLayout):
        layout = preprocessor_or_layout
    else:
        layout = TabularFeatureLayout.from_preprocessor(preprocessor_or_layout)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PeriodicPiecewiseResidualModel(layout, config)
    model.initialization_seed = seed
    return model


__all__ = [
    "PeriodicPiecewiseModelConfig",
    "PeriodicPiecewiseResidualModel",
    "build_periodic_piecewise_model",
]
