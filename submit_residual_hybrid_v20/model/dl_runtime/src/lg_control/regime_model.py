"""Structurally distinct DCNv2-style candidate for tabular regime shifts.

The existing preprocessor already produces dense, row-aligned categorical
indices and standardized numerical features.  This module consumes that exact
contract, embeds the categorical columns, and learns explicit bounded-rank
feature crosses through a row-wise mixture of experts.  A parallel deep tower
is combined with the cross tower by a learned per-row gate.

Every operation is row independent (LayerNorm is used instead of BatchNorm),
so evaluation is invariant to batch composition and row order.  The model is
deliberately compact enough for deterministic CPU experiments.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import TabularFeatureLayout


_RESERVED_CATEGORY_COUNT = 2  # missing=0, OOV=1 in DeepTabularPreprocessor


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


def _dropout_probability(value: Any, name: str) -> float:
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
class GatedCrossModelConfig:
    """Capacity controls for the DCNv2-style gated-expert candidate."""

    embedding_dim_cap: int = 16
    interaction_dim: int = 96
    cross_layers: int = 3
    num_experts: int = 4
    low_rank: int = 24
    deep_hidden_dim: int = 128
    deep_layers: int = 2
    dropout: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "embedding_dim_cap",
            "interaction_dim",
            "cross_layers",
            "num_experts",
            "low_rank",
            "deep_hidden_dim",
            "deep_layers",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "dropout",
            _dropout_probability(self.dropout, "dropout"),
        )
        if self.embedding_dim_cap < 2:
            raise ValueError("embedding_dim_cap must be at least 2")
        if self.low_rank > self.interaction_dim:
            raise ValueError("low_rank must not exceed interaction_dim")

    def to_dict(self) -> dict[str, object]:
        return {
            "embedding_dim_cap": self.embedding_dim_cap,
            "interaction_dim": self.interaction_dim,
            "cross_layers": self.cross_layers,
            "num_experts": self.num_experts,
            "low_rank": self.low_rank,
            "deep_hidden_dim": self.deep_hidden_dim,
            "deep_layers": self.deep_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GatedCrossModelConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("model config payload must be a mapping")
        return cls(**dict(payload))


AUXILIARY_TARGET_NAMES = ("reverse", "middle", "ball", "strike")


@dataclass(frozen=True)
class MultiTaskHeadConfig:
    """Configuration for the four fixed auxiliary binary-logit heads."""

    hidden_dim: int = 64
    dropout: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hidden_dim",
            _positive_int(self.hidden_dim, "hidden_dim"),
        )
        object.__setattr__(
            self,
            "dropout",
            _dropout_probability(self.dropout, "dropout"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "auxiliary_targets": list(AUXILIARY_TARGET_NAMES),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiTaskHeadConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("multi-task head config payload must be a mapping")
        expected_keys = {"auxiliary_targets", "hidden_dim", "dropout"}
        if set(payload) != expected_keys:
            raise ValueError("multi-task head config keys differ from the contract")
        if payload.get("auxiliary_targets") != list(AUXILIARY_TARGET_NAMES):
            raise ValueError("auxiliary target order differs from the fixed contract")
        return cls(
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
        )


@dataclass(frozen=True)
class StructuredResidualConfig:
    """Bound for feeding structured auxiliary evidence back to success."""

    correction_cap: float = 0.50

    def __post_init__(self) -> None:
        if isinstance(self.correction_cap, bool):
            raise TypeError("correction_cap must be a real number, not bool")
        try:
            value = float(self.correction_cap)
        except (TypeError, ValueError) as error:
            raise TypeError("correction_cap must be a real number") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("correction_cap must be finite and positive")
        object.__setattr__(self, "correction_cap", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "correction_cap": self.correction_cap,
            "exclusive_outcomes": ["control_success", "reverse", "middle"],
            "bounded_channels": [
                "exclusive_failure",
                "ball_centered",
                "strike_centered",
            ],
            "initialization": "sigmoid_gate_minus_one_half",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredResidualConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("structured residual config payload must be a mapping")
        expected = {
            "correction_cap": None,
            "exclusive_outcomes": ["control_success", "reverse", "middle"],
            "bounded_channels": [
                "exclusive_failure",
                "ball_centered",
                "strike_centered",
            ],
            "initialization": "sigmoid_gate_minus_one_half",
        }
        if set(payload) != set(expected):
            raise ValueError("structured residual config keys differ from the contract")
        for name in ("exclusive_outcomes", "bounded_channels", "initialization"):
            if payload.get(name) != expected[name]:
                raise ValueError(f"structured residual contract differs: {name}")
        return cls(correction_cap=payload["correction_cap"])


class _LowRankCrossExpert(nn.Module):
    """One U-C-V low-rank feature-cross expert."""

    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.core = nn.Linear(rank, rank, bias=True)
        self.up = nn.Linear(rank, width, bias=False)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.down(inputs))
        hidden = self.activation(self.core(hidden))
        return self.up(hidden)


class _GatedCrossLayer(nn.Module):
    """DCNv2-style residual cross with a soft mixture of low-rank experts."""

    def __init__(
        self,
        width: int,
        *,
        rank: int,
        num_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.expert_gate = nn.Linear(width, num_experts)
        self.experts = nn.ModuleList(
            _LowRankCrossExpert(width, rank) for _ in range(num_experts)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), 0.10))

    def forward(self, base: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm(state)
        weights = torch.softmax(self.expert_gate(normalized), dim=1)
        expert_crosses = torch.stack(
            tuple(base * expert(normalized) for expert in self.experts),
            dim=1,
        )
        update = torch.sum(weights.unsqueeze(2) * expert_crosses, dim=1)
        return state + self.layer_scale * self.dropout(update), weights


class _DeepResidualBlock(nn.Module):
    def __init__(self, width: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, hidden_dim)
        self.activation = nn.GELU()
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, width)
        self.dropout_out = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((width,), 0.10))

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.linear_in(self.norm(inputs))
        hidden = self.dropout_in(self.activation(hidden))
        hidden = self.dropout_out(self.linear_out(hidden))
        return inputs + self.layer_scale * hidden


class GatedCrossRegimeModel(nn.Module):
    """Categorical embeddings + DCNv2 crosses + gated parallel deep tower."""

    architecture_name = "dcnv2_gated_experts_v1"

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: GatedCrossModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_layout, TabularFeatureLayout):
            raise TypeError("feature_layout must be a TabularFeatureLayout")
        if config is None:
            config = GatedCrossModelConfig()
        if not isinstance(config, GatedCrossModelConfig):
            raise TypeError("config must be a GatedCrossModelConfig")
        self.feature_layout = feature_layout
        self.config = config

        self.embedding_dims = tuple(
            min(
                config.embedding_dim_cap,
                max(4, int(round(1.6 * cardinality**0.56))),
            )
            for cardinality in feature_layout.categorical_cardinalities
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

        raw_width = sum(self.embedding_dims) + len(feature_layout.numerical_names)
        self.input_norm = nn.LayerNorm(raw_width)
        self.input_projection = nn.Linear(raw_width, config.interaction_dim)
        self.input_activation = nn.GELU()
        self.input_dropout = nn.Dropout(config.dropout)

        self.cross_network = nn.ModuleList(
            _GatedCrossLayer(
                config.interaction_dim,
                rank=config.low_rank,
                num_experts=config.num_experts,
                dropout=config.dropout,
            )
            for _ in range(config.cross_layers)
        )
        self.deep_network = nn.ModuleList(
            _DeepResidualBlock(
                config.interaction_dim,
                config.deep_hidden_dim,
                config.dropout,
            )
            for _ in range(config.deep_layers)
        )
        self.cross_output_norm = nn.LayerNorm(config.interaction_dim)
        self.deep_output_norm = nn.LayerNorm(config.interaction_dim)
        self.cross_head = nn.Linear(config.interaction_dim, 1, bias=False)
        self.deep_head = nn.Linear(config.interaction_dim, 1, bias=False)
        self.fusion_gate = nn.Linear(config.interaction_dim * 2, 1)
        self.numeric_wide_head = nn.Linear(
            len(feature_layout.numerical_names), 1, bias=False
        )
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[:_RESERVED_CATEGORY_COUNT].zero_()
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        for layer in self.cross_network:
            nn.init.ones_(layer.norm.weight)
            nn.init.zeros_(layer.norm.bias)
            nn.init.xavier_uniform_(layer.expert_gate.weight)
            nn.init.zeros_(layer.expert_gate.bias)
            with torch.no_grad():
                layer.layer_scale.fill_(0.10)
            for expert in layer.experts:
                nn.init.xavier_uniform_(expert.down.weight)
                nn.init.xavier_uniform_(expert.core.weight)
                nn.init.zeros_(expert.core.bias)
                nn.init.xavier_uniform_(expert.up.weight, gain=0.5)
        for block in self.deep_network:
            nn.init.ones_(block.norm.weight)
            nn.init.zeros_(block.norm.bias)
            nn.init.xavier_uniform_(block.linear_in.weight)
            nn.init.zeros_(block.linear_in.bias)
            nn.init.xavier_uniform_(block.linear_out.weight, gain=0.5)
            nn.init.zeros_(block.linear_out.bias)
            with torch.no_grad():
                block.layer_scale.fill_(0.10)

        for norm in (self.cross_output_norm, self.deep_output_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.normal_(self.cross_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.deep_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        nn.init.normal_(self.numeric_wide_head.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.output_bias.zero_()

    def _validate_inputs(self, categorical: Tensor, numerical: Tensor) -> None:
        if not isinstance(categorical, Tensor) or not isinstance(numerical, Tensor):
            raise TypeError("categorical and numerical inputs must be torch tensors")
        expected_categorical = len(self.feature_layout.categorical_names)
        expected_numerical = len(self.feature_layout.numerical_names)
        if categorical.ndim != 2 or categorical.shape[1] != expected_categorical:
            raise ValueError(
                f"categorical must have shape [batch, {expected_categorical}]"
            )
        if numerical.ndim != 2 or numerical.shape[1] != expected_numerical:
            raise ValueError(f"numerical must have shape [batch, {expected_numerical}]")
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
        for position, embedding in enumerate(self.embeddings):
            values = categorical[:, position]
            representation = embedding(values)
            known = values.ge(_RESERVED_CATEGORY_COUNT).unsqueeze(1)
            parts.append(representation * known.to(representation.dtype))
        parts.append(numerical)
        return torch.cat(parts, dim=1)

    def _shared_states(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        """Encode one batch once for the main and optional auxiliary heads."""

        self._validate_inputs(categorical, numerical)
        raw = self.input_norm(self._raw_features(categorical, numerical))
        base = self.input_dropout(self.input_activation(self.input_projection(raw)))

        cross_state = base
        expert_weights: list[Tensor] = []
        for layer in self.cross_network:
            cross_state, weights = layer(base, cross_state)
            expert_weights.append(weights)

        deep_state = base
        for block in self.deep_network:
            deep_state = block(deep_state)

        cross_state = self.cross_output_norm(cross_state)
        deep_state = self.deep_output_norm(deep_state)
        fusion_weight = torch.sigmoid(
            self.fusion_gate(torch.cat([cross_state, deep_state], dim=1))
        ).squeeze(1)
        fused_state = (
            fusion_weight.unsqueeze(1) * cross_state
            + (1.0 - fusion_weight).unsqueeze(1) * deep_state
        )
        return {
            "cross_state": cross_state,
            "deep_state": deep_state,
            "fused_state": fused_state,
            "expert_weights": torch.stack(expert_weights, dim=1),
            "fusion_weight": fusion_weight,
        }

    def _main_outputs(
        self,
        shared: Mapping[str, Tensor],
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        cross_state = shared["cross_state"]
        deep_state = shared["deep_state"]
        fusion_weight = shared["fusion_weight"]
        cross_logit = self.cross_head(cross_state).squeeze(1)
        deep_logit = self.deep_head(deep_state).squeeze(1)
        numeric_wide_logit = self.numeric_wide_head(numerical).squeeze(1)
        logits = (
            self.output_bias
            + numeric_wide_logit
            + fusion_weight * cross_logit
            + (1.0 - fusion_weight) * deep_logit
        )
        return {
            "logits": logits,
            "expert_weights": shared["expert_weights"],
            "fusion_weight": fusion_weight,
            "cross_logit": cross_logit,
            "deep_logit": deep_logit,
            "numeric_wide_logit": numeric_wide_logit,
        }

    def forward_with_aux(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        shared = self._shared_states(categorical, numerical)
        return self._main_outputs(shared, numerical)

    def forward(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        return self.forward_with_aux(categorical, numerical)["logits"]

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> "GatedCrossRegimeModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            GatedCrossModelConfig.from_dict(payload["model"]),
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
            prefix = name.split(".", 1)[0]
            groups[prefix] = groups.get(prefix, 0) + count
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "fp32_size_mib": total * 4 / (1024**2),
            "by_top_level_module": dict(sorted(groups.items())),
        }


class MultiTaskGatedCrossRegimeModel(GatedCrossRegimeModel):
    """Main success logit plus four auxiliary logits from the shared state.

    :meth:`forward` is intentionally inherited unchanged and therefore returns
    only the main success logit.  Training code that opts into multi-task loss
    can read ``auxiliary_logits`` from :meth:`forward_with_aux`.
    """

    architecture_name = "dcnv2_gated_experts_multitask_v1"
    main_target_name = "control_success"
    auxiliary_target_names = AUXILIARY_TARGET_NAMES

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: GatedCrossModelConfig | None = None,
        multitask_config: MultiTaskHeadConfig | None = None,
    ) -> None:
        if multitask_config is None:
            multitask_config = MultiTaskHeadConfig()
        if not isinstance(multitask_config, MultiTaskHeadConfig):
            raise TypeError("multitask_config must be a MultiTaskHeadConfig")
        self.multitask_config = multitask_config
        super().__init__(feature_layout, config)
        width = self.config.interaction_dim
        self.auxiliary_norm = nn.LayerNorm(width)
        self.auxiliary_hidden = nn.Linear(width, multitask_config.hidden_dim)
        self.auxiliary_activation = nn.GELU()
        self.auxiliary_dropout = nn.Dropout(multitask_config.dropout)
        self.auxiliary_head = nn.Linear(
            multitask_config.hidden_dim,
            len(AUXILIARY_TARGET_NAMES),
        )
        self._reset_auxiliary_parameters()

    def _reset_auxiliary_parameters(self) -> None:
        nn.init.ones_(self.auxiliary_norm.weight)
        nn.init.zeros_(self.auxiliary_norm.bias)
        nn.init.xavier_uniform_(self.auxiliary_hidden.weight)
        nn.init.zeros_(self.auxiliary_hidden.bias)
        nn.init.normal_(self.auxiliary_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.auxiliary_head.bias)

    def reset_parameters(self) -> None:
        # GatedCrossRegimeModel.__init__ dispatches here before the auxiliary
        # modules exist; the guard preserves identical backbone initialization.
        super().reset_parameters()
        if hasattr(self, "auxiliary_head"):
            self._reset_auxiliary_parameters()

    def _auxiliary_logits(self, shared: Mapping[str, Tensor]) -> Tensor:
        auxiliary_hidden = self.auxiliary_hidden(
            self.auxiliary_norm(shared["fused_state"])
        )
        auxiliary_hidden = self.auxiliary_dropout(
            self.auxiliary_activation(auxiliary_hidden)
        )
        return self.auxiliary_head(auxiliary_hidden)

    def forward_with_aux(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        shared = self._shared_states(categorical, numerical)
        outputs = self._main_outputs(shared, numerical)
        outputs["auxiliary_logits"] = self._auxiliary_logits(shared)
        return outputs

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
            "multitask": self.multitask_config.to_dict(),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, Any],
    ) -> "MultiTaskGatedCrossRegimeModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            GatedCrossModelConfig.from_dict(payload["model"]),
            MultiTaskHeadConfig.from_dict(payload["multitask"]),
        )


class StructuredMultiTaskGatedCrossRegimeModel(MultiTaskGatedCrossRegimeModel):
    """Feed structured auxiliary probabilities back as a bounded correction.

    Success, reverse, and middle are normalized as mutually exclusive outcomes
    before reverse/middle are combined into one failure-mass channel.  Ball and
    strike use centered sigmoid probabilities.  Three row-wise gates are
    represented as ``sigmoid(logit) - 0.5`` and their logits start at exactly
    zero, so the initial structured correction is bit-exactly zero while the
    gate remains differentiable on the first optimizer step.
    """

    architecture_name = "dcnv2_gated_experts_structured_multitask_v1"

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: GatedCrossModelConfig | None = None,
        multitask_config: MultiTaskHeadConfig | None = None,
        structured_config: StructuredResidualConfig | None = None,
    ) -> None:
        if structured_config is None:
            structured_config = StructuredResidualConfig()
        if not isinstance(structured_config, StructuredResidualConfig):
            raise TypeError("structured_config must be a StructuredResidualConfig")
        self.structured_config = structured_config
        super().__init__(feature_layout, config, multitask_config)
        self.structured_gate = nn.Linear(self.config.interaction_dim + 3, 3)
        self._reset_structured_parameters()

    def _reset_structured_parameters(self) -> None:
        nn.init.zeros_(self.structured_gate.weight)
        nn.init.zeros_(self.structured_gate.bias)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "structured_gate"):
            self._reset_structured_parameters()

    def forward_with_aux(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        shared = self._shared_states(categorical, numerical)
        outputs = self._main_outputs(shared, numerical)
        base_logits = outputs["logits"]
        auxiliary_logits = self._auxiliary_logits(shared)

        # A three-way softmax makes the success/reverse/middle relationship an
        # explicit exclusive outcome simplex instead of four unrelated sigmoids.
        exclusive_probabilities = torch.softmax(
            torch.stack(
                [base_logits, auxiliary_logits[:, 0], auxiliary_logits[:, 1]],
                dim=1,
            ),
            dim=1,
        )
        failure_mass = exclusive_probabilities[:, 1:].sum(dim=1)
        ball_probability = torch.sigmoid(auxiliary_logits[:, 2])
        strike_probability = torch.sigmoid(auxiliary_logits[:, 3])
        structured_channels = torch.stack(
            [
                -failure_mass,
                ball_probability - 0.5,
                strike_probability - 0.5,
            ],
            dim=1,
        )
        gate_input = torch.cat(
            [
                shared["fused_state"],
                failure_mass.unsqueeze(1),
                ball_probability.unsqueeze(1),
                strike_probability.unsqueeze(1),
            ],
            dim=1,
        )
        gate_delta = torch.sigmoid(self.structured_gate(gate_input)) - 0.5
        raw_correction = torch.sum(gate_delta * structured_channels, dim=1)
        structured_correction = self.structured_config.correction_cap * torch.tanh(
            raw_correction
        )

        outputs["base_logits"] = base_logits
        outputs["auxiliary_logits"] = auxiliary_logits
        outputs["structured_correction"] = structured_correction
        outputs["logits"] = base_logits + structured_correction
        return outputs

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
            "multitask": self.multitask_config.to_dict(),
            "structured": self.structured_config.to_dict(),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, Any],
    ) -> "StructuredMultiTaskGatedCrossRegimeModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            GatedCrossModelConfig.from_dict(payload["model"]),
            MultiTaskHeadConfig.from_dict(payload["multitask"]),
            StructuredResidualConfig.from_dict(payload["structured"]),
        )


def build_regime_model(
    preprocessor: Any,
    *,
    config: GatedCrossModelConfig | None = None,
    initialization_seed: int = 42,
) -> GatedCrossRegimeModel:
    """Build on CPU without changing the caller's ambient PyTorch RNG state."""

    if isinstance(initialization_seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    try:
        seed = operator.index(initialization_seed)
    except TypeError as error:
        raise TypeError("initialization_seed must be an integer") from error
    if seed < 0:
        raise ValueError("initialization_seed must be non-negative")
    layout = TabularFeatureLayout.from_preprocessor(preprocessor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GatedCrossRegimeModel(layout, config)
    model.initialization_seed = seed
    return model


def build_multitask_regime_model(
    preprocessor: Any,
    *,
    config: GatedCrossModelConfig | None = None,
    multitask_config: MultiTaskHeadConfig | None = None,
    initialization_seed: int = 42,
) -> MultiTaskGatedCrossRegimeModel:
    """Build the compatible five-output training variant deterministically."""

    if isinstance(initialization_seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    try:
        seed = operator.index(initialization_seed)
    except TypeError as error:
        raise TypeError("initialization_seed must be an integer") from error
    if seed < 0:
        raise ValueError("initialization_seed must be non-negative")
    layout = TabularFeatureLayout.from_preprocessor(preprocessor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MultiTaskGatedCrossRegimeModel(
            layout,
            config,
            multitask_config,
        )
    model.initialization_seed = seed
    return model


def build_structured_multitask_regime_model(
    preprocessor: Any,
    *,
    config: GatedCrossModelConfig | None = None,
    multitask_config: MultiTaskHeadConfig | None = None,
    structured_config: StructuredResidualConfig | None = None,
    initialization_seed: int = 42,
) -> StructuredMultiTaskGatedCrossRegimeModel:
    """Build the zero-correction structured multi-task variant on CPU."""

    if isinstance(initialization_seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    try:
        seed = operator.index(initialization_seed)
    except TypeError as error:
        raise TypeError("initialization_seed must be an integer") from error
    if seed < 0:
        raise ValueError("initialization_seed must be non-negative")
    layout = TabularFeatureLayout.from_preprocessor(preprocessor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = StructuredMultiTaskGatedCrossRegimeModel(
            layout,
            config,
            multitask_config,
            structured_config,
        )
    model.initialization_seed = seed
    return model


__all__ = [
    "AUXILIARY_TARGET_NAMES",
    "GatedCrossModelConfig",
    "GatedCrossRegimeModel",
    "MultiTaskHeadConfig",
    "MultiTaskGatedCrossRegimeModel",
    "StructuredResidualConfig",
    "StructuredMultiTaskGatedCrossRegimeModel",
    "build_regime_model",
    "build_multitask_regime_model",
    "build_structured_multitask_regime_model",
]
