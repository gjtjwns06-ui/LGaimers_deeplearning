"""Deterministically routed F/R residual heads for the gated-cross model.

The backbone is the existing :class:`GatedCrossRegimeModel`.  Two compact,
independent heads consume its shared fused representation.  Routing is based
only on the encoded ``game_type`` category and the exact fitted vocabulary
index for ``F``; no batch statistic or heuristic participates in the route.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import TabularFeatureLayout
from .regime_model import GatedCrossModelConfig, GatedCrossRegimeModel


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
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


def _feature_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("targeted_numerical_names must be a sequence of names")
    names = tuple(str(value).strip() for value in values)
    if any(not name for name in names):
        raise ValueError("targeted numerical feature names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("targeted numerical feature names must be unique")
    return names


@dataclass(frozen=True)
class RegimeSplitHeadConfig:
    """Serializable routing and compact-head configuration."""

    game_type_categorical_index: int
    f_vocabulary_index: int
    hidden_dim: int = 32
    dropout: float = 0.05
    correction_cap: float = 0.35
    targeted_numerical_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "game_type_categorical_index",
            _integer(
                self.game_type_categorical_index,
                "game_type_categorical_index",
                minimum=0,
            ),
        )
        # 0 and 1 are the preprocessor's missing and OOV buckets, never F.
        object.__setattr__(
            self,
            "f_vocabulary_index",
            _integer(self.f_vocabulary_index, "f_vocabulary_index", minimum=2),
        )
        object.__setattr__(
            self, "hidden_dim", _integer(self.hidden_dim, "hidden_dim", minimum=1)
        )
        object.__setattr__(self, "dropout", _probability(self.dropout, "dropout"))
        if isinstance(self.correction_cap, bool):
            raise TypeError("correction_cap must be a real number, not bool")
        try:
            cap = float(self.correction_cap)
        except (TypeError, ValueError) as error:
            raise TypeError("correction_cap must be a real number") from error
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("correction_cap must be finite and positive")
        object.__setattr__(self, "correction_cap", cap)
        object.__setattr__(
            self,
            "targeted_numerical_names",
            _feature_names(self.targeted_numerical_names),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "game_type_categorical_index": self.game_type_categorical_index,
            "f_vocabulary_index": self.f_vocabulary_index,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "correction_cap": self.correction_cap,
            "targeted_numerical_names": list(self.targeted_numerical_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeSplitHeadConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("regime split-head config must be a mapping")
        expected = {
            "game_type_categorical_index",
            "f_vocabulary_index",
            "hidden_dim",
            "dropout",
            "correction_cap",
            "targeted_numerical_names",
        }
        if set(payload) != expected:
            raise ValueError("regime split-head config keys differ from the contract")
        return cls(**dict(payload))


class _RegimeResidualHead(nn.Module):
    """Small row-local head with an exactly zero initial output."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        # The output itself is zero, while d(output)/d(output parameters) is
        # non-zero on the first backward pass.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.hidden(self.norm(inputs))
        hidden = self.dropout(self.activation(hidden))
        return self.output(hidden).squeeze(1)


class RegimeSplitGatedCrossModel(GatedCrossRegimeModel):
    """Gated-cross backbone plus independently routed F and R corrections."""

    architecture_name = "dcnv2_gated_experts_regime_split_v1"

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: GatedCrossModelConfig | None,
        split_head_config: RegimeSplitHeadConfig,
    ) -> None:
        if not isinstance(split_head_config, RegimeSplitHeadConfig):
            raise TypeError("split_head_config must be a RegimeSplitHeadConfig")
        self.split_head_config = split_head_config
        super().__init__(feature_layout, config)

        try:
            actual_game_type_position = feature_layout.categorical_index("game_type")
        except KeyError as error:
            raise ValueError("feature layout must contain categorical game_type") from error
        if actual_game_type_position != split_head_config.game_type_categorical_index:
            raise ValueError(
                "configured game_type categorical index does not match feature layout"
            )
        game_type_cardinality = feature_layout.categorical_cardinalities[
            actual_game_type_position
        ]
        if split_head_config.f_vocabulary_index >= game_type_cardinality:
            raise ValueError("configured F vocabulary index is outside game_type vocabulary")

        missing = [
            name
            for name in split_head_config.targeted_numerical_names
            if name not in feature_layout.numerical_names
        ]
        if missing:
            raise ValueError(f"unknown targeted numerical features: {missing}")
        numerical_positions = tuple(
            feature_layout.numerical_index(name)
            for name in split_head_config.targeted_numerical_names
        )
        self.register_buffer(
            "_game_type_categorical_position",
            torch.tensor(actual_game_type_position, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_f_vocabulary_index",
            torch.tensor(split_head_config.f_vocabulary_index, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_targeted_numerical_positions",
            torch.tensor(numerical_positions, dtype=torch.long),
            persistent=True,
        )
        head_input_dim = self.config.interaction_dim + len(numerical_positions)
        self.f_residual_head = _RegimeResidualHead(
            head_input_dim,
            split_head_config.hidden_dim,
            split_head_config.dropout,
        )
        self.r_residual_head = _RegimeResidualHead(
            head_input_dim,
            split_head_config.hidden_dim,
            split_head_config.dropout,
        )

    def _head_inputs(self, fused_state: Tensor, numerical: Tensor) -> Tensor:
        if self._targeted_numerical_positions.numel() == 0:
            return fused_state
        targeted = numerical.index_select(1, self._targeted_numerical_positions)
        return torch.cat([fused_state, targeted], dim=1)

    def forward_with_aux(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        shared = self._shared_states(categorical, numerical)
        outputs = self._main_outputs(shared, numerical)
        base_logits = outputs["logits"]
        head_inputs = self._head_inputs(shared["fused_state"], numerical)

        f_raw = self.f_residual_head(head_inputs)
        r_raw = self.r_residual_head(head_inputs)
        cap = self.split_head_config.correction_cap
        f_correction = cap * torch.tanh(f_raw)
        r_correction = cap * torch.tanh(r_raw)
        game_type = categorical.index_select(
            1, self._game_type_categorical_position.reshape(1)
        ).squeeze(1)
        is_f = game_type.eq(self._f_vocabulary_index)
        correction = torch.where(is_f, f_correction, r_correction)

        outputs["base_logits"] = base_logits
        outputs["f_regime_correction"] = f_correction
        outputs["r_regime_correction"] = r_correction
        outputs["regime_correction"] = correction
        outputs["is_f_regime"] = is_f
        outputs["logits"] = base_logits + correction
        return outputs

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
            "regime_split_head": self.split_head_config.to_dict(),
        }

    @classmethod
    def from_config(
        cls, payload: Mapping[str, Any]
    ) -> "RegimeSplitGatedCrossModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        expected = {"architecture", "feature_layout", "model", "regime_split_head"}
        if set(payload) != expected:
            raise ValueError("regime split model config keys differ from the contract")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            GatedCrossModelConfig.from_dict(payload["model"]),
            RegimeSplitHeadConfig.from_dict(payload["regime_split_head"]),
        )

    def parameter_report(self) -> dict[str, object]:
        report = super().parameter_report()
        report["regime_split_head_parameters"] = {
            "F": sum(parameter.numel() for parameter in self.f_residual_head.parameters()),
            "R": sum(parameter.numel() for parameter in self.r_residual_head.parameters()),
        }
        report["routing"] = {
            "categorical_name": "game_type",
            "categorical_index": self.split_head_config.game_type_categorical_index,
            "f_vocabulary_index": self.split_head_config.f_vocabulary_index,
            "targeted_numerical_names": list(
                self.split_head_config.targeted_numerical_names
            ),
        }
        return report


def _layout_from_preprocessor(preprocessor: Any) -> TabularFeatureLayout:
    feature_layout = getattr(preprocessor, "feature_layout", None)
    if callable(feature_layout):
        layout = feature_layout()
        if not isinstance(layout, TabularFeatureLayout):
            raise TypeError("preprocessor feature_layout() returned an invalid layout")
        return layout
    return TabularFeatureLayout.from_preprocessor(preprocessor)


def _find_game_type_vocabulary(preprocessor: Any) -> Mapping[str, int] | None:
    """Walk known wrapper ``base`` links to the fitted canonical vocabulary."""

    current = preprocessor
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        vocabs = getattr(current, "category_vocabs_", None)
        if isinstance(vocabs, Mapping):
            vocabulary = vocabs.get("game_type")
            if isinstance(vocabulary, Mapping):
                return vocabulary
        current = getattr(current, "base", None)
    return None


def build_regime_split_model(
    preprocessor: Any,
    *,
    config: GatedCrossModelConfig | None = None,
    hidden_dim: int = 32,
    dropout: float = 0.05,
    correction_cap: float = 0.35,
    targeted_numerical_names: Sequence[str] = (),
    f_vocabulary_index: int | None = None,
    initialization_seed: int = 42,
) -> RegimeSplitGatedCrossModel:
    """Build a deterministic F/R model from a fitted base or wrapper preprocessor.

    The builder prefers the canonical fitted ``game_type`` vocabulary.  A
    caller may provide ``f_vocabulary_index`` only when a wrapper intentionally
    hides that state; silent guesses are never made.
    """

    seed = _integer(initialization_seed, "initialization_seed", minimum=0)
    layout = _layout_from_preprocessor(preprocessor)
    try:
        game_type_position = layout.categorical_index("game_type")
    except KeyError as error:
        raise ValueError("feature layout must contain categorical game_type") from error

    vocabulary = _find_game_type_vocabulary(preprocessor)
    discovered: int | None = None
    if vocabulary is not None:
        raw_discovered = vocabulary.get("F")
        if raw_discovered is None:
            raise ValueError("fitted game_type vocabulary does not contain F")
        discovered = _integer(raw_discovered, "fitted F vocabulary index", minimum=2)
    if f_vocabulary_index is None:
        if discovered is None:
            raise ValueError(
                "cannot resolve fitted F vocabulary index; pass f_vocabulary_index explicitly"
            )
        resolved_f_index = discovered
    else:
        resolved_f_index = _integer(
            f_vocabulary_index, "f_vocabulary_index", minimum=2
        )
        if discovered is not None and resolved_f_index != discovered:
            raise ValueError("explicit F vocabulary index disagrees with fitted vocabulary")

    split_config = RegimeSplitHeadConfig(
        game_type_categorical_index=game_type_position,
        f_vocabulary_index=resolved_f_index,
        hidden_dim=hidden_dim,
        dropout=dropout,
        correction_cap=correction_cap,
        targeted_numerical_names=tuple(targeted_numerical_names),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RegimeSplitGatedCrossModel(layout, config, split_config)
    model.initialization_seed = seed
    return model


__all__ = [
    "RegimeSplitGatedCrossModel",
    "RegimeSplitHeadConfig",
    "build_regime_split_model",
]
