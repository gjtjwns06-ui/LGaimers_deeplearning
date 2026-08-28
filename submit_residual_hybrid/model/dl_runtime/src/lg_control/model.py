"""Row-independent neural network for tabular probability prediction.

The model deliberately uses only per-row operations.  In particular, it uses
LayerNorm rather than a batch-dependent normalization layer, and entity
embedding dropout samples a separate whole-embedding mask for every row.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _as_non_negative_int(value: Any, name: str) -> int:
    """Return ``value`` as an integer, rejecting floats and booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _normalise_cardinalities(
    values: Sequence[int] | None,
    name: str,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers")

    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of positive integers") from exc

    cardinalities: list[int] = []
    for position, value in enumerate(raw_values):
        cardinality = _as_non_negative_int(value, f"{name}[{position}]")
        if cardinality == 0:
            raise ValueError(f"{name}[{position}] must be positive")
        cardinalities.append(cardinality)
    return tuple(cardinalities)


class _ResidualBlock(nn.Module):
    """Pre-normalized residual MLP block with no cross-row operations."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        expanded_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, expanded_dim)
        self.activation = nn.GELU()
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(expanded_dim, hidden_dim)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        hidden = self.norm(inputs)
        hidden = self.linear_in(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout_in(hidden)
        hidden = self.linear_out(hidden)
        hidden = self.dropout_out(hidden)
        return residual + hidden


class TabularResidualModel(nn.Module):
    """Residual MLP with categorical embeddings and a numeric wide head.

    Parameters
    ----------
    cardinalities:
        Vocabulary size (including any missing/OOV bucket) of each categorical
        feature.
    num_numeric:
        Number of already-imputed and scaled numeric features.
    hidden_dim:
        Width of the deep residual branch.
    num_blocks:
        Number of pre-LayerNorm residual MLP blocks.
    dropout:
        Element-wise dropout probability in the deep branch.
    embedding_dropout:
        Probability of dropping an entire entity embedding for one row.  The
        mask is sampled independently for each row and entity feature.
    entity_cat_indices:
        Indices of high-cardinality entity features (for example pitcher and
        batter IDs) to which whole-embedding dropout is applied.

    Notes
    -----
    ``category_cardinalities`` and ``num_numerical`` are accepted as aliases
    for compatibility with the preprocessing/training API.
    """

    def __init__(
        self,
        cardinalities: Sequence[int] | None = None,
        num_numeric: int | None = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
        embedding_dropout: float = 0.10,
        entity_cat_indices: Sequence[int] = (3, 4),
        *,
        category_cardinalities: Sequence[int] | None = None,
        num_numerical: int | None = None,
    ) -> None:
        super().__init__()

        primary_cardinalities = _normalise_cardinalities(
            cardinalities, "cardinalities"
        )
        alias_cardinalities = _normalise_cardinalities(
            category_cardinalities, "category_cardinalities"
        )
        if primary_cardinalities is None and alias_cardinalities is None:
            raise TypeError(
                "missing required argument: 'cardinalities' "
                "(or 'category_cardinalities')"
            )
        if (
            primary_cardinalities is not None
            and alias_cardinalities is not None
            and primary_cardinalities != alias_cardinalities
        ):
            raise ValueError(
                "cardinalities and category_cardinalities must match when "
                "both are provided"
            )
        resolved_cardinalities = (
            primary_cardinalities
            if primary_cardinalities is not None
            else alias_cardinalities
        )
        assert resolved_cardinalities is not None

        resolved_num_numeric = self._resolve_numeric_count(
            num_numeric, num_numerical
        )
        resolved_hidden_dim = _as_non_negative_int(hidden_dim, "hidden_dim")
        if resolved_hidden_dim == 0:
            raise ValueError("hidden_dim must be positive")
        resolved_num_blocks = _as_non_negative_int(num_blocks, "num_blocks")
        if resolved_num_blocks == 0:
            raise ValueError("num_blocks must be positive")

        resolved_dropout = self._validate_probability(dropout, "dropout")
        resolved_embedding_dropout = self._validate_probability(
            embedding_dropout, "embedding_dropout"
        )
        resolved_entity_indices = self._validate_entity_indices(
            entity_cat_indices, len(resolved_cardinalities)
        )

        if not resolved_cardinalities and resolved_num_numeric == 0:
            raise ValueError(
                "the model needs at least one categorical or numeric feature"
            )

        self.cardinalities = resolved_cardinalities
        self.category_cardinalities = resolved_cardinalities
        self.num_numeric = resolved_num_numeric
        self.num_numerical = resolved_num_numeric
        self.hidden_dim = resolved_hidden_dim
        self.num_blocks = resolved_num_blocks
        self.dropout = resolved_dropout
        self.embedding_dropout = resolved_embedding_dropout
        self.entity_cat_indices = resolved_entity_indices
        self._entity_cat_index_set = frozenset(resolved_entity_indices)

        self.embedding_dims = tuple(
            self._embedding_dim(cardinality)
            for cardinality in self.cardinalities
        )
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim)
            for cardinality, embedding_dim in zip(
                self.cardinalities, self.embedding_dims
            )
        )

        deep_input_dim = sum(self.embedding_dims) + self.num_numeric
        self.input_projection = nn.Linear(deep_input_dim, self.hidden_dim)
        self.input_activation = nn.GELU()
        self.input_dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList(
            _ResidualBlock(self.hidden_dim, self.dropout)
            for _ in range(self.num_blocks)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.deep_head = nn.Linear(self.hidden_dim, 1, bias=False)
        self.numeric_wide_head: nn.Linear | None
        if self.num_numeric:
            self.numeric_wide_head = nn.Linear(
                self.num_numeric, 1, bias=False
            )
        else:
            self.numeric_wide_head = None
        self.output_bias = nn.Parameter(torch.zeros(()))

        self.reset_parameters()

    @staticmethod
    def _resolve_numeric_count(
        num_numeric: int | None,
        num_numerical: int | None,
    ) -> int:
        if num_numeric is None and num_numerical is None:
            raise TypeError(
                "missing required argument: 'num_numeric' "
                "(or 'num_numerical')"
            )
        primary = (
            None
            if num_numeric is None
            else _as_non_negative_int(num_numeric, "num_numeric")
        )
        alias = (
            None
            if num_numerical is None
            else _as_non_negative_int(num_numerical, "num_numerical")
        )
        if primary is not None and alias is not None and primary != alias:
            raise ValueError(
                "num_numeric and num_numerical must match when both are "
                "provided"
            )
        return primary if primary is not None else alias  # type: ignore[return-value]

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real number, not bool")
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number") from exc
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1)")
        return probability

    @staticmethod
    def _validate_entity_indices(
        values: Sequence[int], num_categorical: int
    ) -> tuple[int, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError("entity_cat_indices must be a sequence of integers")
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise TypeError(
                "entity_cat_indices must be a sequence of integers"
            ) from exc

        indices: list[int] = []
        seen: set[int] = set()
        for position, value in enumerate(raw_values):
            index = _as_non_negative_int(
                value, f"entity_cat_indices[{position}]"
            )
            # Defaults are useful across schemas: an index for a feature absent
            # from a smaller schema simply has no effect.
            if index >= num_categorical:
                continue
            if index in seen:
                raise ValueError("entity_cat_indices must not contain duplicates")
            seen.add(index)
            indices.append(index)
        return tuple(indices)

    @staticmethod
    def _embedding_dim(cardinality: int) -> int:
        """FastAI-style smooth heuristic, capped for large entity IDs."""

        return min(64, max(4, int(round(1.6 * cardinality**0.56))))

    def reset_parameters(self) -> None:
        """Initialize shallow paths conservatively and residual paths stably."""

        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        residual_gain = 1.0 / math.sqrt(2.0 * self.num_blocks)
        for block in self.blocks:
            nn.init.ones_(block.norm.weight)
            nn.init.zeros_(block.norm.bias)
            nn.init.xavier_uniform_(block.linear_in.weight)
            nn.init.zeros_(block.linear_in.bias)
            nn.init.xavier_uniform_(block.linear_out.weight, gain=residual_gain)
            nn.init.zeros_(block.linear_out.bias)

        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)
        nn.init.normal_(self.deep_head.weight, mean=0.0, std=0.01)
        if self.numeric_wide_head is not None:
            nn.init.normal_(self.numeric_wide_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_bias)

    def _drop_entity_embedding(self, embedding: Tensor) -> Tensor:
        if not self.training or self.embedding_dropout == 0.0:
            return embedding
        keep_probability = 1.0 - self.embedding_dropout
        mask = embedding.new_empty((embedding.shape[0], 1))
        mask.bernoulli_(keep_probability).div_(keep_probability)
        return embedding * mask

    def _validate_inputs(self, cat_x: Tensor, num_x: Tensor) -> None:
        if not isinstance(cat_x, Tensor):
            raise TypeError("cat_x must be a torch.Tensor")
        if not isinstance(num_x, Tensor):
            raise TypeError("num_x must be a torch.Tensor")
        if cat_x.ndim != 2:
            raise ValueError(
                f"cat_x must have shape [batch, {len(self.cardinalities)}], "
                f"got {tuple(cat_x.shape)}"
            )
        if num_x.ndim != 2:
            raise ValueError(
                f"num_x must have shape [batch, {self.num_numeric}], "
                f"got {tuple(num_x.shape)}"
            )
        if cat_x.shape[1] != len(self.cardinalities):
            raise ValueError(
                f"cat_x has {cat_x.shape[1]} columns; expected "
                f"{len(self.cardinalities)}"
            )
        if num_x.shape[1] != self.num_numeric:
            raise ValueError(
                f"num_x has {num_x.shape[1]} columns; expected "
                f"{self.num_numeric}"
            )
        if cat_x.shape[0] != num_x.shape[0]:
            raise ValueError(
                "cat_x and num_x must have the same batch dimension, got "
                f"{cat_x.shape[0]} and {num_x.shape[0]}"
            )
        if cat_x.dtype != torch.long:
            raise TypeError(f"cat_x must have dtype torch.long, got {cat_x.dtype}")
        if not torch.is_floating_point(num_x):
            raise TypeError(
                f"num_x must have a floating-point dtype, got {num_x.dtype}"
            )
        if cat_x.device != num_x.device:
            raise ValueError(
                "cat_x and num_x must be on the same device, got "
                f"{cat_x.device} and {num_x.device}"
            )

    def forward(self, cat_x: Tensor, num_x: Tensor) -> Tensor:
        """Return one unbounded logit per input row."""

        self._validate_inputs(cat_x, num_x)

        features: list[Tensor] = []
        for index, embedding_layer in enumerate(self.embeddings):
            embedded = embedding_layer(cat_x[:, index])
            if index in self._entity_cat_index_set:
                embedded = self._drop_entity_embedding(embedded)
            features.append(embedded)
        if self.num_numeric:
            features.append(num_x)

        deep_input = torch.cat(features, dim=1)
        hidden = self.input_projection(deep_input)
        hidden = self.input_activation(hidden)
        hidden = self.input_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)

        logits = self.deep_head(hidden).squeeze(-1) + self.output_bias
        if self.numeric_wide_head is not None:
            logits = logits + self.numeric_wide_head(num_x).squeeze(-1)
        return logits

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable constructor configuration."""

        return {
            "category_cardinalities": list(self.cardinalities),
            "num_numerical": self.num_numeric,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout,
            "embedding_dropout": self.embedding_dropout,
            "entity_cat_indices": list(self.entity_cat_indices),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TabularResidualModel":
        """Construct a model from :meth:`get_config` output."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        return cls(**dict(config))


_FORBIDDEN_MODEL_FEATURES = frozenset({"row_id", "control_success"})
_ENTITY_CATEGORICAL_GROUPS = {
    "pitcher": ("pitcher_id", "pitcher_hand", "pitcher_team_id"),
    "batter": ("batter_id", "batter_hand", "batter_team_id"),
}
_ENTITY_ID_FEATURES = {"pitcher": "pitcher_id", "batter": "batter_id"}


def _normalise_feature_names(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of feature names")
    try:
        names = tuple(str(value).strip() for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of feature names") from error
    if not allow_empty and not names:
        raise ValueError(f"{name} must not be empty")
    if any(not value for value in names):
        raise ValueError(f"{name} contains an empty feature name")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} contains duplicate feature names")
    forbidden = sorted(_FORBIDDEN_MODEL_FEATURES.intersection(names))
    if forbidden:
        raise ValueError(f"{name} contains forbidden model inputs: {forbidden}")
    return names


def _preprocessor_cardinalities(preprocessor: Any) -> tuple[int, ...]:
    values = getattr(preprocessor, "category_cardinalities_", None)
    if values is None:
        values = getattr(preprocessor, "category_cardinalities", None)
    cardinalities = _normalise_cardinalities(values, "category_cardinalities")
    if cardinalities is None:
        raise TypeError(
            "preprocessor must expose category_cardinalities_ or "
            "category_cardinalities"
        )
    return cardinalities


def _is_fitted_preprocessor(preprocessor: Any) -> bool:
    fitted_flags = [
        getattr(preprocessor, "is_fitted_", None),
        getattr(preprocessor, "_is_fitted", None),
    ]
    observed = [value for value in fitted_flags if value is not None]
    return bool(observed and any(bool(value) for value in observed))


def _canonical_preprocessor_fingerprint(preprocessor: Any) -> str:
    """Hash the complete target-independent encoding/scaling state."""

    from .preprocessing import DeepTabularPreprocessor

    if not isinstance(preprocessor, DeepTabularPreprocessor):
        raise TypeError(
            "ContextGatedResidualModel requires the canonical "
            "DeepTabularPreprocessor"
        )
    if not _is_fitted_preprocessor(preprocessor):
        raise RuntimeError("preprocessor must be fitted before fingerprinting")
    payload = preprocessor.to_state_dict()
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TabularFeatureLayout:
    """Name-resolved feature contract for the context-gated model.

    Numeric positions are never hard-coded.  The pitcher and batter branches
    are resolved from feature names, which keeps the model safe when a
    preprocessor changes column order.  Context features are the exhaustive
    complement of the two entity groups.
    """

    categorical_names: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]
    numerical_names: tuple[str, ...]
    pitcher_numerical_names: tuple[str, ...]
    batter_numerical_names: tuple[str, ...]
    preprocessor_fingerprint: str

    def __post_init__(self) -> None:
        categorical_names = _normalise_feature_names(
            self.categorical_names, name="categorical_names"
        )
        numerical_names = _normalise_feature_names(
            self.numerical_names, name="numerical_names"
        )
        cardinalities = _normalise_cardinalities(
            self.categorical_cardinalities, "categorical_cardinalities"
        )
        assert cardinalities is not None
        pitcher_numerical = _normalise_feature_names(
            self.pitcher_numerical_names,
            name="pitcher_numerical_names",
        )
        batter_numerical = _normalise_feature_names(
            self.batter_numerical_names,
            name="batter_numerical_names",
        )
        object.__setattr__(self, "categorical_names", categorical_names)
        object.__setattr__(self, "categorical_cardinalities", cardinalities)
        object.__setattr__(self, "numerical_names", numerical_names)
        object.__setattr__(self, "pitcher_numerical_names", pitcher_numerical)
        object.__setattr__(self, "batter_numerical_names", batter_numerical)
        fingerprint = str(self.preprocessor_fingerprint).strip().lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("preprocessor_fingerprint must be a SHA-256 hex digest")
        object.__setattr__(self, "preprocessor_fingerprint", fingerprint)

        if len(categorical_names) != len(cardinalities):
            raise ValueError(
                "categorical_names and categorical_cardinalities must have "
                "the same length"
            )
        too_small = [
            categorical_names[index]
            for index, cardinality in enumerate(cardinalities)
            if cardinality < 2
        ]
        if too_small:
            raise ValueError(
                "categorical cardinalities must reserve missing and OOV indices; "
                f"too small: {too_small}"
            )
        required_categorical = {
            value
            for group in _ENTITY_CATEGORICAL_GROUPS.values()
            for value in group
        }
        missing_categorical = sorted(required_categorical.difference(categorical_names))
        if missing_categorical:
            raise ValueError(
                "categorical layout is missing required entity features: "
                f"{missing_categorical}"
            )
        unknown_pitcher = sorted(set(pitcher_numerical).difference(numerical_names))
        unknown_batter = sorted(set(batter_numerical).difference(numerical_names))
        if unknown_pitcher or unknown_batter:
            raise ValueError(
                "entity numerical groups contain unknown features; "
                f"pitcher={unknown_pitcher}, batter={unknown_batter}"
            )
        overlap = sorted(set(pitcher_numerical).intersection(batter_numerical))
        if overlap:
            raise ValueError(
                f"pitcher and batter numerical groups overlap: {overlap}"
            )
        context_categorical = set(categorical_names).difference(required_categorical)
        context_numerical = set(numerical_names).difference(
            pitcher_numerical, batter_numerical
        )
        if not context_categorical and not context_numerical:
            raise ValueError("layout needs at least one context feature")

    @classmethod
    def from_preprocessor(cls, preprocessor: Any) -> "TabularFeatureLayout":
        """Build the canonical layout from a fitted tabular preprocessor."""

        if not _is_fitted_preprocessor(preprocessor):
            raise RuntimeError("preprocessor must be fitted before model layout creation")
        categorical_names = tuple(getattr(preprocessor, "categorical_columns"))
        numerical_names = tuple(getattr(preprocessor, "numerical_columns"))
        pitcher_numerical = tuple(
            name
            for name in numerical_names
            if name.startswith("asof_pitcher_")
            or name
            in {
                "pitcher_n_log1p",
                "pitcher_cold_start",
                "pitcher_history_reliability",
            }
        )
        batter_numerical = tuple(
            name
            for name in numerical_names
            if name.startswith("asof_batter_")
            or name
            in {
                "batter_n_log1p",
                "batter_cold_start",
                "batter_history_reliability",
            }
        )
        return cls(
            categorical_names=categorical_names,
            categorical_cardinalities=_preprocessor_cardinalities(preprocessor),
            numerical_names=numerical_names,
            pitcher_numerical_names=pitcher_numerical,
            batter_numerical_names=batter_numerical,
            preprocessor_fingerprint=_canonical_preprocessor_fingerprint(preprocessor),
        )

    def categorical_index(self, name: str) -> int:
        try:
            return self.categorical_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown categorical feature: {name}") from error

    def numerical_index(self, name: str) -> int:
        try:
            return self.numerical_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown numerical feature: {name}") from error

    @property
    def pitcher_categorical_names(self) -> tuple[str, ...]:
        return _ENTITY_CATEGORICAL_GROUPS["pitcher"]

    @property
    def batter_categorical_names(self) -> tuple[str, ...]:
        return _ENTITY_CATEGORICAL_GROUPS["batter"]

    @property
    def context_categorical_names(self) -> tuple[str, ...]:
        entities = set(self.pitcher_categorical_names + self.batter_categorical_names)
        return tuple(name for name in self.categorical_names if name not in entities)

    @property
    def context_numerical_names(self) -> tuple[str, ...]:
        entities = set(self.pitcher_numerical_names + self.batter_numerical_names)
        return tuple(name for name in self.numerical_names if name not in entities)

    def to_dict(self) -> dict[str, object]:
        return {
            "categorical_names": list(self.categorical_names),
            "categorical_cardinalities": list(self.categorical_cardinalities),
            "numerical_names": list(self.numerical_names),
            "pitcher_numerical_names": list(self.pitcher_numerical_names),
            "batter_numerical_names": list(self.batter_numerical_names),
            "preprocessor_fingerprint": self.preprocessor_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TabularFeatureLayout":
        if not isinstance(payload, Mapping):
            raise TypeError("feature layout payload must be a mapping")
        return cls(
            categorical_names=tuple(payload["categorical_names"]),
            categorical_cardinalities=tuple(payload["categorical_cardinalities"]),
            numerical_names=tuple(payload["numerical_names"]),
            pitcher_numerical_names=tuple(payload["pitcher_numerical_names"]),
            batter_numerical_names=tuple(payload["batter_numerical_names"]),
            preprocessor_fingerprint=str(payload["preprocessor_fingerprint"]),
        )


@dataclass(frozen=True)
class ContextGatedModelConfig:
    """Frozen, JSON-serializable architecture choices for the primary model."""

    tower_dim: int = 64
    hidden_dim: int = 128
    num_blocks: int = 3
    ffn_dim: int = 192
    dropout: float = 0.15
    entity_embedding_dropout: float = 0.10
    embedding_dim_cap: int = 32
    game_type_mode: str = "unrestricted"
    game_type_embedding_dropout: float = 0.0
    regime_logit_cap: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "tower_dim",
            "hidden_dim",
            "num_blocks",
            "ffn_dim",
            "embedding_dim_cap",
        ):
            value = _as_non_negative_int(getattr(self, name), name)
            if value == 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "dropout", TabularResidualModel._validate_probability(self.dropout, "dropout")
        )
        object.__setattr__(
            self,
            "entity_embedding_dropout",
            TabularResidualModel._validate_probability(
                self.entity_embedding_dropout, "entity_embedding_dropout"
            ),
        )
        mode = str(self.game_type_mode).strip().lower()
        allowed_modes = {
            "unrestricted",
            "no_bias",
            "neutral",
            "bounded_history_residual",
        }
        if mode not in allowed_modes:
            raise ValueError(
                "game_type_mode must be one of "
                f"{sorted(allowed_modes)}, got {self.game_type_mode!r}"
            )
        object.__setattr__(self, "game_type_mode", mode)
        object.__setattr__(
            self,
            "game_type_embedding_dropout",
            TabularResidualModel._validate_probability(
                self.game_type_embedding_dropout,
                "game_type_embedding_dropout",
            ),
        )
        cap = float(self.regime_logit_cap)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("regime_logit_cap must be positive and finite")
        object.__setattr__(self, "regime_logit_cap", cap)
        if mode in {"neutral", "bounded_history_residual"} and (
            self.game_type_embedding_dropout != 0.0
        ):
            raise ValueError(
                "game_type_embedding_dropout applies only when the unrestricted "
                "deep game_type path is enabled"
            )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tower_dim": self.tower_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "entity_embedding_dropout": self.entity_embedding_dropout,
            "embedding_dim_cap": self.embedding_dim_cap,
        }
        # Preserve exact compatibility with the already persisted V1 baseline:
        # new fields are emitted only for a non-default ablation procedure.
        if self.game_type_mode != "unrestricted":
            payload["game_type_mode"] = self.game_type_mode
        if self.game_type_embedding_dropout != 0.0:
            payload["game_type_embedding_dropout"] = self.game_type_embedding_dropout
        if self.regime_logit_cap != 0.5:
            payload["regime_logit_cap"] = self.regime_logit_cap
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextGatedModelConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("model config payload must be a mapping")
        return cls(**dict(payload))


class _InputTower(nn.Module):
    """Per-row input projection with feature-axis normalization only."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.activation(self.projection(self.norm(inputs))))


class _GatedResidualBlock(nn.Module):
    """GELU-gated pre-norm residual block with conservative LayerScale."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.value_gate = nn.Linear(hidden_dim, ffn_dim * 2)
        self.output = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((hidden_dim,), 1e-2))

    def forward(self, inputs: Tensor) -> Tensor:
        value, gate = self.value_gate(self.norm(inputs)).chunk(2, dim=-1)
        update = torch.nn.functional.gelu(value) * torch.sigmoid(gate)
        update = self.output(self.dropout(update))
        return inputs + self.layer_scale * self.dropout(update)


class ContextGatedResidualModel(nn.Module):
    """Cold-start-safe, row-local three-tower wide-and-deep network.

    The context, pitcher-history, and batter-history paths are projected
    separately.  Player ID embeddings enter only as gated residuals; missing
    or OOV entity IDs therefore fall back to row-local history and context.
    Every missing/OOV categorical token is zeroed before the towers, avoiding
    untrained random OOV embeddings for future seasons and new entities.
    """

    architecture_name = "context_gated_residual_v1"
    reserved_category_count = 2
    _REGIME_HISTORY_NUMERICAL_NAMES = (
        "pitcher_n_log1p",
        "pitcher_cold_start",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_success_rate__missing",
        "asof_pitcher_reverse_rate__missing",
        "asof_pitcher_prev1_game_success_rate__missing",
        "asof_pitcher_prev3_game_success_rate__missing",
        "asof_pitcher_prev5_game_success_rate__missing",
    )

    def __init__(
        self,
        feature_layout: TabularFeatureLayout,
        config: ContextGatedModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_layout, TabularFeatureLayout):
            raise TypeError("feature_layout must be a TabularFeatureLayout")
        if config is None:
            config = ContextGatedModelConfig()
        if not isinstance(config, ContextGatedModelConfig):
            raise TypeError("config must be a ContextGatedModelConfig")
        self.feature_layout = feature_layout
        self.config = config

        self.embedding_dims = tuple(
            min(
                config.embedding_dim_cap,
                TabularResidualModel._embedding_dim(cardinality),
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
        self._categorical_positions = {
            name: position for position, name in enumerate(feature_layout.categorical_names)
        }
        self._numerical_positions = {
            name: position for position, name in enumerate(feature_layout.numerical_names)
        }
        self._pitcher_id_position = self._categorical_positions[
            _ENTITY_ID_FEATURES["pitcher"]
        ]
        self._batter_id_position = self._categorical_positions[
            _ENTITY_ID_FEATURES["batter"]
        ]
        self._game_type_position = self._categorical_positions["game_type"]
        game_type_unrestricted = config.game_type_mode in {
            "unrestricted",
            "no_bias",
        }
        game_type_has_direct_bias = config.game_type_mode == "unrestricted"
        self._categorical_bias_positions = tuple(
            position
            for position in range(len(feature_layout.categorical_names))
            if position not in {self._pitcher_id_position, self._batter_id_position}
            and (position != self._game_type_position or game_type_has_direct_bias)
        )
        self.categorical_biases = nn.ModuleDict(
            {
                str(position): nn.Embedding(
                    feature_layout.categorical_cardinalities[position], 1
                )
                for position in self._categorical_bias_positions
            }
        )

        context_cat_names = tuple(
            name
            for name in feature_layout.context_categorical_names
            if name != "game_type" or game_type_unrestricted
        )
        pitcher_base_cat_names = tuple(
            name
            for name in feature_layout.pitcher_categorical_names
            if name != _ENTITY_ID_FEATURES["pitcher"]
        )
        batter_base_cat_names = tuple(
            name
            for name in feature_layout.batter_categorical_names
            if name != _ENTITY_ID_FEATURES["batter"]
        )
        self._context_cat_positions = tuple(
            self._categorical_positions[name] for name in context_cat_names
        )
        self._pitcher_base_cat_positions = tuple(
            self._categorical_positions[name] for name in pitcher_base_cat_names
        )
        self._batter_base_cat_positions = tuple(
            self._categorical_positions[name] for name in batter_base_cat_names
        )
        self._context_num_positions = tuple(
            self._numerical_positions[name]
            for name in feature_layout.context_numerical_names
        )
        self._pitcher_num_positions = tuple(
            self._numerical_positions[name]
            for name in feature_layout.pitcher_numerical_names
        )
        self._batter_num_positions = tuple(
            self._numerical_positions[name]
            for name in feature_layout.batter_numerical_names
        )

        self.register_buffer(
            "_cardinality_limits",
            torch.tensor(feature_layout.categorical_cardinalities, dtype=torch.long),
            persistent=False,
        )
        for buffer_name, values in (
            ("_context_num_index", self._context_num_positions),
            ("_pitcher_num_index", self._pitcher_num_positions),
            ("_batter_num_index", self._batter_num_positions),
        ):
            self.register_buffer(
                buffer_name,
                torch.tensor(values, dtype=torch.long),
                persistent=False,
            )

        regime_history_positions: tuple[int, ...] = ()
        if config.game_type_mode == "bounded_history_residual":
            forbidden = {"control_success", "reverse", "current_reverse"}
            if forbidden.intersection(self._REGIME_HISTORY_NUMERICAL_NAMES):
                raise ValueError("regime history allowlist contains a forbidden feature")
            missing = sorted(
                set(self._REGIME_HISTORY_NUMERICAL_NAMES).difference(
                    self._numerical_positions
                )
            )
            if missing:
                raise ValueError(f"regime history features are missing: {missing}")
            regime_history_positions = tuple(
                self._numerical_positions[name]
                for name in self._REGIME_HISTORY_NUMERICAL_NAMES
            )
        self.register_buffer(
            "_regime_history_num_index",
            torch.tensor(regime_history_positions, dtype=torch.long),
            persistent=False,
        )

        context_input_dim = self._embedded_width(self._context_cat_positions) + len(
            self._context_num_positions
        )
        pitcher_base_input_dim = self._embedded_width(
            self._pitcher_base_cat_positions
        ) + len(self._pitcher_num_positions)
        batter_base_input_dim = self._embedded_width(
            self._batter_base_cat_positions
        ) + len(self._batter_num_positions)
        if min(context_input_dim, pitcher_base_input_dim, batter_base_input_dim) <= 0:
            raise ValueError("each model tower must receive at least one feature")

        tower_dim = config.tower_dim
        self.context_tower = _InputTower(context_input_dim, tower_dim, config.dropout)
        self.pitcher_base_tower = _InputTower(
            pitcher_base_input_dim, tower_dim, config.dropout
        )
        self.batter_base_tower = _InputTower(
            batter_base_input_dim, tower_dim, config.dropout
        )
        self.pitcher_id_projection = nn.Linear(
            self.embedding_dims[self._pitcher_id_position], tower_dim, bias=False
        )
        self.batter_id_projection = nn.Linear(
            self.embedding_dims[self._batter_id_position], tower_dim, bias=False
        )
        self.pitcher_reliability_gate = nn.Sequential(
            nn.LayerNorm(tower_dim * 2), nn.Linear(tower_dim * 2, 1), nn.Sigmoid()
        )
        self.batter_reliability_gate = nn.Sequential(
            nn.LayerNorm(tower_dim * 2), nn.Linear(tower_dim * 2, 1), nn.Sigmoid()
        )

        interaction_dim = tower_dim * 7
        self.interaction_projection = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(
            _GatedResidualBlock(
                config.hidden_dim,
                config.ffn_dim,
                config.dropout,
            )
            for _ in range(config.num_blocks)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.deep_head = nn.Linear(config.hidden_dim, 1, bias=False)
        self.numeric_wide_head = nn.Linear(
            len(feature_layout.numerical_names), 1, bias=False
        )
        if config.game_type_mode == "bounded_history_residual":
            history_width = len(self._REGIME_HISTORY_NUMERICAL_NAMES)
            game_type_width = self.embedding_dims[self._game_type_position]
            self.regime_history_norm: nn.Module | None = nn.LayerNorm(history_width)
            self.regime_history_projection: nn.Module | None = nn.Linear(
                history_width,
                game_type_width,
                bias=False,
            )
            self.regime_residual_head: nn.Module | None = nn.Linear(
                game_type_width,
                1,
                bias=False,
            )
        else:
            self.regime_history_norm = None
            self.regime_history_projection = None
            self.regime_residual_head = None
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    @property
    def pitcher_id_index(self) -> int:
        return self._pitcher_id_position

    @property
    def batter_id_index(self) -> int:
        return self._batter_id_position

    def _embedded_width(self, positions: Sequence[int]) -> int:
        return sum(self.embedding_dims[position] for position in positions)

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[: self.reserved_category_count].zero_()
        for bias_embedding in self.categorical_biases.values():
            nn.init.zeros_(bias_embedding.weight)

        for tower in (
            self.context_tower,
            self.pitcher_base_tower,
            self.batter_base_tower,
        ):
            nn.init.ones_(tower.norm.weight)
            nn.init.zeros_(tower.norm.bias)
            nn.init.xavier_uniform_(tower.projection.weight)
            nn.init.zeros_(tower.projection.bias)
        for projection in (self.pitcher_id_projection, self.batter_id_projection):
            nn.init.xavier_uniform_(projection.weight)
        for gate in (self.pitcher_reliability_gate, self.batter_reliability_gate):
            norm = gate[0]
            linear = gate[1]
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        interaction_linear = self.interaction_projection[1]
        nn.init.xavier_uniform_(interaction_linear.weight)
        nn.init.zeros_(interaction_linear.bias)
        for block in self.blocks:
            nn.init.ones_(block.norm.weight)
            nn.init.zeros_(block.norm.bias)
            nn.init.xavier_uniform_(block.value_gate.weight)
            nn.init.zeros_(block.value_gate.bias)
            nn.init.xavier_uniform_(block.output.weight)
            nn.init.zeros_(block.output.bias)
            nn.init.constant_(block.layer_scale, 1e-2)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)
        nn.init.normal_(self.deep_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.numeric_wide_head.weight, mean=0.0, std=0.01)
        if self.regime_history_norm is not None:
            assert isinstance(self.regime_history_norm, nn.LayerNorm)
            assert isinstance(self.regime_history_projection, nn.Linear)
            assert isinstance(self.regime_residual_head, nn.Linear)
            nn.init.ones_(self.regime_history_norm.weight)
            nn.init.zeros_(self.regime_history_norm.bias)
            nn.init.xavier_uniform_(self.regime_history_projection.weight)
            # The bounded regime path starts as an exact no-op.
            nn.init.zeros_(self.regime_residual_head.weight)
        with torch.no_grad():
            self.output_bias.zero_()

    def _validate_inputs(self, categorical: Tensor, numerical: Tensor) -> None:
        if not isinstance(categorical, Tensor) or not isinstance(numerical, Tensor):
            raise TypeError("categorical and numerical inputs must be torch tensors")
        if categorical.ndim != 2 or categorical.shape[1] != len(
            self.feature_layout.categorical_names
        ):
            raise ValueError(
                "categorical must have shape "
                f"[batch, {len(self.feature_layout.categorical_names)}]"
            )
        if numerical.ndim != 2 or numerical.shape[1] != len(
            self.feature_layout.numerical_names
        ):
            raise ValueError(
                "numerical must have shape "
                f"[batch, {len(self.feature_layout.numerical_names)}]"
            )
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

    def _embedded_categories(self, categorical: Tensor) -> list[Tensor]:
        embedded: list[Tensor] = []
        for position, embedding_layer in enumerate(self.embeddings):
            values = categorical[:, position]
            representation = embedding_layer(values)
            known = values.ge(self.reserved_category_count).unsqueeze(1)
            representation = representation * known.to(representation.dtype)
            if (
                position == self._game_type_position
                and self.training
                and self.config.game_type_embedding_dropout > 0.0
            ):
                keep_probability = 1.0 - self.config.game_type_embedding_dropout
                mask = representation.new_empty((representation.shape[0], 1))
                mask.bernoulli_(keep_probability).div_(keep_probability)
                representation = representation * mask
            embedded.append(representation)
        return embedded

    @staticmethod
    def _concat_parts(parts: Sequence[Tensor]) -> Tensor:
        if not parts:
            raise RuntimeError("model feature group unexpectedly became empty")
        return torch.cat(tuple(parts), dim=1)

    def _select_numerical(self, numerical: Tensor, index: Tensor) -> Tensor:
        return torch.index_select(numerical, 1, index)

    def _entity_embedding_dropout(self, inputs: Tensor) -> Tensor:
        probability = self.config.entity_embedding_dropout
        if not self.training or probability == 0.0:
            return inputs
        keep_probability = 1.0 - probability
        mask = inputs.new_empty((inputs.shape[0], 1))
        mask.bernoulli_(keep_probability).div_(keep_probability)
        return inputs * mask

    def forward_with_aux(
        self,
        categorical: Tensor,
        numerical: Tensor,
    ) -> dict[str, Tensor]:
        """Return logits and row-level gates for architecture auditing."""

        self._validate_inputs(categorical, numerical)
        embedded = self._embedded_categories(categorical)

        context_input = self._concat_parts(
            [embedded[position] for position in self._context_cat_positions]
            + [self._select_numerical(numerical, self._context_num_index)]
        )
        pitcher_base_input = self._concat_parts(
            [embedded[position] for position in self._pitcher_base_cat_positions]
            + [self._select_numerical(numerical, self._pitcher_num_index)]
        )
        batter_base_input = self._concat_parts(
            [embedded[position] for position in self._batter_base_cat_positions]
            + [self._select_numerical(numerical, self._batter_num_index)]
        )

        context = self.context_tower(context_input)
        pitcher_base = self.pitcher_base_tower(pitcher_base_input)
        batter_base = self.batter_base_tower(batter_base_input)

        pitcher_values = categorical[:, self._pitcher_id_position]
        batter_values = categorical[:, self._batter_id_position]
        pitcher_known = pitcher_values.ge(self.reserved_category_count).to(numerical.dtype)
        batter_known = batter_values.ge(self.reserved_category_count).to(numerical.dtype)
        pitcher_id_delta = self.pitcher_id_projection(
            self._entity_embedding_dropout(embedded[self._pitcher_id_position])
        )
        batter_id_delta = self.batter_id_projection(
            self._entity_embedding_dropout(embedded[self._batter_id_position])
        )
        pitcher_gate = self.pitcher_reliability_gate(
            torch.cat([context, pitcher_base], dim=1)
        ).squeeze(1) * pitcher_known
        batter_gate = self.batter_reliability_gate(
            torch.cat([context, batter_base], dim=1)
        ).squeeze(1) * batter_known
        pitcher = pitcher_base + pitcher_gate.unsqueeze(1) * pitcher_id_delta
        batter = batter_base + batter_gate.unsqueeze(1) * batter_id_delta

        interactions = torch.cat(
            [
                pitcher,
                batter,
                context,
                pitcher * batter,
                pitcher * context,
                batter * context,
                torch.abs(pitcher - batter),
            ],
            dim=1,
        )
        hidden = self.interaction_projection(interactions)
        for block in self.blocks:
            hidden = block(hidden)
        deep_logit = self.deep_head(self.output_norm(hidden)).squeeze(1)
        numeric_wide_logit = self.numeric_wide_head(numerical).squeeze(1)
        categorical_bias_logit = torch.zeros_like(deep_logit)
        for position in self._categorical_bias_positions:
            bias_embedding = self.categorical_biases[str(position)]
            values = categorical[:, position]
            known = values.ge(self.reserved_category_count).to(numerical.dtype)
            categorical_bias_logit = categorical_bias_logit + (
                bias_embedding(values).squeeze(1) * known
            )
        regime_residual_logit = torch.zeros_like(deep_logit)
        if self.regime_residual_head is not None:
            assert isinstance(self.regime_history_norm, nn.LayerNorm)
            assert isinstance(self.regime_history_projection, nn.Linear)
            history = self._select_numerical(
                numerical,
                self._regime_history_num_index,
            )
            history_state = torch.tanh(
                self.regime_history_projection(self.regime_history_norm(history))
            )
            game_type_state = embedded[self._game_type_position]
            raw_residual = self.regime_residual_head(
                history_state * game_type_state
            ).squeeze(1)
            regime_residual_logit = self.config.regime_logit_cap * torch.tanh(
                raw_residual
            )
        logits = (
            self.output_bias
            + deep_logit
            + numeric_wide_logit
            + categorical_bias_logit
            + regime_residual_logit
        )
        return {
            "logits": logits,
            "pitcher_id_gate": pitcher_gate,
            "batter_id_gate": batter_gate,
            "deep_logit": deep_logit,
            "numeric_wide_logit": numeric_wide_logit,
            "categorical_bias_logit": categorical_bias_logit,
            "regime_residual_logit": regime_residual_logit,
        }

    def forward(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        """Return one finite-candidate logit per independent input row."""

        return self.forward_with_aux(categorical, numerical)["logits"]

    def get_config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture_name,
            "feature_layout": self.feature_layout.to_dict(),
            "model": self.config.to_dict(),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> "ContextGatedResidualModel":
        if not isinstance(payload, Mapping):
            raise TypeError("config must be a mapping")
        if payload.get("architecture") != cls.architecture_name:
            raise ValueError(
                f"unsupported architecture: {payload.get('architecture')!r}"
            )
        return cls(
            TabularFeatureLayout.from_dict(payload["feature_layout"]),
            ContextGatedModelConfig.from_dict(payload["model"]),
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

    def feature_layout_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for position, (name, cardinality, dimension) in enumerate(
            zip(
                self.feature_layout.categorical_names,
                self.feature_layout.categorical_cardinalities,
                self.embedding_dims,
                strict=True,
            )
        ):
            if name in self.feature_layout.pitcher_categorical_names:
                group = "pitcher"
            elif name in self.feature_layout.batter_categorical_names:
                group = "batter"
            else:
                group = "context"
            records.append(
                {
                    "position": position,
                    "feature": name,
                    "kind": "categorical",
                    "group": group,
                    "cardinality": cardinality,
                    "embedding_dim": dimension,
                }
            )
        for position, name in enumerate(self.feature_layout.numerical_names):
            if name in self.feature_layout.pitcher_numerical_names:
                group = "pitcher"
            elif name in self.feature_layout.batter_numerical_names:
                group = "batter"
            else:
                group = "context"
            records.append(
                {
                    "position": position,
                    "feature": name,
                    "kind": "numerical",
                    "group": group,
                    "cardinality": None,
                    "embedding_dim": None,
                }
            )
        return records


def build_context_gated_model(
    preprocessor: Any,
    *,
    config: ContextGatedModelConfig | None = None,
    initialization_seed: int = 42,
) -> ContextGatedResidualModel:
    """Create a deterministically initialized model from named preprocessing state.

    ``torch.random.fork_rng`` restores the caller's ambient RNG state after
    construction, so initialization is a pure function of the explicit seed.
    The model is built on CPU and can be moved to a target device afterwards.
    """

    seed = _as_non_negative_int(initialization_seed, "initialization_seed")
    feature_layout = TabularFeatureLayout.from_preprocessor(preprocessor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ContextGatedResidualModel(feature_layout, config)
    model.initialization_seed = seed
    return model


def assert_preprocessor_compatible(
    feature_layout: TabularFeatureLayout,
    preprocessor: Any,
) -> None:
    """Reject a same-shaped preprocessor whose token/scaler state differs."""

    if not isinstance(feature_layout, TabularFeatureLayout):
        raise TypeError("feature_layout must be a TabularFeatureLayout")
    candidate = TabularFeatureLayout.from_preprocessor(preprocessor)
    if candidate.categorical_names != feature_layout.categorical_names:
        raise ValueError("preprocessor categorical feature names/order do not match model")
    if candidate.numerical_names != feature_layout.numerical_names:
        raise ValueError("preprocessor numerical feature names/order do not match model")
    if candidate.categorical_cardinalities != feature_layout.categorical_cardinalities:
        raise ValueError("preprocessor cardinalities do not match model")
    if candidate.preprocessor_fingerprint != feature_layout.preprocessor_fingerprint:
        raise ValueError(
            "preprocessor vocabulary/scaler fingerprint does not match model layout"
        )


def validate_transformed_arrays(
    feature_layout: TabularFeatureLayout,
    transformed: Mapping[str, Any],
    *,
    expected_row_ids: Sequence[object] | np.ndarray | None = None,
) -> dict[str, object]:
    """Validate the exact NumPy contract between preprocessing and the model."""

    if not isinstance(feature_layout, TabularFeatureLayout):
        raise TypeError("feature_layout must be a TabularFeatureLayout")
    if not isinstance(transformed, Mapping):
        raise TypeError("transformed arrays must be supplied as a mapping")
    missing = sorted({"categorical", "numerical"}.difference(transformed))
    if missing:
        raise ValueError(f"transformed arrays are missing required keys: {missing}")
    categorical = np.asarray(transformed["categorical"])
    numerical = np.asarray(transformed["numerical"])
    if categorical.ndim != 2 or categorical.shape[1] != len(
        feature_layout.categorical_names
    ):
        raise ValueError(
            "categorical array must have shape "
            f"[rows, {len(feature_layout.categorical_names)}]"
        )
    if numerical.ndim != 2 or numerical.shape[1] != len(
        feature_layout.numerical_names
    ):
        raise ValueError(
            "numerical array must have shape "
            f"[rows, {len(feature_layout.numerical_names)}]"
        )
    if categorical.shape[0] != numerical.shape[0]:
        raise ValueError("categorical and numerical arrays have different row counts")
    if categorical.dtype != np.int64:
        raise TypeError(f"categorical array must be int64, got {categorical.dtype}")
    if numerical.dtype != np.float32:
        raise TypeError(f"numerical array must be float32, got {numerical.dtype}")
    if categorical.size:
        limits = np.asarray(feature_layout.categorical_cardinalities, dtype=np.int64)
        if (categorical < 0).any() or (categorical >= limits[None, :]).any():
            raise ValueError("categorical array contains an index outside its vocabulary")
    if not np.isfinite(numerical).all():
        raise ValueError("numerical array contains NaN or infinity")

    if expected_row_ids is not None:
        if "row_id" not in transformed:
            raise ValueError("row_id is required when expected_row_ids is supplied")
        expected = np.asarray(expected_row_ids, dtype=object).reshape(-1)
        actual = np.asarray(transformed["row_id"], dtype=object).reshape(-1)
        if len(actual) != categorical.shape[0]:
            raise ValueError("row_id count differs from transformed row count")
        if not np.array_equal(actual, expected):
            raise ValueError("transformed row_id values or order differ from expectation")
    return {
        "rows": int(categorical.shape[0]),
        "categorical_features": int(categorical.shape[1]),
        "numerical_features": int(numerical.shape[1]),
        "categorical_dtype": str(categorical.dtype),
        "numerical_dtype": str(numerical.dtype),
        "all_numerical_finite": True,
        "target_consumed_by_model": False,
    }


__all__ = [
    "TabularResidualModel",
    "TabularFeatureLayout",
    "ContextGatedModelConfig",
    "ContextGatedResidualModel",
    "build_context_gated_model",
    "assert_preprocessor_compatible",
    "validate_transformed_arrays",
]
