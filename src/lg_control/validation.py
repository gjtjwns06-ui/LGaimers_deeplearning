"""Leakage-safe temporal validation design for the competition data.

This module only defines and audits row partitions.  It deliberately contains
no model, optimisation, metric, prediction, or checkpoint code.

The outer validation season is reserved for one final fold evaluation.  Any
epoch/checkpoint or hyperparameter selection must use ``inner_stop_split``,
which is wholly contained in the corresponding outer training rows.  Stateful
preprocessing should be created through the helpers at the bottom of this
module so that ``fit`` receives training rows only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .constants import ID_COL, TARGET_COL
from .schema import SchemaError, validate_input_frame


EXPECTED_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
ROLLING_VALIDATION_SEASONS = (2022, 2023, 2024)
PRIMARY_VALIDATION_SEASON = 2024
DEVELOPMENT_SOURCE_SEASONS = (2019, 2020, 2021, 2022, 2023)
DEVELOPMENT_VALIDATION_SEASONS = (2022, 2023)


class ValidationDesignError(ValueError):
    """Raised when a fold or its source data violates the validation contract."""


def _readonly_int_array(values: Iterable[int], *, name: str) -> np.ndarray:
    raw = np.asarray(list(values))
    if raw.ndim != 1:
        raise ValidationDesignError(f"{name} must be one-dimensional")
    if raw.size:
        try:
            numeric = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValidationDesignError(f"{name} must contain integer positions") from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValidationDesignError(f"{name} must contain finite integer positions")
        result = numeric.astype(np.int64)
    else:
        result = np.empty(0, dtype=np.int64)
    result.setflags(write=False)
    return result


def _normalise_seasons(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    raw = list(values)
    if not raw:
        raise ValidationDesignError(f"{name} must not be empty")
    result: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValidationDesignError(f"{name} must contain integer years, not booleans")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValidationDesignError(f"{name} must contain integer years") from error
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValidationDesignError(f"{name} must contain finite integer years")
        result.append(int(numeric))
    if len(set(result)) != len(result):
        raise ValidationDesignError(f"{name} contains duplicate seasons: {result}")
    return tuple(sorted(result))


@dataclass(frozen=True)
class _SourceRows:
    """Shared immutable row identity used by every partition in a plan."""

    row_ids: np.ndarray = field(repr=False, compare=False)
    seasons: np.ndarray = field(repr=False, compare=False)
    id_column: str = ID_COL
    season_column: str = "season"

    def __post_init__(self) -> None:
        row_ids = np.asarray(self.row_ids, dtype=object).reshape(-1).copy()
        seasons = np.asarray(self.seasons, dtype=np.int64).reshape(-1).copy()
        if len(row_ids) != len(seasons):
            raise ValidationDesignError("source row IDs and seasons have different lengths")
        row_ids.setflags(write=False)
        seasons.setflags(write=False)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "seasons", seasons)

    @property
    def row_count(self) -> int:
        return int(len(self.row_ids))

    def assert_matches(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if self.id_column not in frame.columns:
            raise ValidationDesignError(f"frame is missing identity column {self.id_column!r}")
        if len(frame) != self.row_count:
            raise ValidationDesignError(
                f"source row count changed: expected {self.row_count}, found {len(frame)}"
            )
        actual = frame[self.id_column].to_numpy(dtype=object, copy=False)
        if not np.array_equal(actual, self.row_ids):
            raise ValidationDesignError(
                "source row order/identity changed after the validation plan was built"
            )

    def row_ids_at(self, positions: np.ndarray) -> np.ndarray:
        result = self.row_ids[positions].copy()
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class SeasonPartition:
    """One chronological train/validation partition in original row order."""

    name: str
    train_seasons: tuple[int, ...]
    validation_season: int
    train_positions: np.ndarray = field(repr=False, compare=False)
    validation_positions: np.ndarray = field(repr=False, compare=False)
    _source: _SourceRows = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationDesignError("partition name must be a non-empty string")
        train_seasons = _normalise_seasons(self.train_seasons, name="train_seasons")
        if isinstance(self.validation_season, (bool, np.bool_)):
            raise ValidationDesignError("validation_season must be an integer year")
        try:
            validation_numeric = float(self.validation_season)
        except (TypeError, ValueError) as error:
            raise ValidationDesignError("validation_season must be an integer year") from error
        if not np.isfinite(validation_numeric) or validation_numeric != np.floor(validation_numeric):
            raise ValidationDesignError("validation_season must be a finite integer year")
        validation_season = int(validation_numeric)
        train_positions = _readonly_int_array(self.train_positions, name="train_positions")
        validation_positions = _readonly_int_array(
            self.validation_positions, name="validation_positions"
        )

        object.__setattr__(self, "train_seasons", train_seasons)
        object.__setattr__(self, "validation_season", validation_season)
        object.__setattr__(self, "train_positions", train_positions)
        object.__setattr__(self, "validation_positions", validation_positions)

        if not len(train_positions):
            raise ValidationDesignError(f"{self.name}: training partition is empty")
        if not len(validation_positions):
            raise ValidationDesignError(f"{self.name}: validation partition is empty")
        if max(train_seasons) >= validation_season:
            raise ValidationDesignError(
                f"{self.name}: train max season must be earlier than validation season"
            )

        all_positions = np.concatenate([train_positions, validation_positions])
        if (all_positions < 0).any() or (all_positions >= self._source.row_count).any():
            raise ValidationDesignError(f"{self.name}: row positions are outside source bounds")
        if len(np.unique(train_positions)) != len(train_positions):
            raise ValidationDesignError(f"{self.name}: duplicate training row positions")
        if len(np.unique(validation_positions)) != len(validation_positions):
            raise ValidationDesignError(f"{self.name}: duplicate validation row positions")
        if np.intersect1d(train_positions, validation_positions, assume_unique=True).size:
            raise ValidationDesignError(f"{self.name}: training and validation rows overlap")

        expected_train = np.flatnonzero(np.isin(self._source.seasons, train_seasons))
        expected_validation = np.flatnonzero(
            self._source.seasons == validation_season
        )
        if not np.array_equal(train_positions, expected_train):
            raise ValidationDesignError(
                f"{self.name}: training positions do not preserve every source row "
                f"from seasons {train_seasons} in source order"
            )
        if not np.array_equal(validation_positions, expected_validation):
            raise ValidationDesignError(
                f"{self.name}: validation positions do not preserve every source row "
                f"from season {validation_season} in source order"
            )

        train_values = self._source.seasons[train_positions]
        validation_values = self._source.seasons[validation_positions]
        if int(train_values.max()) >= int(validation_values.min()):
            raise ValidationDesignError(
                f"{self.name}: train max season must be less than validation min season"
            )

    @property
    def train_row_ids(self) -> np.ndarray:
        return self._source.row_ids_at(self.train_positions)

    @property
    def validation_row_ids(self) -> np.ndarray:
        return self._source.row_ids_at(self.validation_positions)

    @property
    def train_row_count(self) -> int:
        return int(len(self.train_positions))

    @property
    def validation_row_count(self) -> int:
        return int(len(self.validation_positions))

    def split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return copies with original index labels, row order, IDs, and target intact."""

        train = self.training_frame(frame)
        validation = self.validation_frame(frame)
        return train, validation

    def training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only training rows, without materializing validation rows."""

        self._source.assert_matches(frame)
        train = frame.iloc[self.train_positions].copy()
        if not np.array_equal(
            train[self._source.id_column].to_numpy(dtype=object), self.train_row_ids
        ):
            raise ValidationDesignError(f"{self.name}: training row IDs were not preserved")
        return train

    def validation_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only validation rows, for an explicitly authorized consumer."""

        self._source.assert_matches(frame)
        validation = frame.iloc[self.validation_positions].copy()
        if not np.array_equal(
            validation[self._source.id_column].to_numpy(dtype=object),
            self.validation_row_ids,
        ):
            raise ValidationDesignError(f"{self.name}: validation row IDs were not preserved")
        return validation

    def to_summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "train_seasons": list(self.train_seasons),
            "validation_season": self.validation_season,
            "train_rows": self.train_row_count,
            "validation_rows": self.validation_row_count,
        }


@dataclass(frozen=True)
class SeasonFold(SeasonPartition):
    """Outer fold with an optional train-contained split for model selection."""

    role: str = "rolling"
    inner_stop_split: SeasonPartition | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        role = str(self.role).strip()
        if not role:
            raise ValidationDesignError(f"{self.name}: role must be a non-empty string")
        object.__setattr__(self, "role", role)

        inner = self.inner_stop_split
        if inner is None:
            return
        if inner._source is not self._source:
            raise ValidationDesignError(f"{self.name}: inner split must use the same source rows")
        if inner.validation_season >= self.validation_season:
            raise ValidationDesignError(
                f"{self.name}: inner validation season must precede outer validation"
            )
        inner_rows = np.union1d(inner.train_positions, inner.validation_positions)
        if not np.array_equal(inner_rows, self.train_positions):
            raise ValidationDesignError(
                f"{self.name}: inner train plus inner validation must exactly equal outer train"
            )
        if np.intersect1d(inner_rows, self.validation_positions).size:
            raise ValidationDesignError(
                f"{self.name}: outer validation leaked into inner selection rows"
            )

    @property
    def inner_train_seasons(self) -> tuple[int, ...]:
        return () if self.inner_stop_split is None else self.inner_stop_split.train_seasons

    @property
    def inner_validation_season(self) -> int | None:
        return (
            None
            if self.inner_stop_split is None
            else self.inner_stop_split.validation_season
        )

    def inner_selection_frames(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return only inner train/stop frames; outer validation is never returned."""

        if self.inner_stop_split is None:
            raise ValidationDesignError(
                f"{self.name}: no inner split; this diagnostic fold cannot select epochs"
            )
        assert_selection_scope(
            self,
            np.concatenate(
                [
                    self.inner_stop_split.train_positions,
                    self.inner_stop_split.validation_positions,
                ]
            ),
        )
        return self.inner_stop_split.split(frame)

    def to_summary(self) -> dict[str, object]:
        result = super().to_summary()
        result["role"] = self.role
        result["inner_stop_split"] = (
            None
            if self.inner_stop_split is None
            else self.inner_stop_split.to_summary()
        )
        return result


@dataclass(frozen=True)
class ValidationPlan:
    """Competition validation plan without duplicate execution of the 2024 fold."""

    rolling_folds: tuple[SeasonFold, ...]
    primary_holdout: SeasonFold
    recent_only_diagnostic: SeasonFold | None = None

    def __post_init__(self) -> None:
        if not self.rolling_folds:
            raise ValidationDesignError("rolling_folds must not be empty")
        names = [fold.name for fold in self.rolling_folds]
        if len(set(names)) != len(names):
            raise ValidationDesignError(f"duplicate fold names: {names}")
        validation_seasons = tuple(fold.validation_season for fold in self.rolling_folds)
        if validation_seasons != tuple(sorted(validation_seasons)):
            raise ValidationDesignError("rolling folds must be ordered by validation season")
        if self.primary_holdout is not self.rolling_folds[-1]:
            raise ValidationDesignError(
                "primary_holdout must alias the final rolling fold to avoid duplicate execution"
            )
        if self.primary_holdout.role != "anchor":
            raise ValidationDesignError("primary holdout must carry role='anchor'")
        if self.primary_holdout.validation_season != PRIMARY_VALIDATION_SEASON:
            raise ValidationDesignError(
                f"primary holdout must validate season {PRIMARY_VALIDATION_SEASON}"
            )
        if self.recent_only_diagnostic is not None:
            if self.recent_only_diagnostic.name in set(names):
                raise ValidationDesignError("diagnostic fold name duplicates a rolling fold")
            if self.recent_only_diagnostic.role != "diagnostic":
                raise ValidationDesignError("recent-only fold must carry the diagnostic role")
            if self.recent_only_diagnostic.inner_stop_split is not None:
                raise ValidationDesignError(
                    "recent-only diagnostic must not define an inner selection split"
                )

    def execution_folds(self, *, include_diagnostic: bool = True) -> tuple[SeasonFold, ...]:
        """Return each outer partition once; primary is already rolling fold 2024."""

        if include_diagnostic and self.recent_only_diagnostic is not None:
            return (*self.rolling_folds, self.recent_only_diagnostic)
        return self.rolling_folds

    @property
    def development_folds(self) -> tuple[SeasonFold, ...]:
        """Return folds allowed for architecture development before the anchor.

        The 2024 anchor is intentionally excluded so callers do not
        accidentally iterate over it while choosing a model design.
        """

        return tuple(fold for fold in self.rolling_folds if fold is not self.primary_holdout)

    @property
    def anchor_fold(self) -> SeasonFold:
        """Return the recentness anchor through an explicit, separate API."""

        return self.primary_holdout

    def to_summary(self) -> dict[str, object]:
        return {
            "primary_holdout": self.primary_holdout.to_summary(),
            "rolling_folds": [fold.to_summary() for fold in self.rolling_folds],
            "recent_only_diagnostic": (
                None
                if self.recent_only_diagnostic is None
                else self.recent_only_diagnostic.to_summary()
            ),
        }


@dataclass(frozen=True)
class DevelopmentValidationPlan:
    """Anchor-free validation plan for model development and selection.

    This deliberately exposes only the two rolling development folds.  It has
    no anchor accessor and no generic execution-fold method, so a development
    caller cannot accidentally discover or iterate over a 2024 partition.
    """

    development_folds: tuple[SeasonFold, ...]

    def __post_init__(self) -> None:
        folds = tuple(self.development_folds)
        object.__setattr__(self, "development_folds", folds)
        validation_seasons = tuple(fold.validation_season for fold in folds)
        if validation_seasons != DEVELOPMENT_VALIDATION_SEASONS:
            raise ValidationDesignError(
                "development plan must contain exactly the rolling 2022 and 2023 folds"
            )
        if any(fold.role != "rolling" for fold in folds):
            raise ValidationDesignError(
                "every development fold must carry role='rolling'"
            )
        names = tuple(fold.name for fold in folds)
        expected_names = tuple(
            f"rolling_origin_{season}" for season in DEVELOPMENT_VALIDATION_SEASONS
        )
        if names != expected_names:
            raise ValidationDesignError(
                "development fold names/order differ from the fixed contract"
            )
        if len({id(fold._source) for fold in folds}) != 1:
            raise ValidationDesignError(
                "development folds must share one immutable source-row contract"
            )
        if any(fold.inner_stop_split is None for fold in folds):
            raise ValidationDesignError(
                "every development fold must define a train-contained inner stop split"
            )

    def to_summary(self) -> dict[str, object]:
        return {
            "scope": "development_only_2019_2023",
            "development_folds": [fold.to_summary() for fold in self.development_folds],
        }


def _validated_source(
    frame: pd.DataFrame,
    *,
    allowed_extra_columns: Sequence[str] = (),
    require_expected_seasons: bool,
) -> _SourceRows:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].tolist()
        raise ValidationDesignError(f"duplicate dataframe columns: {duplicates}")
    try:
        validate_input_frame(
            frame,
            require_target=True,
            allow_target=True,
            allowed_extra_columns=allowed_extra_columns,
        )
    except SchemaError as error:
        raise ValidationDesignError(str(error)) from error

    raw_seasons = pd.to_numeric(frame["season"], errors="coerce")
    season_values = raw_seasons.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(season_values).all() or not np.equal(
        season_values, np.floor(season_values)
    ).all():
        raise ValidationDesignError("season must contain finite integer years")
    seasons = season_values.astype(np.int64)

    target = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValidationDesignError(f"{TARGET_COL} must contain only binary 0/1 values")

    if require_expected_seasons:
        observed = tuple(sorted(int(value) for value in np.unique(seasons)))
        if observed != EXPECTED_SEASONS:
            missing = sorted(set(EXPECTED_SEASONS).difference(observed))
            unexpected = sorted(set(observed).difference(EXPECTED_SEASONS))
            raise ValidationDesignError(
                "season coverage differs from the competition contract; "
                f"missing={missing}, unexpected={unexpected}"
            )

    return _SourceRows(
        row_ids=frame[ID_COL].to_numpy(dtype=object, copy=True),
        seasons=seasons,
    )


def _assert_development_source_scope(frame: pd.DataFrame) -> None:
    """Reject anchor rows before any target value is validated or consumed."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].tolist()
        raise ValidationDesignError(f"duplicate dataframe columns: {duplicates}")
    if "season" not in frame.columns:
        raise ValidationDesignError("frame is missing required column 'season'")
    raw_seasons = pd.to_numeric(frame["season"], errors="coerce")
    season_values = raw_seasons.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(season_values).all() or not np.equal(
        season_values, np.floor(season_values)
    ).all():
        raise ValidationDesignError("season must contain finite integer years")
    seasons = season_values.astype(np.int64)
    if np.equal(seasons, PRIMARY_VALIDATION_SEASON).any():
        raise ValidationDesignError(
            "development input contains the locked anchor season 2024"
        )
    observed = tuple(sorted(int(value) for value in np.unique(seasons)))
    if observed != DEVELOPMENT_SOURCE_SEASONS:
        missing = sorted(set(DEVELOPMENT_SOURCE_SEASONS).difference(observed))
        unexpected = sorted(set(observed).difference(DEVELOPMENT_SOURCE_SEASONS))
        raise ValidationDesignError(
            "development season coverage differs from the 2019-2023 contract; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _partition_from_source(
    source: _SourceRows,
    *,
    name: str,
    train_seasons: Iterable[int],
    validation_season: int,
) -> SeasonPartition:
    train_years = _normalise_seasons(train_seasons, name=f"{name}.train_seasons")
    validation_year = _normalise_seasons(
        (validation_season,), name=f"{name}.validation_season"
    )[0]
    train_positions = np.flatnonzero(np.isin(source.seasons, train_years))
    validation_positions = np.flatnonzero(source.seasons == validation_year)
    return SeasonPartition(
        name=name,
        train_seasons=train_years,
        validation_season=validation_year,
        train_positions=train_positions,
        validation_positions=validation_positions,
        _source=source,
    )


def _fold_from_source(
    source: _SourceRows,
    *,
    name: str,
    train_seasons: Iterable[int],
    validation_season: int,
    role: str,
    inner_validation_season: int | None,
) -> SeasonFold:
    train_years = _normalise_seasons(train_seasons, name=f"{name}.train_seasons")
    inner: SeasonPartition | None = None
    if inner_validation_season is not None:
        inner_validation_year = _normalise_seasons(
            (inner_validation_season,),
            name=f"{name}.inner_validation_season",
        )[0]
        inner_train_years = tuple(
            season for season in train_years if season < inner_validation_year
        )
        inner = _partition_from_source(
            source,
            name=f"{name}__inner_stop_{inner_validation_year}",
            train_seasons=inner_train_years,
            validation_season=inner_validation_year,
        )
    outer = _partition_from_source(
        source,
        name=name,
        train_seasons=train_years,
        validation_season=validation_season,
    )
    return SeasonFold(
        name=outer.name,
        train_seasons=outer.train_seasons,
        validation_season=outer.validation_season,
        train_positions=outer.train_positions,
        validation_positions=outer.validation_positions,
        _source=source,
        role=role,
        inner_stop_split=inner,
    )


def build_season_fold(
    frame: pd.DataFrame,
    *,
    name: str,
    train_seasons: Iterable[int],
    validation_season: int,
    role: str = "custom",
    inner_validation_season: int | None = None,
    allowed_extra_columns: Sequence[str] = (),
) -> SeasonFold:
    """Build one audited custom fold from an official labelled frame."""

    source = _validated_source(
        frame,
        allowed_extra_columns=allowed_extra_columns,
        require_expected_seasons=False,
    )
    return _fold_from_source(
        source,
        name=name,
        train_seasons=train_seasons,
        validation_season=validation_season,
        role=role,
        inner_validation_season=inner_validation_season,
    )


def build_development_plan(
    frame: pd.DataFrame,
    *,
    allowed_extra_columns: Sequence[str] = (),
) -> DevelopmentValidationPlan:
    """Build the anchor-free 2019--2023 model-development contract.

    The supplied frame must contain exactly seasons 2019 through 2023.  A
    single 2024 row is rejected before target validation, and the returned
    object exposes only rolling-origin 2022 and 2023 folds.
    """

    _assert_development_source_scope(frame)
    source = _validated_source(
        frame,
        allowed_extra_columns=allowed_extra_columns,
        require_expected_seasons=False,
    )
    folds = tuple(
        _fold_from_source(
            source,
            name=f"rolling_origin_{validation_season}",
            train_seasons=tuple(
                season
                for season in DEVELOPMENT_SOURCE_SEASONS
                if season < validation_season
            ),
            validation_season=validation_season,
            role="rolling",
            inner_validation_season=validation_season - 1,
        )
        for validation_season in DEVELOPMENT_VALIDATION_SEASONS
    )
    return DevelopmentValidationPlan(development_folds=folds)


def build_validation_plan(
    frame: pd.DataFrame,
    *,
    include_recent_only: bool = False,
    allowed_extra_columns: Sequence[str] = (),
) -> ValidationPlan:
    """Build the fixed competition plan and verify every temporal invariant.

    Rolling outer folds are 2019--2021 -> 2022, 2019--2022 -> 2023, and
    2019--2023 -> 2024.  Each has an inner stop split one season earlier.  The
    primary holdout is an alias of rolling 2024, not a duplicate fold run.
    """

    source = _validated_source(
        frame,
        allowed_extra_columns=allowed_extra_columns,
        require_expected_seasons=True,
    )
    rolling: list[SeasonFold] = []
    for validation_season in ROLLING_VALIDATION_SEASONS:
        train_seasons = tuple(
            season for season in EXPECTED_SEASONS if season < validation_season
        )
        role = "anchor" if validation_season == PRIMARY_VALIDATION_SEASON else "rolling"
        rolling.append(
            _fold_from_source(
                source,
                name=f"rolling_origin_{validation_season}",
                train_seasons=train_seasons,
                validation_season=validation_season,
                role=role,
                inner_validation_season=validation_season - 1,
            )
        )

    diagnostic: SeasonFold | None = None
    if include_recent_only:
        diagnostic = _fold_from_source(
            source,
            name="recent_only_2023_to_2024",
            train_seasons=(2023,),
            validation_season=2024,
            role="diagnostic",
            inner_validation_season=None,
        )

    folds = tuple(rolling)
    return ValidationPlan(
        rolling_folds=folds,
        primary_holdout=folds[-1],
        recent_only_diagnostic=diagnostic,
    )


def assert_selection_scope(fold: SeasonFold, row_positions: Iterable[int]) -> None:
    """Reject any epoch/checkpoint-selection scope containing outer validation.

    Selection rows may be a subset of the outer training rows (normally the
    complete inner train plus inner validation), but may never contain an outer
    holdout row or an unrelated source row.
    """

    positions = _readonly_int_array(row_positions, name="selection row positions")
    if not len(positions):
        raise ValidationDesignError("selection row positions must not be empty")
    if len(np.unique(positions)) != len(positions):
        raise ValidationDesignError("selection row positions contain duplicates")
    if (positions < 0).any() or (positions >= fold._source.row_count).any():
        raise ValidationDesignError("selection row positions are outside source bounds")
    if np.intersect1d(positions, fold.validation_positions).size:
        raise ValidationDesignError(
            f"{fold.name}: outer validation rows are reserved and cannot select epochs/checkpoints"
        )
    outside = np.setdiff1d(positions, fold.train_positions)
    if outside.size:
        raise ValidationDesignError(
            f"{fold.name}: selection rows must be contained in outer training rows"
        )


def assert_train_only_fit_scope(
    partition: SeasonPartition,
    fit_positions: Iterable[int],
    *,
    fit_row_ids: Sequence[object] | np.ndarray | None = None,
    require_all_training_rows: bool = True,
) -> None:
    """Audit externally recorded preprocessing-fit positions for leakage."""

    positions = _readonly_int_array(fit_positions, name="preprocessing fit positions")
    if not len(positions):
        raise ValidationDesignError("preprocessing fit positions must not be empty")
    if len(np.unique(positions)) != len(positions):
        raise ValidationDesignError("preprocessing fit positions contain duplicates")
    if (positions < 0).any() or (positions >= partition._source.row_count).any():
        raise ValidationDesignError("preprocessing fit positions are outside source bounds")
    if np.intersect1d(positions, partition.validation_positions).size:
        raise ValidationDesignError(
            f"{partition.name}: preprocessing fit includes validation rows"
        )
    outside = np.setdiff1d(positions, partition.train_positions)
    if outside.size:
        raise ValidationDesignError(
            f"{partition.name}: preprocessing fit includes non-training rows"
        )
    if require_all_training_rows and not np.array_equal(
        np.sort(positions), np.sort(partition.train_positions)
    ):
        raise ValidationDesignError(
            f"{partition.name}: preprocessing fit does not contain exactly all training rows"
        )
    if fit_row_ids is not None:
        identifiers = np.asarray(fit_row_ids, dtype=object).reshape(-1)
        if len(identifiers) != len(positions):
            raise ValidationDesignError("fit row ID count does not match fit position count")
        expected_ids = partition._source.row_ids[positions]
        if not np.array_equal(identifiers, expected_ids):
            raise ValidationDesignError(
                f"{partition.name}: preprocessing fit row IDs do not match source positions"
            )


@runtime_checkable
class FitTransformPreprocessor(Protocol):
    """Minimal stateful preprocessing interface accepted by fold helpers."""

    def fit(self, frame: pd.DataFrame) -> Any: ...

    def transform(self, frame: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class FitScopeAudit:
    """Immutable record proving which source rows were supplied to ``fit``."""

    partition: SeasonPartition
    fit_positions: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        positions = _readonly_int_array(self.fit_positions, name="fit_positions")
        object.__setattr__(self, "fit_positions", positions)
        assert_train_only_fit_scope(
            self.partition,
            positions,
            fit_row_ids=self.partition._source.row_ids[positions],
            require_all_training_rows=True,
        )

    @property
    def fit_row_ids(self) -> np.ndarray:
        return self.partition._source.row_ids_at(self.fit_positions)

    def assert_train_only(self) -> None:
        assert_train_only_fit_scope(
            self.partition,
            self.fit_positions,
            fit_row_ids=self.fit_row_ids,
            require_all_training_rows=True,
        )


@dataclass(frozen=True)
class FittedFoldPreprocessor:
    """Fitted preprocessor plus an auditable, train-only fit scope."""

    preprocessor: FitTransformPreprocessor
    audit: FitScopeAudit

    @property
    def partition(self) -> SeasonPartition:
        return self.audit.partition

    def transform_train(self, frame: pd.DataFrame) -> Any:
        return self.preprocessor.transform(self.partition.training_frame(frame))

    def transform_validation(self, frame: pd.DataFrame) -> Any:
        return self.preprocessor.transform(self.partition.validation_frame(frame))


def _fit_preprocessor_on_partition(
    preprocessor_factory: Callable[[], FitTransformPreprocessor],
    frame: pd.DataFrame,
    partition: SeasonPartition,
) -> FittedFoldPreprocessor:
    if not callable(preprocessor_factory):
        raise TypeError("preprocessor_factory must be a zero-argument callable")
    partition._source.assert_matches(frame)
    preprocessor = preprocessor_factory()
    if not isinstance(preprocessor, FitTransformPreprocessor):
        raise TypeError("preprocessor_factory must create an object with fit and transform")
    train = partition.training_frame(frame)
    audit = FitScopeAudit(partition=partition, fit_positions=partition.train_positions)
    fitted_result = preprocessor.fit(train)
    fitted = preprocessor if fitted_result is None else fitted_result
    if not isinstance(fitted, FitTransformPreprocessor):
        raise TypeError("fitted preprocessor must provide fit and transform methods")
    audit.assert_train_only()
    return FittedFoldPreprocessor(preprocessor=fitted, audit=audit)


def fit_preprocessor_on_fold_train(
    preprocessor_factory: Callable[[], FitTransformPreprocessor],
    frame: pd.DataFrame,
    fold: SeasonFold,
) -> FittedFoldPreprocessor:
    """Fit a fresh preprocessing pipeline on the outer training rows only."""

    return _fit_preprocessor_on_partition(preprocessor_factory, frame, fold)


def fit_preprocessor_on_inner_train(
    preprocessor_factory: Callable[[], FitTransformPreprocessor],
    frame: pd.DataFrame,
    fold: SeasonFold,
) -> FittedFoldPreprocessor:
    """Fit preprocessing for selection using inner training rows only.

    The returned object's validation transform is the inner stop season.  The
    outer validation partition is intentionally inaccessible from this helper.
    """

    if fold.inner_stop_split is None:
        raise ValidationDesignError(
            f"{fold.name}: no inner split; this diagnostic fold cannot select epochs"
        )
    assert_selection_scope(
        fold,
        np.concatenate(
            [
                fold.inner_stop_split.train_positions,
                fold.inner_stop_split.validation_positions,
            ]
        ),
    )
    return _fit_preprocessor_on_partition(
        preprocessor_factory, frame, fold.inner_stop_split
    )


__all__ = [
    "EXPECTED_SEASONS",
    "ROLLING_VALIDATION_SEASONS",
    "PRIMARY_VALIDATION_SEASON",
    "DEVELOPMENT_SOURCE_SEASONS",
    "DEVELOPMENT_VALIDATION_SEASONS",
    "ValidationDesignError",
    "SeasonPartition",
    "SeasonFold",
    "ValidationPlan",
    "DevelopmentValidationPlan",
    "FitTransformPreprocessor",
    "FitScopeAudit",
    "FittedFoldPreprocessor",
    "build_season_fold",
    "build_validation_plan",
    "build_development_plan",
    "assert_selection_scope",
    "assert_train_only_fit_scope",
    "fit_preprocessor_on_fold_train",
    "fit_preprocessor_on_inner_train",
]
