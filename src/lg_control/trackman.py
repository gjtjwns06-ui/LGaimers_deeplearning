"""Leakage-safe, row-local context features from historical Trackman logs.

The two datasets deliberately use unrelated entity identifiers.  This module
therefore uses only pre-pitch game context shared by both datasets and never
reads player, team, or handedness identifiers.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


ROW_ID_COLUMN = "row_id"
METRIC_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
PITCH_TYPE_GROUPS = ("fastball", "breaking", "offspeed", "other")
CONTEXT_COLUMNS = (
    "top_bottom",
    "game_month",
    "inning_bucket",
    "balls_before",
    "strikes_before",
    "outs_before",
)
HISTORY_REQUIRED_COLUMNS = (
    "trackman_id",
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitch_type_group",
    *METRIC_COLUMNS,
)
MAIN_REQUIRED_COLUMNS = (
    ROW_ID_COLUMN,
    "season",
    "game_month",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
)

TRACKMAN_FEATURE_COLUMNS = (
    "tm_history_count",
    *(
        name
        for metric in METRIC_COLUMNS
        for name in (
            f"tm_{metric}_count",
            f"tm_{metric}_mean",
            f"tm_{metric}_std",
        )
    ),
    "tm_pitch_type_count",
    *(f"tm_pitch_type_{group}_rate" for group in PITCH_TYPE_GROUPS),
)
JOINED_TRACKMAN_FEATURE_COLUMNS = (*TRACKMAN_FEATURE_COLUMNS, "tm_context_found")
LOOKUP_COLUMNS = (
    "lookup_level",
    "target_season",
    *CONTEXT_COLUMNS,
    *TRACKMAN_FEATURE_COLUMNS,
)

_GLOBAL_TEXT_SENTINEL = "*"
_GLOBAL_INT_SENTINEL = -1
_DATE_FORMATS = (
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%Y-%m-%d %H:%M:%S",
)


class TrackmanDataError(ValueError):
    """Raised when Trackman history or a lookup violates its contract."""


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


def _parse_mixed_game_dates(values: pd.Series) -> pd.Series:
    """Parse supported date layouts deterministically, without locale guessing."""

    raw = values.astype("string").str.strip()
    unique = pd.Series(raw.dropna().unique(), dtype="string")
    parsed_unique = pd.Series(pd.NaT, index=unique.index, dtype="datetime64[ns]")
    for date_format in _DATE_FORMATS:
        unresolved = parsed_unique.isna()
        if not unresolved.any():
            break
        parsed_unique.loc[unresolved] = pd.to_datetime(
            unique.loc[unresolved], format=date_format, errors="coerce"
        )
    bad = unique.loc[parsed_unique.isna()].head(5).tolist()
    if raw.isna().any() or bad:
        examples = bad or ["<missing>"]
        raise TrackmanDataError(f"game_date contains unparseable values: {examples}")
    mapping = dict(zip(unique.astype(str), parsed_unique, strict=True))
    parsed = raw.astype(str).map(mapping)
    if parsed.isna().any():
        raise TrackmanDataError("game_date parsing produced unexpected missing values")
    return pd.to_datetime(parsed).dt.normalize()


def _integer_values(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=np.float64)), index=values.index)
    integral = finite & numeric.eq(np.floor(numeric))
    return numeric, integral


def _normalise_top_bottom(values: pd.Series, *, strict: bool) -> pd.Series:
    tokens = values.astype("string").str.strip().str.upper()
    result = tokens.map({"T": "T", "TOP": "T", "B": "B", "BOTTOM": "B"}).astype("string")
    if strict and result.isna().any():
        examples = values.loc[result.isna()].drop_duplicates().head(5).tolist()
        raise TrackmanDataError(f"unsupported top_bottom values: {examples}")
    return result


def _normalise_main_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    season, season_ok = _integer_values(frame["season"])
    result["target_season"] = season.where(season_ok).astype("Int64")
    result["top_bottom"] = _normalise_top_bottom(frame["top_bottom"], strict=False)

    for column in ("game_month", "balls_before", "strikes_before", "outs_before"):
        numeric, valid = _integer_values(frame[column])
        result[column] = numeric.where(valid).astype("Int64")
    inning, inning_ok = _integer_values(frame["inning"])
    result["inning_bucket"] = inning.where(inning_ok).clip(upper=10).astype("Int64")
    return result[["target_season", *CONTEXT_COLUMNS]]


def prepare_trackman_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate, de-duplicate, sort, and normalise raw history.

    Invalid pre-pitch count rows are removed.  Invalid dates, seasons, months,
    identifiers, or half-inning labels fail loudly because silently accepting
    them could invalidate a temporal cutoff.
    """

    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame")
    missing = _missing_columns(history, HISTORY_REQUIRED_COLUMNS)
    if missing:
        raise TrackmanDataError(f"history is missing required columns: {missing}")
    if len(history) == 0:
        raise TrackmanDataError("history must not be empty")

    # Only columns permitted by the context design are retained.  In
    # particular, no entity or handedness columns can enter an aggregation.
    work = history.loc[:, HISTORY_REQUIRED_COLUMNS].copy()
    # ``trackman_id`` is a surrogate row identifier.  The official history
    # contains a small number of duplicated observations whose only difference
    # is that surrogate, so content identity intentionally excludes it.
    duplicate_columns = [column for column in HISTORY_REQUIRED_COLUMNS if column != "trackman_id"]
    duplicate_mask = work.duplicated(subset=duplicate_columns, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        work = work.loc[~duplicate_mask].copy()
    if work["trackman_id"].isna().any():
        raise TrackmanDataError("trackman_id contains missing values")
    if work["trackman_id"].duplicated().any():
        examples = work.loc[
            work["trackman_id"].duplicated(keep=False), "trackman_id"
        ].head(5).tolist()
        raise TrackmanDataError(
            "trackman_id has conflicting duplicate rows after exact duplicate removal: "
            f"{examples}"
        )

    work["_game_date"] = _parse_mixed_game_dates(work["game_date"])
    season, season_ok = _integer_values(work["season"])
    if not season_ok.all():
        raise TrackmanDataError("season must contain finite integer years")
    work["source_season"] = season.astype(np.int64)
    date_year = work["_game_date"].dt.year.astype(np.int64)
    if not date_year.eq(work["source_season"]).all():
        examples = work.loc[
            ~date_year.eq(work["source_season"]), ["trackman_id", "season", "game_date"]
        ].head(5).to_dict("records")
        raise TrackmanDataError(f"game_date year does not match season: {examples}")

    month, month_ok = _integer_values(work["game_month"])
    if not (month_ok & month.between(1, 12)).all():
        raise TrackmanDataError("game_month must contain integers in [1, 12]")
    work["game_month"] = month.astype(np.int64)
    date_month = work["_game_date"].dt.month.astype(np.int64)
    if not date_month.eq(work["game_month"]).all():
        examples = work.loc[
            ~date_month.eq(work["game_month"]),
            ["trackman_id", "game_date", "game_month"],
        ].head(5).to_dict("records")
        raise TrackmanDataError(f"game_date month does not match game_month: {examples}")
    dayofweek, dayofweek_ok = _integer_values(work["game_dayofweek"])
    expected_dayofweek = work["_game_date"].dt.dayofweek.astype(np.int64)
    if not (dayofweek_ok & dayofweek.between(0, 6)).all():
        raise TrackmanDataError("game_dayofweek must contain integers in [0, 6]")
    if not dayofweek.astype(np.int64).eq(expected_dayofweek).all():
        examples = work.loc[
            ~dayofweek.astype(np.int64).eq(expected_dayofweek),
            ["trackman_id", "game_date", "game_dayofweek"],
        ].head(5).to_dict("records")
        raise TrackmanDataError(f"game_date weekday does not match game_dayofweek: {examples}")

    work["top_bottom"] = _normalise_top_bottom(work["top_bottom"], strict=True)
    valid_context = pd.Series(True, index=work.index)
    bounds = {
        "inning": (1, None),
        "balls_before": (0, 3),
        "strikes_before": (0, 2),
        "outs_before": (0, 2),
    }
    for column, (lower, upper) in bounds.items():
        numeric, integral = _integer_values(work[column])
        valid = integral & numeric.ge(lower)
        if upper is not None:
            valid &= numeric.le(upper)
        valid_context &= valid
        work[column] = numeric
    invalid_context_count = int((~valid_context).sum())
    work = work.loc[valid_context].copy()
    if len(work) == 0:
        raise TrackmanDataError("no rows remain after invalid count filtering")
    for column in bounds:
        work[column] = work[column].astype(np.int64)
    work["inning_bucket"] = work["inning"].clip(upper=10).astype(np.int64)

    work["_sort_game_id"] = work["trackman_game_id"].astype("string").fillna("")
    work["_sort_pitch_no"] = pd.to_numeric(work["pitch_no"], errors="coerce")
    work["_sort_trackman_id"] = work["trackman_id"].astype("string")
    work = work.sort_values(
        ["_game_date", "_sort_game_id", "_sort_pitch_no", "_sort_trackman_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    if not work["_game_date"].is_monotonic_increasing:
        raise RuntimeError("internal Trackman chronological sort verification failed")
    pitch_order = work.groupby(
        ["_game_date", "_sort_game_id"], sort=False, dropna=False
    )["_sort_pitch_no"].diff()
    if (pitch_order.dropna() < 0).any():
        raise RuntimeError("internal Trackman pitch-order sort verification failed")

    for metric in METRIC_COLUMNS:
        numeric = pd.to_numeric(work[metric], errors="coerce")
        work[metric] = numeric.where(np.isfinite(numeric), np.nan).astype(np.float64)
    pitch_type = work["pitch_type_group"].astype("string").str.strip().str.lower()
    known = pitch_type.isin(PITCH_TYPE_GROUPS)
    unknown_pitch_types = int((~known).sum())
    work["pitch_type_group"] = pitch_type.where(known, "other")

    keep = [
        "source_season",
        *CONTEXT_COLUMNS,
        "pitch_type_group",
        *METRIC_COLUMNS,
    ]
    result = work.loc[:, keep].copy()
    report: dict[str, object] = {
        "input_rows": int(len(history)),
        "exact_duplicates_removed": duplicate_count,
        "invalid_context_rows_removed": invalid_context_count,
        "unknown_pitch_types_mapped_to_other": unknown_pitch_types,
        "retained_rows": int(len(result)),
        "source_seasons": sorted(int(value) for value in result["source_season"].unique()),
        "date_min": work["_game_date"].min().date().isoformat(),
        "date_max": work["_game_date"].max().date().isoformat(),
        "chronologically_sorted": True,
    }
    return result, report


def _source_season_sufficient_statistics(history: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["source_season", *CONTEXT_COLUMNS]
    grouped = history.groupby(group_columns, sort=True, observed=True, dropna=False)
    sufficient = grouped.size().rename("_history_count").to_frame()

    metric_stats = grouped[list(METRIC_COLUMNS)].agg(["count", "mean", "std"])
    for metric in METRIC_COLUMNS:
        count = metric_stats[(metric, "count")].astype(np.int64)
        mean = metric_stats[(metric, "mean")].fillna(0.0).astype(np.float64)
        std = metric_stats[(metric, "std")].fillna(0.0).astype(np.float64)
        sufficient[f"_{metric}_count"] = count
        sufficient[f"_{metric}_sum"] = count * mean
        sufficient[f"_{metric}_sumsq"] = (
            count * mean.pow(2) + (count - 1).clip(lower=0) * std.pow(2)
        )

    pitch_counts = (
        history.groupby(
            [*group_columns, "pitch_type_group"], sort=True, observed=True, dropna=False
        )
        .size()
        .unstack("pitch_type_group", fill_value=0)
    )
    for group in PITCH_TYPE_GROUPS:
        values = pitch_counts[group] if group in pitch_counts else 0
        sufficient[f"_pitch_{group}_count"] = values
    sufficient = sufficient.reset_index()
    count_columns = [
        "_history_count",
        *(f"_{metric}_count" for metric in METRIC_COLUMNS),
        *(f"_pitch_{group}_count" for group in PITCH_TYPE_GROUPS),
    ]
    sufficient[count_columns] = sufficient[count_columns].fillna(0).astype(np.int64)
    return sufficient


def _sufficient_columns() -> list[str]:
    columns = ["_history_count"]
    for metric in METRIC_COLUMNS:
        columns.extend(
            [f"_{metric}_count", f"_{metric}_sum", f"_{metric}_sumsq"]
        )
    columns.extend(f"_pitch_{group}_count" for group in PITCH_TYPE_GROUPS)
    return columns


def _finalise_features(
    sufficient: pd.DataFrame,
    *,
    fallback: pd.Series | None,
) -> pd.DataFrame:
    result = pd.DataFrame(index=sufficient.index)
    result["tm_history_count"] = sufficient["_history_count"].round().astype(np.int64)
    for metric in METRIC_COLUMNS:
        count = sufficient[f"_{metric}_count"].to_numpy(dtype=np.float64)
        total = sufficient[f"_{metric}_sum"].to_numpy(dtype=np.float64)
        sumsq = sufficient[f"_{metric}_sumsq"].to_numpy(dtype=np.float64)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        correction = np.divide(
            total * total,
            count,
            out=np.zeros_like(total),
            where=count > 0,
        )
        variance = np.divide(
            np.maximum(sumsq - correction, 0.0),
            count - 1.0,
            out=np.full_like(total, np.nan),
            where=count > 1,
        )
        std = np.sqrt(variance)
        mean_fallback = 0.0 if fallback is None else float(fallback[f"tm_{metric}_mean"])
        std_fallback = 0.0 if fallback is None else float(fallback[f"tm_{metric}_std"])
        result[f"tm_{metric}_count"] = count.round().astype(np.int64)
        result[f"tm_{metric}_mean"] = np.nan_to_num(mean, nan=mean_fallback)
        result[f"tm_{metric}_std"] = np.nan_to_num(std, nan=std_fallback)

    pitch_counts = np.column_stack(
        [sufficient[f"_pitch_{group}_count"].to_numpy(dtype=np.float64) for group in PITCH_TYPE_GROUPS]
    )
    pitch_total = pitch_counts.sum(axis=1)
    result["tm_pitch_type_count"] = pitch_total.round().astype(np.int64)
    for index, group in enumerate(PITCH_TYPE_GROUPS):
        rate = np.divide(
            pitch_counts[:, index],
            pitch_total,
            out=np.full_like(pitch_total, np.nan),
            where=pitch_total > 0,
        )
        rate_fallback = (
            0.0 if fallback is None else float(fallback[f"tm_pitch_type_{group}_rate"])
        )
        result[f"tm_pitch_type_{group}_rate"] = np.nan_to_num(rate, nan=rate_fallback)
    return result.loc[:, TRACKMAN_FEATURE_COLUMNS]


def _target_lookup(
    source_statistics: pd.DataFrame,
    target_season: int,
    min_group_count: int,
) -> pd.DataFrame:
    eligible = source_statistics.loc[source_statistics["source_season"] < target_season]
    if len(eligible) == 0:
        zero_features = {column: 0 for column in TRACKMAN_FEATURE_COLUMNS}
        return pd.DataFrame(
            [
                {
                    "lookup_level": "global",
                    "target_season": int(target_season),
                    "top_bottom": _GLOBAL_TEXT_SENTINEL,
                    **{
                        column: _GLOBAL_INT_SENTINEL
                        for column in CONTEXT_COLUMNS
                        if column != "top_bottom"
                    },
                    **zero_features,
                }
            ],
            columns=LOOKUP_COLUMNS,
        )
    if int(eligible["source_season"].max()) >= target_season:
        raise RuntimeError("temporal cutoff verification failed")

    sufficient_columns = _sufficient_columns()
    pooled = (
        eligible.groupby(list(CONTEXT_COLUMNS), sort=True, observed=True, dropna=False)[
            sufficient_columns
        ]
        .sum()
        .reset_index()
    )
    global_sufficient = pd.DataFrame(
        [{column: eligible[column].sum() for column in sufficient_columns}]
    )
    global_features = _finalise_features(global_sufficient, fallback=None)
    global_row = {
        "lookup_level": "global",
        "target_season": int(target_season),
        "top_bottom": _GLOBAL_TEXT_SENTINEL,
        **{column: _GLOBAL_INT_SENTINEL for column in CONTEXT_COLUMNS if column != "top_bottom"},
        **global_features.iloc[0].to_dict(),
    }

    pooled = pooled.loc[pooled["_history_count"] >= min_group_count].copy()
    context_features = _finalise_features(pooled, fallback=global_features.iloc[0])
    context = pooled.loc[:, CONTEXT_COLUMNS].reset_index(drop=True)
    context.insert(0, "target_season", int(target_season))
    context.insert(0, "lookup_level", "context")
    context = pd.concat([context, context_features.reset_index(drop=True)], axis=1)
    return pd.concat(
        [pd.DataFrame([global_row], columns=LOOKUP_COLUMNS), context.loc[:, LOOKUP_COLUMNS]],
        ignore_index=True,
    )


def build_trackman_lookup(
    history: pd.DataFrame,
    *,
    target_seasons: Iterable[int] = range(2019, 2026),
    min_group_count: int = 100,
) -> pd.DataFrame:
    """Build one context lookup per target season from strictly prior seasons."""

    if isinstance(min_group_count, bool) or int(min_group_count) != min_group_count:
        raise ValueError("min_group_count must be a positive integer")
    min_group_count = int(min_group_count)
    if min_group_count < 1:
        raise ValueError("min_group_count must be >= 1")
    seasons = list(target_seasons)
    if not seasons:
        raise ValueError("target_seasons must not be empty")
    if any(isinstance(value, bool) or int(value) != value for value in seasons):
        raise ValueError("target_seasons must contain integer years")
    seasons = sorted(set(int(value) for value in seasons))

    prepared, report = prepare_trackman_history(history)
    source_statistics = _source_season_sufficient_statistics(prepared)
    pieces = [
        _target_lookup(source_statistics, target_season, min_group_count)
        for target_season in seasons
    ]
    lookup = pd.concat(pieces, ignore_index=True)
    level_order = lookup["lookup_level"].map({"global": 0, "context": 1})
    lookup = (
        lookup.assign(_level_order=level_order)
        .sort_values(
            ["target_season", "_level_order", *CONTEXT_COLUMNS],
            kind="mergesort",
        )
        .drop(columns="_level_order")
        .reset_index(drop=True)
    )
    lookup = validate_trackman_lookup(lookup)
    report.update(
        {
            "target_seasons": seasons,
            "min_group_count": min_group_count,
            "source_season_context_rows": int(len(source_statistics)),
            "lookup_rows": int(len(lookup)),
            "context_lookup_rows": int(lookup["lookup_level"].eq("context").sum()),
        }
    )
    lookup.attrs["build_report"] = report
    return lookup


def validate_trackman_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised copy after strict lookup validation."""

    if not isinstance(lookup, pd.DataFrame):
        raise TypeError("lookup must be a pandas DataFrame")
    missing = _missing_columns(lookup, LOOKUP_COLUMNS)
    if missing:
        raise TrackmanDataError(f"lookup is missing required columns: {missing}")
    extras = sorted(set(lookup.columns).difference(LOOKUP_COLUMNS))
    if extras:
        raise TrackmanDataError(f"lookup has unexpected columns: {extras}")
    result = lookup.loc[:, LOOKUP_COLUMNS].copy()
    if len(result) == 0:
        raise TrackmanDataError("lookup must not be empty")
    if not result["lookup_level"].isin(["global", "context"]).all():
        raise TrackmanDataError("lookup_level must contain only 'global' or 'context'")

    target, target_ok = _integer_values(result["target_season"])
    if not target_ok.all():
        raise TrackmanDataError("target_season must contain finite integer years")
    result["target_season"] = target.astype(np.int64)
    global_rows = result["lookup_level"].eq("global")
    if not result.loc[global_rows].groupby("target_season").size().eq(1).all():
        raise TrackmanDataError("lookup must contain exactly one global row per represented season")
    if set(result.loc[global_rows, "target_season"]) != set(result["target_season"]):
        raise TrackmanDataError("every target season must have a global fallback row")

    contexts = result.loc[~global_rows].copy()
    contexts["top_bottom"] = _normalise_top_bottom(contexts["top_bottom"], strict=True)
    numeric_bounds = {
        "game_month": (1, 12),
        "inning_bucket": (1, 10),
        "balls_before": (0, 3),
        "strikes_before": (0, 2),
        "outs_before": (0, 2),
    }
    for column, (lower, upper) in numeric_bounds.items():
        numeric, valid = _integer_values(contexts[column])
        if not (valid & numeric.between(lower, upper)).all():
            raise TrackmanDataError(f"context lookup has invalid {column} values")
        contexts[column] = numeric.astype(np.int64)
    if contexts.duplicated(["target_season", *CONTEXT_COLUMNS]).any():
        raise TrackmanDataError("context lookup keys must be unique within target season")

    for column in TRACKMAN_FEATURE_COLUMNS:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise TrackmanDataError(f"lookup feature {column} contains non-finite values")
        if column.endswith("_count") or column.endswith("_std"):
            if (numeric < 0).any():
                raise TrackmanDataError(f"lookup feature {column} must be non-negative")
        if column.endswith("_count"):
            if not numeric.eq(np.floor(numeric)).all():
                raise TrackmanDataError(f"lookup feature {column} must contain integer counts")
            numeric = numeric.astype(np.int64)
        if column.endswith("_rate") and not numeric.between(0.0, 1.0).all():
            raise TrackmanDataError(f"lookup feature {column} must be in [0, 1]")
        result[column] = numeric
    pitch_rates = result[[f"tm_pitch_type_{group}_rate" for group in PITCH_TYPE_GROUPS]].sum(axis=1)
    nonempty_pitch = result["tm_pitch_type_count"] > 0
    if not np.allclose(pitch_rates.loc[nonempty_pitch], 1.0, rtol=1e-8, atol=1e-8):
        raise TrackmanDataError("pitch type rates must sum to one when history is available")

    # Put validated, canonical context values back without changing row order.
    result.loc[~global_rows, CONTEXT_COLUMNS] = contexts.loc[:, CONTEXT_COLUMNS]
    result.loc[global_rows, "top_bottom"] = _GLOBAL_TEXT_SENTINEL
    for column in CONTEXT_COLUMNS:
        if column != "top_bottom":
            result.loc[global_rows, column] = _GLOBAL_INT_SENTINEL
    return result.loc[:, LOOKUP_COLUMNS]


def load_trackman_lookup(path: str | Path) -> pd.DataFrame:
    """Load and validate a lookup CSV generated by :func:`build_trackman_lookup`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return validate_trackman_lookup(pd.read_csv(source))


def join_trackman_context(
    frame: pd.DataFrame,
    lookup: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Append Trackman context features with a row-local many-to-one lookup.

    Unseen contexts use the global prior for that row's target season.  No
    statistic is computed from ``frame`` and its row order is never semantic.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = _missing_columns(frame, MAIN_REQUIRED_COLUMNS)
    if missing:
        raise TrackmanDataError(f"main frame is missing required columns: {missing}")
    collisions = sorted(set(JOINED_TRACKMAN_FEATURE_COLUMNS).intersection(frame.columns))
    if collisions:
        raise TrackmanDataError(f"main frame already contains Trackman features: {collisions}")
    if frame[ROW_ID_COLUMN].isna().any() or frame[ROW_ID_COLUMN].duplicated().any():
        raise TrackmanDataError("row_id must be non-missing and unique")
    lookup_frame = (
        load_trackman_lookup(lookup)
        if isinstance(lookup, (str, Path))
        else validate_trackman_lookup(lookup)
    )

    original_ids = frame[ROW_ID_COLUMN].to_numpy(copy=True)
    keys = _normalise_main_keys(frame).reset_index(drop=True)
    keys.insert(0, "_tm_row_order", np.arange(len(frame), dtype=np.int64))
    contexts = lookup_frame.loc[
        lookup_frame["lookup_level"].eq("context"),
        ["target_season", *CONTEXT_COLUMNS, *TRACKMAN_FEATURE_COLUMNS],
    ].copy()
    globals_ = lookup_frame.loc[
        lookup_frame["lookup_level"].eq("global"),
        ["target_season", *TRACKMAN_FEATURE_COLUMNS],
    ].copy()
    contexts["target_season"] = contexts["target_season"].astype("Int64")
    globals_["target_season"] = globals_["target_season"].astype("Int64")
    for column in CONTEXT_COLUMNS:
        if column != "top_bottom":
            contexts[column] = contexts[column].astype("Int64")

    requested_seasons = set(keys["target_season"].dropna().astype(int))
    available_seasons = set(globals_["target_season"].astype(int))
    missing_seasons = sorted(requested_seasons.difference(available_seasons))
    if keys["target_season"].isna().any() or missing_seasons:
        raise TrackmanDataError(
            "main frame contains invalid or uncovered target seasons: "
            f"{missing_seasons if missing_seasons else ['<invalid>']}"
        )

    merged = keys.merge(
        contexts,
        on=["target_season", *CONTEXT_COLUMNS],
        how="left",
        sort=False,
        validate="many_to_one",
        indicator="_tm_context_merge",
    )
    merged["_tm_context_found"] = merged["_tm_context_merge"].eq("both")
    global_renames = {column: f"_global_{column}" for column in TRACKMAN_FEATURE_COLUMNS}
    merged = merged.merge(
        globals_.rename(columns=global_renames),
        on="target_season",
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_tm_row_order", kind="mergesort")
    if len(merged) != len(frame):
        raise AssertionError("Trackman many-to-one join changed the row count")
    if not np.array_equal(merged["_tm_row_order"].to_numpy(), np.arange(len(frame))):
        raise AssertionError("Trackman many-to-one join changed row order")
    context_found = merged["_tm_context_found"].to_numpy(dtype=bool)

    result = frame.copy()
    for column in TRACKMAN_FEATURE_COLUMNS:
        values = merged[column].where(merged[column].notna(), merged[f"_global_{column}"])
        if values.isna().any():
            raise TrackmanDataError(f"global fallback is unavailable for {column}")
        dtype = np.int64 if column.endswith("_count") else np.float64
        result[column] = values.to_numpy(dtype=dtype)
    result["tm_context_found"] = context_found.astype(np.int8)
    if len(result) != len(frame):
        raise AssertionError("Trackman feature attachment changed the row count")
    if not np.array_equal(result[ROW_ID_COLUMN].to_numpy(), original_ids):
        raise AssertionError("Trackman feature attachment changed row_id values or order")
    return result


# A descriptive alias for callers that prefer "attach" terminology.
attach_trackman_context = join_trackman_context
