"""Strict, forward-only TrackMan identity mapping and entity lookups.

The implementation follows the high-confidence path documented in
``artifacts/target1200_signal_audit.md``:

1. segment main rows into games in source order;
2. stably sort TrackMan games by ``pitch_no`` and ``trackman_id``;
3. hash the complete sequence of five shared pre-pitch context fields;
4. retain only bidirectionally unique exact game matches, then reject reused
   game IDs and any row-level month/weekday mismatch;
5. build main-ID -> TrackMan-ID maps from *earlier-season* matched rows only,
   requiring top count >= 20 and purity >= 99.9%; and
6. aggregate physical and pitch-type history with ``source_season <
   target_season`` before creating a row-local lookup.

``transform`` performs only a many-to-one lookup using the current row's
season and anonymous main ID.  It never groups, sorts, or inspects peer
evaluation rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .trackman import METRIC_COLUMNS, PITCH_TYPE_GROUPS


STRICT_TRACKMAN_STATE_VERSION = 1
SEQUENCE_COLUMNS = (
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
)
MAIN_ALIGNMENT_REQUIRED_COLUMNS = (
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    *SEQUENCE_COLUMNS,
    "pitcher_id",
    "batter_id",
)
TRACKMAN_ALIGNMENT_REQUIRED_COLUMNS = (
    "trackman_id",
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    *SEQUENCE_COLUMNS,
    "pitcher_trackman_id",
    "batter_trackman_id",
    "pitch_type_group",
    *METRIC_COLUMNS,
)
_DATE_FORMATS = (
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%Y-%m-%d %H:%M:%S",
)


def _role_feature_columns(role: str) -> tuple[str, ...]:
    prefix = f"tm_{role}"
    return (
        f"{prefix}_id_mapped",
        f"{prefix}_id_oov",
        f"{prefix}_mapping_top_count",
        f"{prefix}_mapping_top_count_log1p",
        f"{prefix}_mapping_purity",
        f"{prefix}_mapping_last_season_gap",
        f"{prefix}_history_found",
        f"{prefix}_history_count",
        f"{prefix}_history_count_log1p",
        f"{prefix}_history_last_season_gap",
        *(
            feature
            for metric in METRIC_COLUMNS
            for feature in (
                f"{prefix}_{metric}_count_log1p",
                f"{prefix}_{metric}_mean",
                f"{prefix}_{metric}_std",
            )
        ),
        *(f"{prefix}_pitch_type_{group}_rate" for group in PITCH_TYPE_GROUPS),
    )


ROLE_FEATURE_COLUMNS = {
    role: _role_feature_columns(role) for role in ("pitcher", "batter")
}
STRICT_TRACKMAN_FEATURE_COLUMNS = (
    *ROLE_FEATURE_COLUMNS["pitcher"],
    *ROLE_FEATURE_COLUMNS["batter"],
)


class StrictTrackmanError(ValueError):
    """Raised when strict TrackMan mapping or temporal state is invalid."""


class TrackmanTemporalLeakageError(StrictTrackmanError):
    """Raised when a target/future-season TrackMan row enters an aggregate."""


@dataclass(frozen=True)
class StrictGameAlignment:
    """Exact game matches and aligned player-ID observations."""

    matches: pd.DataFrame
    aligned_rows: pd.DataFrame
    audit: dict[str, int]


def _canonical_tokens(values: pd.Series) -> pd.Series:
    tokens = values.astype("string").str.strip().mask(lambda value: value.eq(""))
    return tokens.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].tolist()
        raise StrictTrackmanError(f"{name} has duplicate columns: {duplicates}")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise StrictTrackmanError(f"{name} is missing required columns: {missing}")


def _numeric(values: pd.Series, *, name: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & result.isna()
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise StrictTrackmanError(f"{name} contains non-numeric values such as {examples}")
    return result.astype(np.float64)


def _integer(
    values: pd.Series,
    *,
    name: str,
    lower: int | None = None,
    upper: int | None = None,
    allow_invalid: bool = False,
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.eq(np.floor(numeric))
    if lower is not None:
        valid &= numeric.ge(lower)
    if upper is not None:
        valid &= numeric.le(upper)
    if not allow_invalid and not valid.all():
        positions = np.flatnonzero(~valid.to_numpy())[:5].tolist()
        raise StrictTrackmanError(f"{name} is invalid at row positions {positions}")
    return numeric.astype(np.float64), valid


def _normalise_half(values: pd.Series, *, allow_invalid: bool = False) -> tuple[pd.Series, pd.Series]:
    tokens = values.astype("string").str.strip().str.upper()
    result = tokens.map({"T": "T", "TOP": "T", "B": "B", "BOTTOM": "B"}).astype(
        "string"
    )
    valid = result.notna()
    if not allow_invalid and not valid.all():
        examples = values.loc[~valid].drop_duplicates().head(5).tolist()
        raise StrictTrackmanError(f"top_bottom contains unsupported values: {examples}")
    return result, valid


def _parse_dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    unique = pd.Series(raw.dropna().unique(), dtype="string")
    parsed = pd.Series(pd.NaT, index=unique.index, dtype="datetime64[ns]")
    for date_format in _DATE_FORMATS:
        unresolved = parsed.isna()
        if not unresolved.any():
            break
        parsed.loc[unresolved] = pd.to_datetime(
            unique.loc[unresolved], format=date_format, errors="coerce"
        )
    if raw.isna().any() or parsed.isna().any():
        examples = unique.loc[parsed.isna()].head(5).tolist() or ["<missing>"]
        raise StrictTrackmanError(f"game_date contains unparseable values: {examples}")
    mapping = dict(zip(unique.astype(str), parsed, strict=True))
    result = raw.astype(str).map(mapping)
    if result.isna().any():
        raise RuntimeError("date parser produced unexpected missing values")
    return pd.to_datetime(result).dt.normalize()


def _sequence_hash(
    inning: np.ndarray,
    half: np.ndarray,
    balls: np.ndarray,
    strikes: np.ndarray,
    outs: np.ndarray,
) -> str:
    half_code = np.where(half == "T", 0, 1)
    matrix = np.column_stack([inning, half_code, balls, strikes, outs]).astype(
        "<i2", copy=False
    )
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _prepare_main_games(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(main, MAIN_ALIGNMENT_REQUIRED_COLUMNS, name="main frame")
    if main["row_id"].isna().any() or main["row_id"].duplicated().any():
        raise StrictTrackmanError("main row_id must be non-missing and unique")
    if len(main) == 0:
        raise StrictTrackmanError("main frame must not be empty")
    work = main.loc[:, MAIN_ALIGNMENT_REQUIRED_COLUMNS].copy()
    work["_main_position"] = np.arange(len(work), dtype=np.int64)

    integer_specs = {
        "season": (1900, None),
        "game_month": (1, 12),
        "game_dayofweek": (0, 6),
        "inning": (1, None),
        "balls_before": (0, 3),
        "strikes_before": (0, 2),
        "outs_before": (0, 2),
    }
    for column, (lower, upper) in integer_specs.items():
        numeric, _valid = _integer(
            work[column], name=f"main.{column}", lower=lower, upper=upper
        )
        work[column] = numeric.astype(np.int64)
    work["top_bottom"], _valid = _normalise_half(work["top_bottom"])
    for column in ("pitcher_id", "batter_id"):
        work[column] = _canonical_tokens(work[column])
        if work[column].isna().any():
            raise StrictTrackmanError(f"main.{column} contains missing values")
    seasons = work["season"].to_numpy(dtype=np.int64)
    if (np.diff(seasons) < 0).any():
        raise StrictTrackmanError("main seasons must be non-decreasing in source order")

    start_state = work["inning"].eq(1) & work["top_bottom"].eq("T")
    previous_start_state = start_state.shift(fill_value=False)
    boundaries = start_state & ~previous_start_state
    boundaries.iloc[0] = True
    if not start_state.iloc[0]:
        raise StrictTrackmanError("the first main game must begin at inning=1, top_bottom=T")
    season_change = work["season"].ne(work["season"].shift())
    if (season_change & ~boundaries).iloc[1:].any():
        raise StrictTrackmanError("a main season changes without a new-game boundary")
    work["_main_game_index"] = boundaries.cumsum().astype(np.int64) - 1

    records: list[dict[str, Any]] = []
    for game_index, group in work.groupby("_main_game_index", sort=True, observed=True):
        context = [group[column].to_numpy() for column in SEQUENCE_COLUMNS]
        records.append(
            {
                "main_game_index": int(game_index),
                "season": int(group["season"].iloc[0]),
                "game_month": int(group["game_month"].iloc[0]),
                "game_dayofweek": int(group["game_dayofweek"].iloc[0]),
                "pitch_count": int(len(group)),
                "sequence_sha256": _sequence_hash(*context),
            }
        )
    return work, pd.DataFrame.from_records(records)


def _prepare_trackman_history(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _require_columns(history, TRACKMAN_ALIGNMENT_REQUIRED_COLUMNS, name="TrackMan history")
    if len(history) == 0:
        raise StrictTrackmanError("TrackMan history must not be empty")
    # Detect the two documented content duplicates against the caller's complete
    # schema, then immediately project to the strict contract.  Keeping unrelated
    # text columns through the 1.8M-row sort needlessly doubles peak memory.
    duplicate_columns = [column for column in history.columns if column != "trackman_id"]
    duplicate_mask = history.duplicated(subset=duplicate_columns, keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    work = history.loc[~duplicate_mask, TRACKMAN_ALIGNMENT_REQUIRED_COLUMNS].copy()
    if work["trackman_id"].isna().any() or work["trackman_id"].duplicated().any():
        raise StrictTrackmanError("trackman_id must be non-missing and unique after de-duplication")

    trackman_id, valid_trackman_id = _integer(
        work["trackman_id"], name="TrackMan.trackman_id", lower=0
    )
    pitch_no, valid_pitch_no = _integer(
        work["pitch_no"], name="TrackMan.pitch_no", lower=0
    )
    if not valid_trackman_id.all() or not valid_pitch_no.all():
        raise StrictTrackmanError("TrackMan identifiers and pitch numbers must be integral")
    work["_trackman_id_int"] = trackman_id.astype(np.int64)
    work["pitch_no"] = pitch_no.astype(np.int64)
    work["_game_id"] = _canonical_tokens(work["trackman_game_id"])
    if work["_game_id"].isna().any():
        raise StrictTrackmanError("trackman_game_id contains missing values")
    work["_game_date"] = _parse_dates(work["game_date"])

    for column, lower, upper in (
        ("season", 1900, None),
        ("game_month", 1, 12),
        ("game_dayofweek", 0, 6),
    ):
        numeric, valid = _integer(
            work[column], name=f"TrackMan.{column}", lower=lower, upper=upper
        )
        if not valid.all():
            raise StrictTrackmanError(f"TrackMan.{column} is invalid")
        work[column] = numeric.astype(np.int64)

    context_valid = pd.Series(True, index=work.index)
    for column, lower, upper in (
        ("inning", 1, None),
        ("balls_before", 0, 3),
        ("strikes_before", 0, 2),
        ("outs_before", 0, 2),
    ):
        numeric, valid = _integer(
            work[column],
            name=f"TrackMan.{column}",
            lower=lower,
            upper=upper,
            allow_invalid=True,
        )
        work[column] = numeric
        context_valid &= valid
    work["top_bottom"], half_valid = _normalise_half(
        work["top_bottom"], allow_invalid=True
    )
    context_valid &= half_valid
    invalid_context_rows_removed = int((~context_valid).sum())
    work = work.loc[context_valid].copy()
    for column in ("inning", "balls_before", "strikes_before", "outs_before"):
        work[column] = work[column].astype(np.int64)

    # Some supplied game IDs contain a row-local date rollover even though
    # ``pitch_no`` is one continuous, non-repeating sequence.  Those are data
    # quality observations, not sufficient evidence to split or discard a
    # game.  The later row-wise month/weekday comparison rejects any candidate
    # whose common calendar fields disagree with the main sequence.
    date_counts = work.groupby("_game_id", sort=False)["_game_date"].nunique()
    multi_date_ids = set(date_counts.index[date_counts.gt(1)].astype(str))
    multi_date_game_rows = int(work["_game_id"].isin(multi_date_ids).sum())

    expected_year = work["_game_date"].dt.year.to_numpy(dtype=np.int64)
    expected_month = work["_game_date"].dt.month.to_numpy(dtype=np.int64)
    expected_weekday = work["_game_date"].dt.dayofweek.to_numpy(dtype=np.int64)
    calendar_bad = (
        work["season"].to_numpy(dtype=np.int64) != expected_year
    ) | (work["game_month"].to_numpy(dtype=np.int64) != expected_month) | (
        work["game_dayofweek"].to_numpy(dtype=np.int64) != expected_weekday
    )
    bad_calendar_ids = set(work.loc[calendar_bad, "_game_id"].astype(str))
    calendar_mismatch_game_rows_removed = int(
        work["_game_id"].isin(bad_calendar_ids).sum()
    )
    work = work.loc[~work["_game_id"].isin(bad_calendar_ids)].copy()
    if len(work) == 0:
        raise StrictTrackmanError("no TrackMan rows remain after strict cleaning")

    for column in ("pitcher_trackman_id", "batter_trackman_id"):
        work[column] = _canonical_tokens(work[column])
    pitch_type = work["pitch_type_group"].astype("string").str.strip().str.lower()
    work["pitch_type_group"] = pitch_type.where(
        pitch_type.isin(PITCH_TYPE_GROUPS), "other"
    )
    for metric in METRIC_COLUMNS:
        values = _numeric(work[metric], name=f"TrackMan.{metric}")
        work[metric] = values.where(np.isfinite(values), np.nan)

    work["_source_season"] = work["season"].astype(np.int64)
    work = work.sort_values(
        ["_game_id", "pitch_no", "_trackman_id_int"], kind="mergesort"
    ).reset_index(drop=True)
    audit = {
        "input_rows": int(len(history)),
        "content_duplicate_rows_removed": duplicate_rows_removed,
        "invalid_context_rows_removed": invalid_context_rows_removed,
        "multi_date_game_ids_observed": int(len(multi_date_ids)),
        "multi_date_game_rows_observed": multi_date_game_rows,
        "calendar_mismatch_game_ids_removed": int(len(bad_calendar_ids)),
        "calendar_mismatch_game_rows_removed": calendar_mismatch_game_rows_removed,
        "retained_rows": int(len(work)),
    }
    return work, audit


def _trackman_game_summaries(work: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for game_id, group in work.groupby("_game_id", sort=True, observed=True):
        context = [group[column].to_numpy() for column in SEQUENCE_COLUMNS]
        records.append(
            {
                "trackman_game_id": str(game_id),
                "season": int(group["season"].iloc[0]),
                "game_month": int(group["game_month"].iloc[0]),
                "game_dayofweek": int(group["game_dayofweek"].iloc[0]),
                "pitch_count": int(len(group)),
                "sequence_sha256": _sequence_hash(*context),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_alignment_from_prepared(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    trackman_audit: dict[str, int],
) -> StrictGameAlignment:
    main_rows, main_games = _prepare_main_games(main)
    trackman_games = _trackman_game_summaries(trackman)
    key_columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "pitch_count",
        "sequence_sha256",
    ]
    main_key_count = main_games.groupby(key_columns, observed=True).size().rename(
        "_main_key_count"
    )
    trackman_key_count = trackman_games.groupby(key_columns, observed=True).size().rename(
        "_trackman_key_count"
    )
    main_candidates = main_games.join(main_key_count, on=key_columns)
    trackman_candidates = trackman_games.join(trackman_key_count, on=key_columns)
    candidate_pairs = main_candidates.merge(
        trackman_candidates,
        on=key_columns,
        how="inner",
        validate="many_to_many",
        suffixes=("_main", "_trackman"),
    )
    raw_candidate_pair_count = int(len(candidate_pairs))
    reused_trackman_mask = candidate_pairs["trackman_game_id"].duplicated(
        keep=False
    )
    reused_main_mask = candidate_pairs["main_game_index"].duplicated(keep=False)
    reused_trackman_games = int(
        candidate_pairs.loc[reused_trackman_mask, "trackman_game_id"].nunique()
    )
    reused_main_games = int(
        candidate_pairs.loc[reused_main_mask, "main_game_index"].nunique()
    )
    non_one_to_one_pairs = int((reused_trackman_mask | reused_main_mask).sum())
    candidate_pairs = candidate_pairs.loc[
        candidate_pairs["_main_key_count"].eq(1)
        & candidate_pairs["_trackman_key_count"].eq(1)
    ].copy()

    match_records: list[dict[str, Any]] = []
    aligned_pieces: list[pd.DataFrame] = []
    month_mismatch_rows = 0
    weekday_mismatch_rows = 0
    calendar_mismatch_games = 0
    context_hash_collision_games = 0
    # ``DataFrame.loc[column == key]`` inside this loop is quadratic at full
    # scale (about 2,700 candidate games over 1.5M+ rows).  Group indices make
    # every extraction proportional only to the matched game's pitch count.
    main_group_indices = main_rows.groupby(
        "_main_game_index", sort=False, observed=True
    ).indices
    trackman_group_indices = trackman.groupby(
        "_game_id", sort=False, observed=True
    ).indices
    for row in candidate_pairs.itertuples(index=False):
        main_group = main_rows.iloc[main_group_indices[int(row.main_game_index)]]
        trackman_group = trackman.iloc[
            trackman_group_indices[str(row.trackman_game_id)]
        ]
        main_context = main_group.loc[:, SEQUENCE_COLUMNS].astype(
            {"top_bottom": "string"}
        )
        trackman_context = trackman_group.loc[:, SEQUENCE_COLUMNS].astype(
            {"top_bottom": "string"}
        )
        if not np.array_equal(
            main_context.to_numpy(dtype=object),
            trackman_context.to_numpy(dtype=object),
        ):
            context_hash_collision_games += 1
            continue
        month_equal_array = main_group["game_month"].to_numpy(
            dtype=np.int64
        ) == trackman_group["game_month"].to_numpy(dtype=np.int64)
        weekday_equal_array = main_group["game_dayofweek"].to_numpy(
            dtype=np.int64
        ) == trackman_group["game_dayofweek"].to_numpy(dtype=np.int64)
        month_mismatch_rows += int((~month_equal_array).sum())
        weekday_mismatch_rows += int((~weekday_equal_array).sum())
        if not month_equal_array.all() or not weekday_equal_array.all():
            calendar_mismatch_games += 1
            continue

        match_records.append(
            {
                "main_game_index": int(row.main_game_index),
                "trackman_game_id": str(row.trackman_game_id),
                "season": int(row.season),
                "game_month": int(row.game_month),
                "game_dayofweek": int(row.game_dayofweek),
                "pitch_count": int(row.pitch_count),
                "sequence_sha256": str(row.sequence_sha256),
            }
        )
        aligned_pieces.append(
            pd.DataFrame(
                {
                    "main_row_id": main_group["row_id"].to_numpy(dtype=object),
                    "main_position": main_group["_main_position"].to_numpy(
                        dtype=np.int64
                    ),
                    "main_season": main_group["season"].to_numpy(dtype=np.int64),
                    "trackman_season": trackman_group["_source_season"].to_numpy(
                        dtype=np.int64
                    ),
                    "main_pitcher_id": main_group["pitcher_id"].astype(str).to_numpy(),
                    "trackman_pitcher_id": trackman_group[
                        "pitcher_trackman_id"
                    ].astype("string").to_numpy(dtype=object),
                    "main_batter_id": main_group["batter_id"].astype(str).to_numpy(),
                    "trackman_batter_id": trackman_group[
                        "batter_trackman_id"
                    ].astype("string").to_numpy(dtype=object),
                    "trackman_game_id": np.repeat(
                        str(row.trackman_game_id), len(main_group)
                    ),
                }
            )
        )

    matches = pd.DataFrame.from_records(
        match_records,
        columns=[
            "main_game_index",
            "trackman_game_id",
            "season",
            "game_month",
            "game_dayofweek",
            "pitch_count",
            "sequence_sha256",
        ],
    )
    aligned = (
        pd.concat(aligned_pieces, ignore_index=True)
        if aligned_pieces
        else pd.DataFrame(
            columns=[
                "main_row_id",
                "main_position",
                "main_season",
                "trackman_season",
                "main_pitcher_id",
                "trackman_pitcher_id",
                "main_batter_id",
                "trackman_batter_id",
                "trackman_game_id",
            ]
        )
    )
    if matches["main_game_index"].duplicated().any() or matches[
        "trackman_game_id"
    ].duplicated().any():
        raise RuntimeError("strict bidirectional game matching produced a reused game")
    audit = {
        **trackman_audit,
        "main_games": int(len(main_games)),
        "trackman_games": int(len(trackman_games)),
        "raw_exact_candidate_pairs": raw_candidate_pair_count,
        "reused_trackman_candidate_games_rejected": reused_trackman_games,
        "reused_main_candidate_games_rejected": reused_main_games,
        "non_one_to_one_candidate_pairs_rejected": non_one_to_one_pairs,
        "bidirectional_unique_candidate_games": int(len(candidate_pairs)),
        "strict_matched_games": int(len(matches)),
        "strict_aligned_rows": int(len(aligned)),
        "row_month_mismatches_rejected": int(month_mismatch_rows),
        "row_weekday_mismatches_rejected": int(weekday_mismatch_rows),
        "row_calendar_mismatch_games_rejected": int(calendar_mismatch_games),
        "sequence_hash_collision_games_rejected": int(context_hash_collision_games),
    }
    return StrictGameAlignment(matches=matches, aligned_rows=aligned, audit=audit)


def build_strict_game_alignment(
    main: pd.DataFrame, history: pd.DataFrame
) -> StrictGameAlignment:
    """Return only exact, bidirectionally unique, row-verified game matches."""

    prepared, audit = _prepare_trackman_history(history)
    return _build_alignment_from_prepared(main, prepared, audit)


def build_high_purity_id_map(
    aligned_rows: pd.DataFrame,
    *,
    target_season: int,
    role: str,
    min_top_count: int = 20,
    min_purity: float = 0.999,
) -> pd.DataFrame:
    """Build an earlier-season anonymous-ID map with strict vote thresholds."""

    if role not in {"pitcher", "batter"}:
        raise StrictTrackmanError("role must be 'pitcher' or 'batter'")
    if isinstance(target_season, bool) or int(target_season) != target_season:
        raise StrictTrackmanError("target_season must be an integer")
    target_season = int(target_season)
    if isinstance(min_top_count, bool) or int(min_top_count) != min_top_count:
        raise StrictTrackmanError("min_top_count must be an integer")
    min_top_count = int(min_top_count)
    if min_top_count < 1:
        raise StrictTrackmanError("min_top_count must be >= 1")
    min_purity = float(min_purity)
    if not np.isfinite(min_purity) or not 0.0 < min_purity <= 1.0:
        raise StrictTrackmanError("min_purity must be inside (0, 1]")
    required = {
        "main_season",
        "trackman_season",
        f"main_{role}_id",
        f"trackman_{role}_id",
    }
    missing = sorted(required.difference(aligned_rows.columns))
    if missing:
        raise StrictTrackmanError(f"aligned rows are missing columns: {missing}")
    past = aligned_rows.loc[
        pd.to_numeric(aligned_rows["main_season"], errors="coerce").lt(target_season)
    ].copy()
    if len(past) == 0:
        return pd.DataFrame(
            columns=[
                "main_id",
                "trackman_id",
                "top_count",
                "total_count",
                "purity",
                "last_mapping_season",
            ]
        )
    if pd.to_numeric(past["trackman_season"], errors="coerce").ge(target_season).any():
        raise TrackmanTemporalLeakageError(
            "target/future-season TrackMan alignment entered the ID map"
        )
    main_column = f"main_{role}_id"
    trackman_column = f"trackman_{role}_id"
    past[main_column] = _canonical_tokens(past[main_column])
    past[trackman_column] = _canonical_tokens(past[trackman_column])
    past = past.dropna(subset=[main_column, trackman_column])
    votes = (
        past.groupby([main_column, trackman_column], sort=True, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = votes.groupby(main_column, observed=True)["count"].sum().rename("total_count")
    votes = votes.join(totals, on=main_column)
    votes = votes.sort_values(
        [main_column, "count", trackman_column],
        ascending=[True, False, True],
        kind="mergesort",
    )
    top = votes.groupby(main_column, sort=True, observed=True).head(1).copy()
    top["purity"] = top["count"] / top["total_count"]
    last_season = past.groupby(main_column, observed=True)["main_season"].max()
    top = top.join(last_season.rename("last_mapping_season"), on=main_column)
    top = top.loc[
        top["count"].ge(min_top_count) & top["purity"].ge(min_purity)
    ].copy()
    return top.rename(
        columns={main_column: "main_id", trackman_column: "trackman_id", "count": "top_count"}
    ).loc[
        :,
        [
            "main_id",
            "trackman_id",
            "top_count",
            "total_count",
            "purity",
            "last_mapping_season",
        ],
    ].reset_index(drop=True)


def _entity_season_sufficient(
    history: pd.DataFrame, *, entity_column: str
) -> pd.DataFrame:
    work = history.loc[
        history[entity_column].notna(),
        ["_source_season", entity_column, "pitch_type_group", *METRIC_COLUMNS],
    ].copy()
    work[entity_column] = _canonical_tokens(work[entity_column])
    work = work.dropna(subset=[entity_column])
    for metric in METRIC_COLUMNS:
        work[f"__{metric}_sq"] = work[metric].astype(np.float64).pow(2)
    for group in PITCH_TYPE_GROUPS:
        work[f"__pitch_{group}"] = work["pitch_type_group"].eq(group).astype(np.int64)

    named_aggregations: dict[str, tuple[str, str]] = {
        "history_count": (entity_column, "size"),
    }
    for metric in METRIC_COLUMNS:
        named_aggregations[f"{metric}_count"] = (metric, "count")
        named_aggregations[f"{metric}_sum"] = (metric, "sum")
        named_aggregations[f"{metric}_sumsq"] = (f"__{metric}_sq", "sum")
    for group in PITCH_TYPE_GROUPS:
        named_aggregations[f"pitch_{group}_count"] = (f"__pitch_{group}", "sum")
    return (
        work.groupby(["_source_season", entity_column], sort=True, observed=True)
        .agg(**named_aggregations)
        .reset_index()
        .rename(columns={entity_column: "trackman_id"})
    )


def _cumulative_entity_summary(
    sufficient: pd.DataFrame, *, target_season: int
) -> pd.DataFrame:
    eligible = sufficient.loc[sufficient["_source_season"].lt(target_season)].copy()
    if len(eligible) == 0:
        columns = [
            "trackman_id",
            "history_count",
            "history_last_season",
            *(
                feature
                for metric in METRIC_COLUMNS
                for feature in (
                    f"{metric}_count",
                    f"{metric}_mean",
                    f"{metric}_std",
                )
            ),
            *(f"pitch_type_{group}_rate" for group in PITCH_TYPE_GROUPS),
        ]
        return pd.DataFrame(columns=columns)
    if eligible["_source_season"].ge(target_season).any():
        raise TrackmanTemporalLeakageError(
            "target/future-season TrackMan sufficient statistic entered an aggregate"
        )
    sum_columns = [
        "history_count",
        *(
            feature
            for metric in METRIC_COLUMNS
            for feature in (
                f"{metric}_count",
                f"{metric}_sum",
                f"{metric}_sumsq",
            )
        ),
        *(f"pitch_{group}_count" for group in PITCH_TYPE_GROUPS),
    ]
    grouped = eligible.groupby("trackman_id", sort=True, observed=True)
    total = grouped[sum_columns].sum()
    total["history_last_season"] = grouped["_source_season"].max()
    result = pd.DataFrame(index=total.index)
    result["history_count"] = total["history_count"]
    result["history_last_season"] = total["history_last_season"]
    for metric in METRIC_COLUMNS:
        count = total[f"{metric}_count"].to_numpy(dtype=np.float64)
        values_sum = total[f"{metric}_sum"].to_numpy(dtype=np.float64)
        sumsq = total[f"{metric}_sumsq"].to_numpy(dtype=np.float64)
        mean = np.divide(
            values_sum, count, out=np.zeros_like(values_sum), where=count > 0.0
        )
        variance = np.divide(
            sumsq,
            count,
            out=np.zeros_like(sumsq),
            where=count > 0.0,
        ) - mean * mean
        result[f"{metric}_count"] = count
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = np.sqrt(np.maximum(variance, 0.0))
    pitch_total = np.column_stack(
        [
            total[f"pitch_{group}_count"].to_numpy(dtype=np.float64)
            for group in PITCH_TYPE_GROUPS
        ]
    ).sum(axis=1)
    for group in PITCH_TYPE_GROUPS:
        count = total[f"pitch_{group}_count"].to_numpy(dtype=np.float64)
        result[f"pitch_type_{group}_rate"] = np.divide(
            count, pitch_total, out=np.zeros_like(count), where=pitch_total > 0.0
        )
    return result.reset_index()


def summarize_strict_trackman_history(
    history: pd.DataFrame, *, target_season: int, entity_column: str
) -> pd.DataFrame:
    """Summarize an already time-restricted history, rejecting any leak.

    This public guard is useful for audits and tests: callers must pass a
    frame containing only seasons strictly earlier than ``target_season``.
    The high-level builder performs that filtering before invoking the same
    sufficient-statistic path.
    """

    if entity_column not in {"pitcher_trackman_id", "batter_trackman_id"}:
        raise StrictTrackmanError("unsupported TrackMan entity column")
    _require_columns(history, ["season", entity_column], name="TrackMan history")
    raw_season, valid = _integer(
        history["season"], name="TrackMan.season", lower=1900
    )
    if not valid.all():
        raise StrictTrackmanError("TrackMan season is invalid")
    if raw_season.ge(int(target_season)).any():
        raise TrackmanTemporalLeakageError(
            "TrackMan history must satisfy source_season < target_season"
        )
    prepared, _audit = _prepare_trackman_history(history)
    sufficient = _entity_season_sufficient(prepared, entity_column=entity_column)
    return _cumulative_entity_summary(sufficient, target_season=int(target_season))


def _build_role_lookup(
    *,
    role: str,
    target_season: int,
    id_map: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = ROLE_FEATURE_COLUMNS[role]
    if len(id_map) == 0:
        return pd.DataFrame(columns=["target_season", "main_id", *feature_columns])
    merged = id_map.merge(
        summary,
        left_on="trackman_id",
        right_on="trackman_id",
        how="left",
        validate="many_to_one",
    )
    prefix = f"tm_{role}"
    output = pd.DataFrame(
        {
            "target_season": np.full(len(merged), target_season, dtype=np.int64),
            "main_id": merged["main_id"].astype("string"),
            f"{prefix}_id_mapped": np.ones(len(merged), dtype=np.float64),
            f"{prefix}_id_oov": np.zeros(len(merged), dtype=np.float64),
            f"{prefix}_mapping_top_count": merged["top_count"].to_numpy(
                dtype=np.float64
            ),
            f"{prefix}_mapping_top_count_log1p": np.log1p(
                merged["top_count"].to_numpy(dtype=np.float64)
            ),
            f"{prefix}_mapping_purity": merged["purity"].to_numpy(dtype=np.float64),
            f"{prefix}_mapping_last_season_gap": target_season
            - merged["last_mapping_season"].to_numpy(dtype=np.float64),
        }
    )
    history_count = pd.to_numeric(merged.get("history_count"), errors="coerce").fillna(
        0.0
    ).to_numpy(dtype=np.float64)
    history_found = history_count > 0.0
    history_last = pd.to_numeric(
        merged.get("history_last_season"), errors="coerce"
    ).to_numpy(dtype=np.float64)
    output[f"{prefix}_history_found"] = history_found.astype(np.float64)
    output[f"{prefix}_history_count"] = history_count
    output[f"{prefix}_history_count_log1p"] = np.log1p(history_count)
    output[f"{prefix}_history_last_season_gap"] = np.where(
        history_found, target_season - history_last, 0.0
    )
    for metric in METRIC_COLUMNS:
        count = pd.to_numeric(merged.get(f"{metric}_count"), errors="coerce").fillna(
            0.0
        )
        mean = pd.to_numeric(merged.get(f"{metric}_mean"), errors="coerce").fillna(
            0.0
        )
        std = pd.to_numeric(merged.get(f"{metric}_std"), errors="coerce").fillna(0.0)
        output[f"{prefix}_{metric}_count_log1p"] = np.log1p(
            count.to_numpy(dtype=np.float64)
        )
        output[f"{prefix}_{metric}_mean"] = mean.to_numpy(dtype=np.float64)
        output[f"{prefix}_{metric}_std"] = std.to_numpy(dtype=np.float64)
    for group in PITCH_TYPE_GROUPS:
        rate = pd.to_numeric(
            merged.get(f"pitch_type_{group}_rate"), errors="coerce"
        ).fillna(0.0)
        output[f"{prefix}_pitch_type_{group}_rate"] = rate.to_numpy(
            dtype=np.float64
        )
    return output.loc[:, ["target_season", "main_id", *feature_columns]]


class StrictTrackmanFeatureBuilder:
    """Create and persist strict target-season entity lookup tables."""

    feature_columns = STRICT_TRACKMAN_FEATURE_COLUMNS

    def __init__(self, *, min_top_count: int = 20, min_purity: float = 0.999) -> None:
        if isinstance(min_top_count, bool) or int(min_top_count) != min_top_count:
            raise StrictTrackmanError("min_top_count must be an integer")
        self.min_top_count = int(min_top_count)
        if self.min_top_count < 1:
            raise StrictTrackmanError("min_top_count must be >= 1")
        self.min_purity = float(min_purity)
        if not np.isfinite(self.min_purity) or not 0.0 < self.min_purity <= 1.0:
            raise StrictTrackmanError("min_purity must be inside (0, 1]")
        self.target_seasons_: tuple[int, ...] = ()
        self.role_lookups_: dict[str, pd.DataFrame] = {
            "pitcher": pd.DataFrame(),
            "batter": pd.DataFrame(),
        }
        self.alignment_audit_: dict[str, int] = {}
        self.is_fitted_ = False

    def fit(
        self,
        main: pd.DataFrame,
        history: pd.DataFrame,
        *,
        target_seasons: Iterable[int] | None = None,
    ) -> "StrictTrackmanFeatureBuilder":
        prepared, trackman_audit = _prepare_trackman_history(history)
        alignment = _build_alignment_from_prepared(main, prepared, trackman_audit)
        main_seasons = pd.to_numeric(main["season"], errors="coerce").astype(np.int64)
        source_seasons = prepared["_source_season"].astype(np.int64)
        if target_seasons is None:
            first = int(min(main_seasons.min(), source_seasons.min()))
            last = int(max(main_seasons.max(), source_seasons.max())) + 1
            targets = tuple(range(first, last + 1))
        else:
            raw_targets = list(target_seasons)
            if not raw_targets:
                raise StrictTrackmanError("target_seasons must not be empty")
            targets_list: list[int] = []
            for value in raw_targets:
                if isinstance(value, bool) or int(value) != value:
                    raise StrictTrackmanError("target_seasons must contain integers")
                targets_list.append(int(value))
            targets = tuple(sorted(set(targets_list)))

        sufficient = {
            "pitcher": _entity_season_sufficient(
                prepared, entity_column="pitcher_trackman_id"
            ),
            "batter": _entity_season_sufficient(
                prepared, entity_column="batter_trackman_id"
            ),
        }
        lookups: dict[str, list[pd.DataFrame]] = {"pitcher": [], "batter": []}
        for target in targets:
            for role in ("pitcher", "batter"):
                id_map = build_high_purity_id_map(
                    alignment.aligned_rows,
                    target_season=target,
                    role=role,
                    min_top_count=self.min_top_count,
                    min_purity=self.min_purity,
                )
                summary = _cumulative_entity_summary(
                    sufficient[role], target_season=target
                )
                lookup = _build_role_lookup(
                    role=role,
                    target_season=target,
                    id_map=id_map,
                    summary=summary,
                )
                lookups[role].append(lookup)

        self.target_seasons_ = targets
        self.role_lookups_ = {}
        for role in ("pitcher", "batter"):
            columns = ["target_season", "main_id", *ROLE_FEATURE_COLUMNS[role]]
            pieces = [piece for piece in lookups[role] if len(piece)]
            table = (
                pd.concat(pieces, ignore_index=True)
                if pieces
                else pd.DataFrame(columns=columns)
            )
            table = table.loc[:, columns]
            if table.duplicated(["target_season", "main_id"]).any():
                raise RuntimeError("strict TrackMan lookup contains duplicate keys")
            self.role_lookups_[role] = table
        self.alignment_audit_ = dict(alignment.audit)
        self.alignment_audit_.update(
            {
                "pitcher_lookup_rows": int(len(self.role_lookups_["pitcher"])),
                "batter_lookup_rows": int(len(self.role_lookups_["batter"])),
            }
        )
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        _require_columns(
            frame, ["row_id", "season", "pitcher_id", "batter_id"], name="main frame"
        )
        if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
            raise StrictTrackmanError("main row_id must be non-missing and unique")
        season, valid = _integer(frame["season"], name="main.season", lower=1900)
        if not valid.all():
            raise StrictTrackmanError("main season is invalid")
        season_array = season.astype(np.int64).to_numpy()
        unknown = sorted(set(int(value) for value in np.unique(season_array)).difference(
            self.target_seasons_
        ))
        if unknown:
            raise StrictTrackmanError(
                f"no strict TrackMan lookup was built for target seasons {unknown}"
            )

        pieces: list[pd.DataFrame] = []
        for role in ("pitcher", "batter"):
            keys = pd.DataFrame(
                {
                    "_row_order": np.arange(len(frame), dtype=np.int64),
                    "target_season": season_array,
                    "main_id": _canonical_tokens(frame[f"{role}_id"]).to_numpy(
                        dtype=object
                    ),
                }
            )
            lookup = self.role_lookups_[role]
            merged = keys.merge(
                lookup,
                on=["target_season", "main_id"],
                how="left",
                sort=False,
                validate="many_to_one",
            ).sort_values("_row_order", kind="mergesort")
            columns = ROLE_FEATURE_COLUMNS[role]
            role_features = merged.loc[:, columns].apply(
                pd.to_numeric, errors="coerce"
            )
            mapped_column = f"tm_{role}_id_mapped"
            oov_column = f"tm_{role}_id_oov"
            mapped = role_features[mapped_column].fillna(0.0)
            role_features = role_features.fillna(0.0)
            role_features[mapped_column] = mapped
            role_features[oov_column] = 1.0 - mapped
            role_features.index = frame.index
            pieces.append(role_features)
        output = pd.concat(pieces, axis=1).loc[:, STRICT_TRACKMAN_FEATURE_COLUMNS]
        if not np.isfinite(output.to_numpy(dtype=np.float64, copy=False)).all():
            raise RuntimeError("strict TrackMan transform produced non-finite values")
        return output

    def to_state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_version": STRICT_TRACKMAN_STATE_VERSION,
            "builder_class": type(self).__name__,
            "min_top_count": self.min_top_count,
            "min_purity": self.min_purity,
            "target_seasons": list(self.target_seasons_),
            "feature_columns": list(STRICT_TRACKMAN_FEATURE_COLUMNS),
            "alignment_audit": dict(self.alignment_audit_),
            "role_lookups": {
                role: self.role_lookups_[role].to_dict("records")
                for role in ("pitcher", "batter")
            },
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "StrictTrackmanFeatureBuilder":
        if not isinstance(payload, dict):
            raise TypeError("strict TrackMan state must be a dictionary")
        version = payload.get("state_version")
        if isinstance(version, bool) or version != STRICT_TRACKMAN_STATE_VERSION:
            raise StrictTrackmanError("unsupported strict TrackMan state version")
        if payload.get("builder_class") != cls.__name__:
            raise StrictTrackmanError("serialized builder class does not match")
        if tuple(payload.get("feature_columns", ())) != STRICT_TRACKMAN_FEATURE_COLUMNS:
            raise StrictTrackmanError("serialized strict TrackMan features do not match code")
        obj = cls(
            min_top_count=payload.get("min_top_count"),
            min_purity=payload.get("min_purity"),
        )
        raw_targets = payload.get("target_seasons")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise StrictTrackmanError("serialized target seasons must be a non-empty list")
        targets: list[int] = []
        for value in raw_targets:
            if isinstance(value, bool) or not isinstance(value, int):
                raise StrictTrackmanError("serialized target seasons must be integers")
            targets.append(int(value))
        if targets != sorted(set(targets)):
            raise StrictTrackmanError("serialized target seasons must be sorted and unique")
        raw_lookups = payload.get("role_lookups")
        if not isinstance(raw_lookups, dict) or set(raw_lookups) != {"pitcher", "batter"}:
            raise StrictTrackmanError("serialized role lookups are malformed")
        role_lookups: dict[str, pd.DataFrame] = {}
        for role in ("pitcher", "batter"):
            records = raw_lookups[role]
            if not isinstance(records, list):
                raise StrictTrackmanError(f"serialized {role} lookup must be a list")
            columns = ["target_season", "main_id", *ROLE_FEATURE_COLUMNS[role]]
            table = pd.DataFrame.from_records(records, columns=columns)
            if len(table):
                table["target_season"] = pd.to_numeric(
                    table["target_season"], errors="raise"
                ).astype(np.int64)
                table["main_id"] = _canonical_tokens(table["main_id"])
                if table["main_id"].isna().any() or table.duplicated(
                    ["target_season", "main_id"]
                ).any():
                    raise StrictTrackmanError(f"serialized {role} lookup keys are invalid")
                for column in ROLE_FEATURE_COLUMNS[role]:
                    table[column] = pd.to_numeric(table[column], errors="raise")
                if not np.isfinite(
                    table.loc[:, ROLE_FEATURE_COLUMNS[role]].to_numpy(dtype=np.float64)
                ).all():
                    raise StrictTrackmanError(
                        f"serialized {role} lookup contains non-finite values"
                    )
                if not set(table["target_season"]).issubset(targets):
                    raise StrictTrackmanError(
                        f"serialized {role} lookup contains an unknown target season"
                    )
            role_lookups[role] = table
        audit = payload.get("alignment_audit")
        if not isinstance(audit, dict):
            raise StrictTrackmanError("serialized alignment audit is malformed")
        obj.target_seasons_ = tuple(targets)
        obj.role_lookups_ = role_lookups
        obj.alignment_audit_ = {str(key): int(value) for key, value in audit.items()}
        obj.is_fitted_ = True
        return obj

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_state_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "StrictTrackmanFeatureBuilder":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("StrictTrackmanFeatureBuilder must be fitted before transform")


__all__ = [
    "STRICT_TRACKMAN_STATE_VERSION",
    "SEQUENCE_COLUMNS",
    "ROLE_FEATURE_COLUMNS",
    "STRICT_TRACKMAN_FEATURE_COLUMNS",
    "StrictTrackmanError",
    "TrackmanTemporalLeakageError",
    "StrictGameAlignment",
    "build_strict_game_alignment",
    "build_high_purity_id_map",
    "summarize_strict_trackman_history",
    "StrictTrackmanFeatureBuilder",
]
