"""Reproducible data-understanding audit for the LG Aimers pitch dataset.

This script intentionally stops at data understanding and preprocessing
evidence.  It does not create folds, models, predictions, or submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lg_control.trackman import METRIC_COLUMNS, prepare_trackman_history


RATE_COLUMNS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


def _records(frame: pd.DataFrame, digits: int = 8) -> list[dict[str, object]]:
    clean = frame.copy()
    for column in clean.select_dtypes(include=["float", "float32", "float64"]):
        clean[column] = clean[column].round(digits)
    return json.loads(clean.to_json(orient="records"))


def _save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def _group_rate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, observed=True, dropna=False)["control_success"]
        .agg(rows="size", success_rate="mean")
        .reset_index()
    )


def _count_update_audit(
    frame: pd.DataFrame, entity: str, count: str, rate: str
) -> dict[str, object]:
    grouped = frame.groupby(entity, sort=False, observed=True)
    next_count = grouped[count].shift(-1)
    next_rate = grouped[rate].shift(-1)
    current_count = pd.to_numeric(frame[count], errors="coerce")
    current_rate = pd.to_numeric(frame[rate], errors="coerce")
    valid = next_count.eq(current_count + 1) & next_rate.notna()
    current_total = current_rate.fillna(0.0) * current_count
    recovered = next_rate * next_count - current_total
    error = (recovered.loc[valid] - frame.loc[valid, "control_success"]).abs()
    rounded = np.rint(recovered.loc[valid]).astype(np.int8)
    rounded_matches = rounded.eq(frame.loc[valid, "control_success"].astype(np.int8))
    return {
        "valid_transitions": int(valid.sum()),
        "rounded_target_recoveries": int(rounded_matches.sum()),
        "rounded_recovery_rate": float(rounded_matches.mean()) if len(error) else None,
        "median_absolute_reconstruction_error": float(error.median()) if len(error) else None,
        "max_absolute_error": float(error.max()) if len(error) else None,
        "interpretation": "CSV rounding leaves a small numeric error, but nearest-integer cross-row differencing reconstructs the label and is forbidden on evaluation rows.",
    }


def _prior_seen_table(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[object] = set()
    for season, group in frame.groupby("season", sort=True):
        values = group[entity]
        rows.append(
            {
                "season": int(season),
                "entity": entity,
                "rows": int(len(group)),
                "rows_seen_in_prior_seasons": int(values.isin(seen).sum()),
                "row_coverage": float(values.isin(seen).mean()),
                "unique_entities": int(values.nunique()),
                "new_unique_entities": int(len(set(values.unique()).difference(seen))),
            }
        )
        seen.update(values.unique().tolist())
    return pd.DataFrame(rows)


def analyse_main(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    target_by_season_game_type = _group_rate(train, ["season", "game_type"])
    target_by_season = _group_rate(train, ["season"])
    target_by_month = _group_rate(train, ["season", "game_month"])
    target_by_count = _group_rate(train, ["balls_before", "strikes_before"])
    target_by_base_out = _group_rate(train, ["base_state", "outs_before"])
    target_by_platoon = _group_rate(train, ["pitcher_hand", "batter_hand"])
    target_by_inning = _group_rate(train.assign(inning_bucket=train["inning"].clip(upper=10)), ["inning_bucket"])

    li_decile = pd.qcut(train["li"], q=10, duplicates="drop")
    target_by_li = (
        train.assign(li_decile=li_decile)
        .groupby("li_decile", observed=True)["control_success"]
        .agg(rows="size", success_rate="mean")
        .reset_index()
    )
    target_by_li["li_decile"] = target_by_li["li_decile"].astype(str)

    rate_bins = np.linspace(0.0, 1.0, 11)
    calibration_tables: list[pd.DataFrame] = []
    for column in [
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]:
        binned = pd.cut(train[column], rate_bins, include_lowest=True)
        table = (
            train.assign(rate_bin=binned)
            .dropna(subset=[column])
            .groupby(["season", "rate_bin"], observed=True)
            .agg(
                rows=("control_success", "size"),
                historical_rate_mean=(column, "mean"),
                observed_success_rate=("control_success", "mean"),
            )
            .reset_index()
        )
        table["feature"] = column
        table["rate_bin"] = table["rate_bin"].astype(str)
        calibration_tables.append(table)
    rate_calibration = pd.concat(calibration_tables, ignore_index=True)

    history_bins = [-1, 0, 9, 49, 199, 999, 4999, np.inf]
    history_labels = ["0", "1-9", "10-49", "50-199", "200-999", "1k-4,999", "5k+"]
    history_effects: list[pd.DataFrame] = []
    for count_column in ["asof_pitcher_n", "asof_batter_n"]:
        bucket = pd.cut(train[count_column], history_bins, labels=history_labels)
        table = (
            train.assign(history_bucket=bucket)
            .groupby("history_bucket", observed=True)["control_success"]
            .agg(rows="size", success_rate="mean")
            .reset_index()
        )
        table["count_feature"] = count_column
        table["history_bucket"] = table["history_bucket"].astype(str)
        history_effects.append(table)
    target_by_history = pd.concat(history_effects, ignore_index=True)

    missingness = (
        train.isna()
        .sum()
        .rename("missing_rows")
        .to_frame()
        .assign(missing_rate=lambda x: x["missing_rows"] / len(train))
        .query("missing_rows > 0")
        .sort_values("missing_rate", ascending=False)
        .reset_index(names="column")
    )

    categorical_candidates = [
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "top_bottom",
        "game_type",
        "balls_before",
        "strikes_before",
        "outs_before",
        "base_state",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
    ]
    cardinality = pd.DataFrame(
        {
            "column": categorical_candidates,
            "unique_values": [int(train[column].nunique(dropna=True)) for column in categorical_candidates],
            "missing_rows": [int(train[column].isna().sum()) for column in categorical_candidates],
        }
    ).sort_values("unique_values", ascending=False)

    prior_seen = pd.concat(
        [_prior_seen_table(train, "pitcher_id"), _prior_seen_table(train, "batter_id")],
        ignore_index=True,
    )

    numeric = train.select_dtypes(include=[np.number]).drop(columns=["row_id"], errors="ignore")
    correlation = numeric.corr(method="pearson")
    target_correlations = (
        correlation["control_success"]
        .drop("control_success")
        .rename("pearson_correlation")
        .to_frame()
        .assign(abs_correlation=lambda x: x["pearson_correlation"].abs())
        .sort_values("abs_correlation", ascending=False)
        .reset_index(names="feature")
    )

    selected_drift_columns = [
        "home_win_expectancy",
        "li",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
    ]
    global_means = train[selected_drift_columns].mean()
    global_stds = train[selected_drift_columns].std().replace(0, 1)
    season_means = train.groupby("season", observed=True)[selected_drift_columns].mean()
    season_drift = ((season_means - global_means) / global_stds).reset_index().melt(
        id_vars="season", var_name="feature", value_name="standardized_mean_shift"
    )

    rate_validity = {
        column: {
            "missing_rows": int(train[column].isna().sum()),
            "below_zero": int(train[column].lt(0).sum()),
            "above_one": int(train[column].gt(1).sum()),
        }
        for column in RATE_COLUMNS
    }
    base_mapping = {
        "___": (0, 0, 0),
        "1__": (1, 0, 0),
        "_2_": (0, 1, 0),
        "__3": (0, 0, 1),
        "12_": (1, 1, 0),
        "1_3": (1, 0, 1),
        "_23": (0, 1, 1),
        "123": (1, 1, 1),
    }
    expected_base = train["base_state"].map(base_mapping)
    base_mismatch = sum(
        value != expected
        for value, expected in zip(
            train[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].itertuples(index=False, name=None),
            expected_base,
            strict=True,
        )
    )
    pitcher_outcome_sum = train[
        [
            "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate",
        ]
    ].sum(axis=1, min_count=4)
    pitchmix_sum = train[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].sum(axis=1, min_count=3)

    redundancies = {
        "run_total_equals_top_plus_bottom_mismatches": int(
            train["run_total_before"].ne(train["run_top_before"] + train["run_bot_before"]).sum()
        ),
        "score_diff_home_equals_bottom_minus_top_mismatches": int(
            train["score_diff_home"].ne(train["run_bot_before"] - train["run_top_before"]).sum()
        ),
        "num_runners_equals_flags_sum_mismatches": int(
            train["num_runners_on"]
            .ne(train[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1))
            .sum()
        ),
        "base_state_equals_flags_mismatches": int(base_mismatch),
        "home_plus_away_expectancy_max_error": float(
            (train["home_win_expectancy"] + train["away_win_expectancy"] - 100.0).abs().max()
        ),
        "pitcher_n_equals_pitchmix_n_mismatches": int(
            train["asof_pitcher_n"].ne(train["asof_pitcher_pitchmix_n"]).sum()
        ),
        "pitcher_outcome_rate_sum": {
            "minimum": float(pitcher_outcome_sum.dropna().min()),
            "median": float(pitcher_outcome_sum.dropna().median()),
            "maximum": float(pitcher_outcome_sum.dropna().max()),
            "interpretation": "These four outcome rates overlap and must not be compositional-normalized.",
        },
        "pitchmix_rates_sum_to_one_max_error": float((pitchmix_sum.dropna() - 1.0).abs().max()),
    }

    row_numbers = train["row_id"].astype("string").str.extract(r"(\d+)$", expand=False).astype(np.int64)
    row_id_audit = {
        "unique": bool(train["row_id"].is_unique),
        "numeric_suffix_min": int(row_numbers.min()),
        "numeric_suffix_max": int(row_numbers.max()),
        "numeric_suffix_is_exact_sequence": bool(
            np.array_equal(row_numbers.to_numpy(), np.arange(1, len(train) + 1))
        ),
        "season_is_non_decreasing_in_file_order": bool(train["season"].is_monotonic_increasing),
        "interpretation": "row_id/file order encode chronology and must remain an identifier rather than a model feature.",
    }

    test_sample_coverage = {
        "warning": "The distributed test file has only five schema examples and is not distributionally representative.",
        "rows": int(len(test)),
        "pitcher_rows_seen_in_train": int(test["pitcher_id"].isin(set(train["pitcher_id"])).sum()),
        "batter_rows_seen_in_train": int(test["batter_id"].isin(set(train["batter_id"])).sum()),
    }

    tables = {
        "target_by_season": target_by_season,
        "target_by_season_game_type": target_by_season_game_type,
        "target_by_month": target_by_month,
        "target_by_count": target_by_count,
        "target_by_base_out": target_by_base_out,
        "target_by_platoon": target_by_platoon,
        "target_by_inning": target_by_inning,
        "target_by_leverage_decile": target_by_li,
        "rate_calibration": rate_calibration,
        "target_by_history_count": target_by_history,
        "missingness": missingness,
        "categorical_cardinality": cardinality,
        "prior_season_entity_coverage": prior_seen,
        "target_correlations": target_correlations,
        "season_numeric_drift": season_drift,
    }
    for name, table in tables.items():
        _save_table(table, output_dir, name)

    summary = {
        "shape": {"rows": int(len(train)), "columns": int(train.shape[1])},
        "target": {
            "positive_rows": int(train["control_success"].sum()),
            "success_rate": float(train["control_success"].mean()),
        },
        "date_scope": {
            "seasons": sorted(int(value) for value in train["season"].unique()),
            "months": sorted(int(value) for value in train["game_month"].unique()),
        },
        "duplicates": {
            "row_id_duplicates": int(train["row_id"].duplicated().sum()),
            "full_feature_and_target_duplicates_excluding_row_id": int(
                train.drop(columns="row_id").duplicated().sum()
            ),
        },
        "row_id_audit": row_id_audit,
        "redundancies": redundancies,
        "rate_validity": rate_validity,
        "cold_start": {
            "pitcher_zero_history_rows": int(train["asof_pitcher_n"].eq(0).sum()),
            "batter_zero_history_rows": int(train["asof_batter_n"].eq(0).sum()),
            "recent_game_missing_rows": int(train["asof_pitcher_prev5_game_success_rate"].isna().sum()),
        },
        "cross_row_leakage_audit": {
            "pitcher": _count_update_audit(
                train, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
            ),
            "batter": _count_update_audit(
                train, "batter_id", "asof_batter_n", "asof_batter_success_rate"
            ),
        },
        "test_sample_coverage": test_sample_coverage,
        "top_absolute_target_correlations": _records(target_correlations.head(12)),
        "tables": {name: f"{name}.csv" for name in tables},
        "dashboard": {
            "target_by_season_game_type": _records(target_by_season_game_type),
            "target_by_count": _records(target_by_count),
            "target_by_base_out": _records(target_by_base_out),
            "rate_calibration": _records(
                rate_calibration.loc[
                    rate_calibration["feature"].eq("asof_pitcher_success_rate")
                ]
            ),
            "missingness": _records(missingness),
            "prior_seen": _records(prior_seen),
            "season_drift": _records(season_drift),
        },
    }
    return summary


def analyse_trackman(
    raw: pd.DataFrame, main_train: pd.DataFrame, output_dir: Path
) -> dict[str, object]:
    prepared, preparation_report = prepare_trackman_history(raw)
    pitch_mix = (
        prepared.groupby(["source_season", "pitch_type_group"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    pitch_mix["rate"] = pitch_mix["rows"] / pitch_mix.groupby("source_season")["rows"].transform("sum")
    physical_trends = (
        prepared.groupby(["source_season", "pitch_type_group"], observed=True)[list(METRIC_COLUMNS)]
        .agg(["count", "mean", "std"])
    )
    physical_trends.columns = [f"{metric}_{stat}" for metric, stat in physical_trends.columns]
    physical_trends = physical_trends.reset_index()
    count_pitch_mix = (
        prepared.groupby(["balls_before", "strikes_before", "pitch_type_group"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    count_pitch_mix["rate"] = count_pitch_mix["rows"] / count_pitch_mix.groupby(
        ["balls_before", "strikes_before"]
    )["rows"].transform("sum")

    physical_missingness = (
        raw[list(METRIC_COLUMNS)]
        .isna()
        .sum()
        .rename("missing_rows")
        .to_frame()
        .assign(missing_rate=lambda x: x["missing_rows"] / len(raw))
        .reset_index(names="metric")
    )
    physical_quantiles = raw[list(METRIC_COLUMNS)].quantile(
        [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]
    ).reset_index(names="quantile")
    physical_correlation = raw[list(METRIC_COLUMNS)].corr().reset_index(names="metric")

    tagged = raw["tagged_pitch_type"].astype("string").str.strip().str.lower()
    automatic = raw["auto_pitch_type"].astype("string").str.strip().str.lower()
    pitch_label_agreement = {
        "both_nonmissing_rows": int((tagged.notna() & automatic.notna()).sum()),
        "exact_name_agreement_rate": float(tagged.eq(automatic).mean()),
        "tagged_unique_values": int(tagged.nunique(dropna=True)),
        "auto_unique_values": int(automatic.nunique(dropna=True)),
    }
    date_tokens = raw["game_date"].astype("string")
    date_formats = {
        "slash_rows": int(date_tokens.str.contains("/", regex=False, na=False).sum()),
        "dash_rows": int(date_tokens.str.contains("-", regex=False, na=False).sum()),
        "other_rows": int(
            (~date_tokens.str.contains(r"[/\-]", regex=True, na=False)).sum()
        ),
    }

    id_overlap = {
        "pitcher_id_overlap": int(
            len(set(main_train["pitcher_id"]).intersection(set(raw["pitcher_trackman_id"])))
        ),
        "batter_id_overlap": int(
            len(set(main_train["batter_id"]).intersection(set(raw["batter_trackman_id"])))
        ),
        "interpretation": "Main and TrackMan entity IDs are separate namespaces; direct player joins are invalid.",
    }

    tables = {
        "trackman_pitch_mix": pitch_mix,
        "trackman_physical_trends": physical_trends,
        "trackman_count_pitch_mix": count_pitch_mix,
        "trackman_physical_missingness": physical_missingness,
        "trackman_physical_quantiles": physical_quantiles,
        "trackman_physical_correlation": physical_correlation,
    }
    for name, table in tables.items():
        _save_table(table, output_dir, name)

    return {
        "shape": {"rows": int(len(raw)), "columns": int(raw.shape[1])},
        "preparation": preparation_report,
        "date_token_formats": date_formats,
        "id_overlap_with_main": id_overlap,
        "pitch_label_agreement": pitch_label_agreement,
        "physical_missingness": _records(physical_missingness),
        "tables": {name: f"{name}.csv" for name in tables},
        "dashboard": {
            "pitch_mix": _records(pitch_mix),
            "physical_trends": _records(
                physical_trends[
                    [
                        "source_season",
                        "pitch_type_group",
                        "rel_speed_count",
                        "rel_speed_mean",
                        "spin_rate_mean",
                    ]
                ]
            ),
            "count_pitch_mix": _records(count_pitch_mix),
            "physical_missingness": _records(physical_missingness),
        },
    }


def make_figures(summary: dict[str, object], output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    target = pd.DataFrame(summary["main"]["dashboard"]["target_by_season_game_type"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(data=target, x="season", y="success_rate", hue="game_type", marker="o", ax=ax)
    ax.set(title="Control success by season and game type", ylabel="Success rate", xlabel="Season")
    ax.set_ylim(0.42, 0.74)
    fig.tight_layout()
    fig.savefig(figure_dir / "01-season-game-type-line.png", dpi=160)
    plt.close(fig)

    count = pd.DataFrame(summary["main"]["dashboard"]["target_by_count"])
    pivot = count.pivot(index="balls_before", columns="strikes_before", values="success_rate")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set(title="Control success by count", xlabel="Strikes before", ylabel="Balls before")
    fig.tight_layout()
    fig.savefig(figure_dir / "02-count-heatmap.png", dpi=160)
    plt.close(fig)

    base = pd.DataFrame(summary["main"]["dashboard"]["target_by_base_out"])
    order = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
    pivot = base.pivot(index="base_state", columns="outs_before", values="success_rate").reindex(order)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako", ax=ax)
    ax.set(title="Control success by base/out state", xlabel="Outs before", ylabel="Base state")
    fig.tight_layout()
    fig.savefig(figure_dir / "03-base-out-heatmap.png", dpi=160)
    plt.close(fig)

    calibration = pd.DataFrame(summary["main"]["dashboard"]["rate_calibration"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.lineplot(
        data=calibration,
        x="historical_rate_mean",
        y="observed_success_rate",
        hue="season",
        marker="o",
        palette="viridis",
        ax=ax,
    )
    limits = [calibration["historical_rate_mean"].min(), calibration["historical_rate_mean"].max()]
    ax.plot(limits, limits, color="grey", linewidth=1, linestyle="--", label="identity")
    ax.set(
        title="Historical pitcher rate vs current-pitch success",
        xlabel="Mean historical success rate in bin",
        ylabel="Observed current success rate",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "04-pitcher-rate-reliability-line.png", dpi=160)
    plt.close(fig)

    missing = pd.DataFrame(summary["main"]["dashboard"]["missingness"])
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    sns.barplot(data=missing, y="column", x="missing_rate", color=sns.color_palette()[0], ax=ax)
    ax.set(title="Missingness in train.csv", xlabel="Missing row fraction", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(figure_dir / "05-main-missingness-bar.png", dpi=160)
    plt.close(fig)

    corr = pd.read_csv(output_dir / "target_correlations.csv").head(18)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.barplot(data=corr, y="feature", x="pearson_correlation", hue="pearson_correlation", palette="vlag", legend=False, ax=ax)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set(title="Strongest univariate target correlations", xlabel="Pearson correlation", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(figure_dir / "06-target-correlation-diverging-bar.png", dpi=160)
    plt.close(fig)

    prior = pd.DataFrame(summary["main"]["dashboard"]["prior_seen"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(data=prior, x="season", y="row_coverage", hue="entity", marker="o", ax=ax)
    ax.set(title="Rows whose entity appeared in an earlier season", xlabel="Season", ylabel="Prior-season coverage")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figure_dir / "07-entity-continuity-line.png", dpi=160)
    plt.close(fig)

    drift = pd.DataFrame(summary["main"]["dashboard"]["season_drift"])
    pivot = drift.pivot(index="feature", columns="season", values="standardized_mean_shift")
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    sns.heatmap(pivot, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set(title="Seasonal numeric drift", xlabel="Season", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(figure_dir / "08-season-drift-heatmap.png", dpi=160)
    plt.close(fig)

    mix = pd.DataFrame(summary["trackman"]["dashboard"]["pitch_mix"])
    mix_pivot = mix.pivot(index="source_season", columns="pitch_type_group", values="rate").fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    mix_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
    ax.set(title="TrackMan pitch-type composition", xlabel="Season", ylabel="Share")
    ax.legend(title="Pitch group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "09-trackman-pitch-mix-stacked-bar.png", dpi=160)
    plt.close(fig)

    physical = pd.DataFrame(summary["trackman"]["dashboard"]["physical_trends"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    sns.lineplot(data=physical, x="source_season", y="rel_speed_mean", hue="pitch_type_group", marker="o", ax=axes[0])
    sns.lineplot(data=physical, x="source_season", y="spin_rate_mean", hue="pitch_type_group", marker="o", ax=axes[1], legend=False)
    axes[0].set(title="Release speed", xlabel="Season", ylabel="Mean rel_speed")
    axes[1].set(title="Spin rate", xlabel="Season", ylabel="Mean spin_rate")
    axes[0].legend(title="Pitch group", fontsize="small")
    fig.tight_layout()
    fig.savefig(figure_dir / "10-trackman-physical-trends-small-multiples.png", dpi=160)
    plt.close(fig)

    tm_corr = pd.read_csv(output_dir / "trackman_physical_correlation.csv").set_index("metric")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(tm_corr, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set(title="TrackMan physical-feature correlations")
    fig.tight_layout()
    fig.savefig(figure_dir / "11-trackman-physical-correlation-heatmap.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "test.csv")
    parser.add_argument(
        "--trackman", type=Path, default=PROJECT_ROOT / "data" / "trackman_history.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_understanding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    main_summary = analyse_main(train, test, args.output_dir)

    trackman_raw = pd.read_csv(args.trackman, low_memory=False)
    trackman_summary = analyse_trackman(trackman_raw, train, args.output_dir)
    summary = {
        "scope": "data understanding and preprocessing evidence only",
        "main": main_summary,
        "trackman": trackman_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(summary, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "train_rows": len(train),
        "trackman_rows": len(trackman_raw),
        "figures": len(list((args.output_dir / "figures").glob("*.png"))),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
