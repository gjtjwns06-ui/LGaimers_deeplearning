"""Diagnose the post-2022 F label-regime shift without using future features.

Auxiliary outcomes are reconstructed only for this offline train-data audit.
They are never serialized as inference lookups or joined into validation rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_control.auxiliary_targets import recover_fold_train_auxiliary_targets  # noqa: E402


DATA = ROOT / "data" / "train.csv"
OUTPUT = ROOT / "reports" / "f_regime_shift"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def standardized_rate(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    outcome: str,
    cells: list[str],
    prior_strength: float = 20.0,
) -> dict[str, float]:
    """Apply old conditional rates to the new season's observed cell mix."""

    old_global = float(old[outcome].mean())
    grouped = old.groupby(cells, observed=True)[outcome].agg(["mean", "size"])
    grouped["smooth"] = (
        grouped["mean"] * grouped["size"] + old_global * prior_strength
    ) / (grouped["size"] + prior_strength)
    new_mix = new.groupby(cells, observed=True).size().rename("new_n").to_frame()
    joined = new_mix.join(grouped[["smooth", "size"]], how="left")
    coverage = float(joined.loc[joined["smooth"].notna(), "new_n"].sum() / len(new))
    joined["smooth"] = joined["smooth"].fillna(old_global)
    counterfactual = weighted_mean(joined["smooth"], joined["new_n"])
    actual_new = float(new[outcome].mean())
    return {
        "old_rate": old_global,
        "new_rate": actual_new,
        "raw_delta": actual_new - old_global,
        "old_rate_new_composition": counterfactual,
        "composition_component": counterfactual - old_global,
        "within_cell_component": actual_new - counterfactual,
        "new_cell_coverage": coverage,
    }


def matched_entity_delta(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    entity: str,
    outcome: str,
    minimum: int,
) -> dict[str, float | int]:
    old_rates = old.groupby(entity, observed=True)[outcome].agg(old_rate="mean", old_n="size")
    new_rates = new.groupby(entity, observed=True)[outcome].agg(new_rate="mean", new_n="size")
    matched = old_rates.join(new_rates, how="inner")
    matched = matched[(matched.old_n >= minimum) & (matched.new_n >= minimum)].copy()
    matched["delta"] = matched.new_rate - matched.old_rate
    matched["weight"] = np.minimum(matched.old_n, matched.new_n)
    if matched.empty:
        return {"entities": 0, "rows_weight": 0, "weighted_delta": np.nan, "median_delta": np.nan}
    return {
        "entities": int(len(matched)),
        "rows_weight": int(matched.weight.sum()),
        "weighted_delta": weighted_mean(matched.delta, matched.weight),
        "median_delta": float(matched.delta.median()),
        "share_same_direction": float((np.sign(matched.delta) == np.sign(weighted_mean(matched.delta, matched.weight))).mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, low_memory=False)
    recovered = recover_fold_train_auxiliary_targets(frame, partition="fold_train")
    for column in ("aux_reverse", "aux_middle", "aux_ball", "aux_strike"):
        frame[column] = recovered[column].to_numpy()
    frame = frame[recovered.auxiliary_valid_mask.to_numpy()].copy()
    frame["failure"] = 1.0 - frame["control_success"].astype(float)
    frame["reverse_only"] = ((frame.aux_reverse == 1) & (frame.aux_middle == 0)).astype(float)
    frame["reverse_middle"] = ((frame.aux_reverse == 1) & (frame.aux_middle == 1)).astype(float)
    frame["middle_only"] = ((frame.aux_reverse == 0) & (frame.aux_middle == 1)).astype(float)
    frame["other_failure"] = (
        (frame.control_success == 0) & (frame.aux_reverse == 0) & (frame.aux_middle == 0)
    ).astype(float)
    frame["count"] = frame.balls_before.astype(str) + "-" + frame.strikes_before.astype(str)
    frame["inning_bucket"] = pd.cut(
        frame.inning, bins=[0, 3, 6, 9, np.inf], labels=["1-3", "4-6", "7-9", "10+"]
    )

    outcomes = [
        "control_success",
        "failure",
        "aux_reverse",
        "aux_middle",
        "reverse_only",
        "reverse_middle",
        "middle_only",
        "other_failure",
        "aux_ball",
        "aux_strike",
    ]
    summary = (
        frame.groupby(["season", "game_type"], observed=True)[outcomes]
        .agg(["size", "mean"])
    )
    # Retain one size per slice and all means.
    compact = pd.DataFrame(index=summary.index)
    compact["n"] = summary[("control_success", "size")]
    for outcome in outcomes:
        compact[outcome] = summary[(outcome, "mean")]
    compact.reset_index().to_csv(OUTPUT / "annual_outcomes.csv", index=False)

    f = frame[(frame.game_type == "F") & frame.season.isin([2021, 2022, 2023, 2024])].copy()
    cells = [
        "count",
        "pitcher_hand",
        "batter_hand",
        "top_bottom",
        "outs_before",
        "inning_bucket",
        "num_runners_on",
    ]
    standardization: list[dict[str, float | int | str]] = []
    matched: list[dict[str, float | int | str]] = []
    for new_season in (2023, 2024):
        old = f[f.season == 2022]
        new = f[f.season == new_season]
        for outcome in ("control_success", "aux_reverse", "aux_middle", "aux_ball", "aux_strike"):
            standardization.append(
                {
                    "old_season": 2022,
                    "new_season": new_season,
                    "outcome": outcome,
                    **standardized_rate(old, new, outcome=outcome, cells=cells),
                }
            )
        for entity, minimum in (("pitcher_id", 50), ("pitcher_team_id", 200), ("batter_team_id", 200)):
            for outcome in ("control_success", "aux_reverse", "aux_middle", "aux_ball", "aux_strike"):
                matched.append(
                    {
                        "old_season": 2022,
                        "new_season": new_season,
                        "entity": entity,
                        "outcome": outcome,
                        **matched_entity_delta(old, new, entity=entity, outcome=outcome, minimum=minimum),
                    }
                )
    pd.DataFrame(standardization).to_csv(OUTPUT / "composition_standardization.csv", index=False)
    pd.DataFrame(matched).to_csv(OUTPUT / "matched_entities.csv", index=False)

    monthly = (
        f.groupby(["season", "game_month"], observed=True)[outcomes]
        .agg(["size", "mean"])
    )
    monthly_compact = pd.DataFrame(index=monthly.index)
    monthly_compact["n"] = monthly[("control_success", "size")]
    for outcome in outcomes:
        monthly_compact[outcome] = monthly[(outcome, "mean")]
    monthly_compact.reset_index().to_csv(OUTPUT / "f_monthly_outcomes.csv", index=False)

    team = (
        f.groupby(["season", "pitcher_team_id"], observed=True)[outcomes]
        .agg(["size", "mean"])
    )
    team_compact = pd.DataFrame(index=team.index)
    team_compact["n"] = team[("control_success", "size")]
    for outcome in outcomes:
        team_compact[outcome] = team[(outcome, "mean")]
    team_compact.reset_index().to_csv(OUTPUT / "f_team_outcomes.csv", index=False)

    count = (
        f.groupby(["season", "count"], observed=True)[outcomes]
        .agg(["size", "mean"])
    )
    count_compact = pd.DataFrame(index=count.index)
    count_compact["n"] = count[("control_success", "size")]
    for outcome in outcomes:
        count_compact[outcome] = count[(outcome, "mean")]
    count_compact.reset_index().to_csv(OUTPUT / "f_count_outcomes.csv", index=False)

    payload = {
        "rows_total": int(len(frame)),
        "valid_fraction": float(len(frame) / 1_475_092),
        "annual": compact.reset_index().to_dict(orient="records"),
        "standardization": standardization,
        "matched": matched,
    }
    (OUTPUT / "diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("ANNUAL F/R 2021-2024")
    print(compact.reset_index().query("season >= 2021").to_string(index=False))
    print("\nSTANDARDIZED F 2022 -> 2023/2024")
    print(pd.DataFrame(standardization).to_string(index=False))
    print("\nMATCHED F ENTITIES")
    print(pd.DataFrame(matched).to_string(index=False))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
