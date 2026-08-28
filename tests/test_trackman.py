from __future__ import annotations

import unittest

import pandas as pd

from src.lg_control.trackman import (
    METRIC_COLUMNS,
    build_trackman_lookup,
    join_trackman_context,
)


def history_fixture() -> pd.DataFrame:
    rows = []
    for index, (season, date, weekday, speed) in enumerate(
        [(2019, "03/29/2019", 4, 140.0), (2020, "2020-03-28", 5, 150.0)],
        start=1,
    ):
        row = {
            "trackman_id": index,
            "season": season,
            "game_date": date,
            "game_month": 3,
            "game_dayofweek": weekday,
            "trackman_game_id": f"G{index}",
            "pitch_no": 1,
            "inning": 1,
            "top_bottom": "Top",
            "balls_before": 0,
            "strikes_before": 0,
            "outs_before": 0,
            "pitch_type_group": "fastball",
        }
        for metric in METRIC_COLUMNS:
            row[metric] = speed if metric == "rel_speed" else float(index)
        rows.append(row)
    return pd.DataFrame(rows)


class TrackmanCutoffTests(unittest.TestCase):
    def test_strict_prior_season_cutoff_and_empty_first_season(self) -> None:
        lookup = build_trackman_lookup(
            history_fixture(), target_seasons=[2019, 2020, 2021], min_group_count=1
        )
        global_rows = lookup.loc[lookup["lookup_level"].eq("global")].set_index(
            "target_season"
        )
        self.assertEqual(int(global_rows.loc[2019, "tm_history_count"]), 0)
        self.assertEqual(int(global_rows.loc[2020, "tm_history_count"]), 1)
        self.assertEqual(int(global_rows.loc[2021, "tm_history_count"]), 2)

    def test_join_is_row_local_and_preserves_order(self) -> None:
        lookup = build_trackman_lookup(
            history_fixture(), target_seasons=[2019, 2020, 2021], min_group_count=1
        )
        frame = pd.DataFrame(
            {
                "row_id": ["B", "A"],
                "season": [2021, 2020],
                "game_month": [3, 3],
                "inning": [1, 1],
                "top_bottom": ["T", "T"],
                "balls_before": [0, 0],
                "strikes_before": [0, 0],
                "outs_before": [0, 0],
            }
        )
        joined = join_trackman_context(frame, lookup)
        self.assertEqual(joined["row_id"].tolist(), ["B", "A"])
        self.assertEqual(joined["tm_history_count"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()

