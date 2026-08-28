from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.lg_control.strict_trackman import (
    STRICT_TRACKMAN_FEATURE_COLUMNS,
    StrictTrackmanError,
    StrictTrackmanFeatureBuilder,
    TrackmanTemporalLeakageError,
    build_high_purity_id_map,
    build_strict_game_alignment,
    summarize_strict_trackman_history,
)
from src.lg_control.trackman import METRIC_COLUMNS, PITCH_TYPE_GROUPS


_GAMES = (
    (2019, "2019-03-29", 4),
    (2020, "2020-03-28", 5),
    (2021, "2021-03-27", 5),
    (2022, "2022-03-26", 5),
)


def _context(pitch_index: int) -> tuple[int, str, int, int, int]:
    inning = 1 + pitch_index // 12
    half = "T" if pitch_index % 12 < 6 else "B"
    return (
        inning,
        half,
        pitch_index % 4,
        (pitch_index // 4) % 3,
        (pitch_index // 2) % 3,
    )


def strict_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows: list[dict[str, object]] = []
    trackman_rows: list[dict[str, object]] = []
    trackman_id = 1
    for season, game_date, weekday in _GAMES:
        for pitch_index in range(24):
            inning, half, balls, strikes, outs = _context(pitch_index)
            main_rows.append(
                {
                    "row_id": f"M{season}_{pitch_index + 1}",
                    "season": season,
                    "game_month": 3,
                    "game_dayofweek": weekday,
                    "inning": inning,
                    "top_bottom": half,
                    "balls_before": balls,
                    "strikes_before": strikes,
                    "outs_before": outs,
                    "pitcher_id": "100",
                    "batter_id": "200",
                }
            )
            row: dict[str, object] = {
                "trackman_id": trackman_id,
                "season": season,
                "game_date": game_date,
                "game_month": 3,
                "game_dayofweek": weekday,
                "trackman_game_id": f"TM{season}",
                "pitch_no": pitch_index + 1,
                "inning": inning,
                "top_bottom": "Top" if half == "T" else "Bottom",
                "balls_before": balls,
                "strikes_before": strikes,
                "outs_before": outs,
                "pitcher_trackman_id": "P900",
                "batter_trackman_id": "B800",
                "pitch_type_group": PITCH_TYPE_GROUPS[
                    pitch_index % len(PITCH_TYPE_GROUPS)
                ],
            }
            for metric_index, metric in enumerate(METRIC_COLUMNS):
                row[metric] = (
                    100.0
                    + (season - 2019) * 10.0
                    + pitch_index * 0.1
                    + metric_index
                )
            trackman_rows.append(row)
            trackman_id += 1
    return pd.DataFrame(main_rows), pd.DataFrame(trackman_rows)


class StrictGameAlignmentTests(unittest.TestCase):
    def test_exact_alignment_is_order_independent_and_row_verified(self) -> None:
        main, history = strict_fixture()
        shuffled = history.sample(frac=1.0, random_state=17).reset_index(drop=True)
        result = build_strict_game_alignment(main, shuffled)

        self.assertEqual(result.audit["strict_matched_games"], 4)
        self.assertEqual(result.audit["strict_aligned_rows"], 96)
        self.assertFalse(result.matches["main_game_index"].duplicated().any())
        self.assertFalse(result.matches["trackman_game_id"].duplicated().any())
        self.assertEqual(
            result.aligned_rows.sort_values("main_position")["main_row_id"].tolist(),
            main["row_id"].tolist(),
        )

    def test_bidirectional_uniqueness_rejects_either_side_ambiguity(self) -> None:
        main, history = strict_fixture()

        copied_main = main.loc[main["season"].eq(2019)].copy()
        copied_main["row_id"] = "COPY_" + copied_main["row_id"].astype(str)
        ambiguous_main = pd.concat(
            [
                main.loc[main["season"].eq(2019)],
                copied_main,
                main.loc[main["season"].gt(2019)],
            ],
            ignore_index=True,
        )
        main_result = build_strict_game_alignment(ambiguous_main, history)
        self.assertEqual(main_result.audit["strict_matched_games"], 3)
        self.assertEqual(
            main_result.audit["reused_trackman_candidate_games_rejected"], 1
        )
        self.assertEqual(
            main_result.audit["non_one_to_one_candidate_pairs_rejected"], 2
        )

        copied_trackman = history.loc[history["season"].eq(2019)].copy()
        copied_trackman["trackman_id"] += int(history["trackman_id"].max())
        copied_trackman["trackman_game_id"] = "TM2019_COPY"
        ambiguous_trackman = pd.concat(
            [history, copied_trackman], ignore_index=True
        )
        trackman_result = build_strict_game_alignment(main, ambiguous_trackman)
        self.assertEqual(trackman_result.audit["strict_matched_games"], 3)
        self.assertEqual(
            trackman_result.audit["reused_main_candidate_games_rejected"], 1
        )

    def test_multi_date_games_and_row_weekday_mismatches_are_rejected(self) -> None:
        main, history = strict_fixture()
        reused = history.loc[history["season"].eq(2019)].copy()
        reused["trackman_id"] += int(history["trackman_id"].max())
        reused["season"] = 2020
        reused["game_date"] = "2020-04-04"
        reused["game_month"] = 4
        reused["game_dayofweek"] = 5
        with_reuse = pd.concat([history, reused], ignore_index=True)

        reuse_result = build_strict_game_alignment(main, with_reuse)
        self.assertEqual(reuse_result.audit["multi_date_game_ids_observed"], 1)
        self.assertEqual(reuse_result.audit["strict_matched_games"], 3)

        bad_main = main.copy()
        bad_main.loc[1, "game_dayofweek"] = 3
        weekday_result = build_strict_game_alignment(bad_main, history)
        self.assertEqual(weekday_result.audit["strict_matched_games"], 3)
        self.assertEqual(weekday_result.audit["row_weekday_mismatches_rejected"], 1)
        self.assertEqual(
            weekday_result.audit["row_calendar_mismatch_games_rejected"], 1
        )

    def test_content_duplicates_and_invalid_context_rows_are_removed(self) -> None:
        main, history = strict_fixture()
        duplicate = history.iloc[[0]].copy()
        duplicate["trackman_id"] = int(history["trackman_id"].max()) + 1
        invalid = history.iloc[[1]].copy()
        invalid["trackman_id"] = int(history["trackman_id"].max()) + 2
        invalid["balls_before"] = 4
        noisy = pd.concat([history, duplicate, invalid], ignore_index=True)

        result = build_strict_game_alignment(main, noisy)
        self.assertEqual(result.audit["content_duplicate_rows_removed"], 1)
        self.assertEqual(result.audit["invalid_context_rows_removed"], 1)
        self.assertEqual(result.audit["strict_matched_games"], 4)


class StrictIdentityMapTests(unittest.TestCase):
    def test_count_and_purity_thresholds_are_both_enforced(self) -> None:
        rows: list[dict[str, object]] = []
        for main_id, trackman_id, count in (
            ("pass", "X", 20),
            ("low_count", "Y", 19),
            ("impure", "Z", 20),
            ("impure", "OTHER", 1),
        ):
            rows.extend(
                {
                    "main_season": 2021,
                    "trackman_season": 2021,
                    "main_pitcher_id": main_id,
                    "trackman_pitcher_id": trackman_id,
                }
                for _ in range(count)
            )
        mapped = build_high_purity_id_map(
            pd.DataFrame(rows), target_season=2022, role="pitcher"
        )
        self.assertEqual(mapped["main_id"].tolist(), ["pass"])
        self.assertEqual(int(mapped.loc[0, "top_count"]), 20)
        self.assertEqual(float(mapped.loc[0, "purity"]), 1.0)

    def test_target_season_alignment_is_refused(self) -> None:
        aligned = pd.DataFrame(
            {
                "main_season": [2021],
                "trackman_season": [2022],
                "main_pitcher_id": ["A"],
                "trackman_pitcher_id": ["P"],
            }
        )
        with self.assertRaises(TrackmanTemporalLeakageError):
            build_high_purity_id_map(
                aligned,
                target_season=2022,
                role="pitcher",
                min_top_count=1,
            )


class StrictHistoryAndLookupTests(unittest.TestCase):
    def test_public_summary_refuses_target_season_log(self) -> None:
        _main, history = strict_fixture()
        with self.assertRaises(TrackmanTemporalLeakageError):
            summarize_strict_trackman_history(
                history.loc[history["season"].le(2022)],
                target_season=2022,
                entity_column="pitcher_trackman_id",
            )

        summary = summarize_strict_trackman_history(
            history.loc[history["season"].lt(2022)],
            target_season=2022,
            entity_column="pitcher_trackman_id",
        )
        self.assertEqual(int(summary.loc[0, "history_count"]), 72)

    def test_builder_uses_only_strictly_prior_physics_and_exposes_oov(self) -> None:
        main, history = strict_fixture()
        builder = StrictTrackmanFeatureBuilder().fit(
            main, history, target_seasons=[2022]
        )
        evaluation = pd.DataFrame(
            {
                "row_id": ["known", "unknown"],
                "season": [2022, 2022],
                "pitcher_id": ["100", "never-seen"],
                "batter_id": ["200", "never-seen"],
            }
        )
        features = builder.transform(evaluation)

        prior = history.loc[history["season"].lt(2022), "rel_speed"]
        all_history = history["rel_speed"]
        self.assertAlmostEqual(
            features.loc[0, "tm_pitcher_rel_speed_mean"], float(prior.mean())
        )
        self.assertNotAlmostEqual(
            features.loc[0, "tm_pitcher_rel_speed_mean"],
            float(all_history.mean()),
        )
        self.assertEqual(features.loc[0, "tm_pitcher_history_count"], 72.0)
        self.assertEqual(features.loc[0, "tm_pitcher_mapping_top_count"], 72.0)
        for role in ("pitcher", "batter"):
            self.assertEqual(features.loc[0, f"tm_{role}_id_mapped"], 1.0)
            self.assertEqual(features.loc[0, f"tm_{role}_id_oov"], 0.0)
            self.assertEqual(features.loc[1, f"tm_{role}_id_mapped"], 0.0)
            self.assertEqual(features.loc[1, f"tm_{role}_id_oov"], 1.0)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertEqual(tuple(features.columns), STRICT_TRACKMAN_FEATURE_COLUMNS)

    def test_transform_is_row_local_and_state_round_trip_is_exact(self) -> None:
        main, history = strict_fixture()
        builder = StrictTrackmanFeatureBuilder().fit(
            main, history, target_seasons=[2022]
        )
        evaluation = pd.DataFrame(
            {
                "row_id": ["a", "b", "c"],
                "season": [2022, 2022, 2022],
                "pitcher_id": ["100", "unknown", "100"],
                "batter_id": ["200", "unknown", "200"],
            }
        )
        expected = builder.transform(evaluation)
        reversed_result = builder.transform(evaluation.iloc[::-1]).sort_index()
        assert_frame_equal(expected, reversed_result)
        for index in evaluation.index:
            singleton = builder.transform(evaluation.loc[[index]])
            assert_frame_equal(singleton, expected.loc[[index]])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict_trackman_state.json"
            builder.save(path)
            loaded = StrictTrackmanFeatureBuilder.load(path)
            assert_frame_equal(expected, loaded.transform(evaluation))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["target_seasons"], [2022])

    def test_transform_rejects_season_without_frozen_lookup(self) -> None:
        main, history = strict_fixture()
        builder = StrictTrackmanFeatureBuilder().fit(
            main, history, target_seasons=[2022]
        )
        evaluation = pd.DataFrame(
            {
                "row_id": ["future"],
                "season": [2023],
                "pitcher_id": ["100"],
                "batter_id": ["200"],
            }
        )
        with self.assertRaises(StrictTrackmanError):
            builder.transform(evaluation)


if __name__ == "__main__":
    unittest.main()
