from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lg_control.recent_regime import (
    RecentRegimePolicy,
    recent_regime_audit,
    recent_regime_weights,
    select_recent_regime_inner_partitions,
    select_recent_regime_training_rows,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "game_type": ["R", "F", "R", "F", "R", "F", "R", "F"],
            "control_success": [0, 1, 1, 1, 0, 0, 1, 0],
        }
    )


def test_stable_r_policy_hard_excludes_old_f_and_future() -> None:
    policy = RecentRegimePolicy(history_policy="stable_r_plus_recent")
    selected = select_recent_regime_training_rows(
        _frame(), cutoff_year=2024, policy=policy
    )
    assert list(zip(selected.season, selected.game_type, strict=True)) == [
        (2021, "R"),
        (2022, "R"),
        (2023, "R"),
        (2023, "F"),
    ]
    assert not ((selected.season < 2023) & selected.game_type.eq("F")).any()


def test_recent_only_uses_only_new_regime() -> None:
    policy = RecentRegimePolicy(history_policy="recent_only")
    selected = select_recent_regime_training_rows(
        _frame(), cutoff_year=2025, policy=policy
    )
    assert selected.season.tolist() == [2023, 2023, 2024, 2024]


def test_weights_reject_obsolete_f_and_emphasize_recent_rows() -> None:
    policy = RecentRegimePolicy(
        history_policy="stable_r_plus_recent",
        historical_r_decay=0.5,
        recent_decay=0.8,
    )
    selected = select_recent_regime_training_rows(
        _frame(), cutoff_year=2025, policy=policy
    )
    weights = recent_regime_weights(selected, cutoff_year=2025, policy=policy)
    assert np.isclose(weights.mean(), 1.0)
    row_weight = dict(zip(selected.index, weights, strict=True))
    assert row_weight[6] > row_weight[4] > row_weight[2] > row_weight[0]

    with pytest.raises(ValueError, match="obsolete"):
        recent_regime_weights(_frame(), cutoff_year=2025, policy=policy)


def test_audit_proves_zero_old_f_exposure() -> None:
    policy = RecentRegimePolicy()
    selected = select_recent_regime_training_rows(
        _frame(), cutoff_year=2025, policy=policy
    )
    weights = recent_regime_weights(selected, cutoff_year=2025, policy=policy)
    audit = recent_regime_audit(selected, weights, policy=policy)
    assert audit["obsolete_pre_regime_f_rows"] == 0
    assert "F" not in audit["by_season_game_type"]["2022"]


@pytest.mark.parametrize(
    ("history_policy", "expected_fit_indices"),
    [
        ("recent_only", [4]),
        ("stable_r_plus_recent", [0, 2, 4]),
    ],
)
def test_inner_split_uses_early_regime_and_never_old_f(
    history_policy: str, expected_fit_indices: list[int]
) -> None:
    frame = _frame().copy()
    frame["game_month"] = [5, 5, 6, 6, 6, 8, 7, 9]
    policy = RecentRegimePolicy(history_policy=history_policy)
    fit, stop = select_recent_regime_inner_partitions(
        frame,
        validation_year=2023,
        validation_start_month=7,
        policy=policy,
    )
    assert fit.index.tolist() == expected_fit_indices
    assert stop.index.tolist() == [5]
    assert not ((fit.season < 2023) & fit.game_type.eq("F")).any()
