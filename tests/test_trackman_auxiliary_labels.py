from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.lg_control.trackman import METRIC_COLUMNS
from src.lg_control.trackman_auxiliary_labels import (
    FoldTrainTrackmanAuxiliaryBuilder,
    TrackmanAuxiliaryBoundaryError,
)
from tests.test_strict_trackman import strict_fixture


def test_fold_boundary_row_order_masks_and_pitch_distribution(tmp_path) -> None:
    main, history = strict_fixture()
    fold_train = main.loc[main["season"].lt(2022)].copy()
    fold_train.index = np.arange(500, 500 + len(fold_train))
    history.loc[
        history["trackman_id"].eq(2), METRIC_COLUMNS[0]
    ] = np.nan

    builder = FoldTrainTrackmanAuxiliaryBuilder(validation_year=2022)
    labels = builder.fit_transform(fold_train, history)

    assert labels.row_ids == tuple(fold_train["row_id"].astype(str))
    assert labels.pitch_type_mask.sum() == 72
    assert labels.pitch_type_targets.shape == (72,)
    assert labels.continuous_targets.shape == (72, len(METRIC_COLUMNS))
    assert labels.continuous_mask[:, 0].sum() == 71
    missing_position = int(np.flatnonzero(~labels.continuous_mask[:, 0])[0])
    assert labels.continuous_targets[missing_position, 0] == 0.0
    assert labels.audit["trackman_boundary_excluded_rows"] == 24
    assert labels.audit["pitch_type_counts"] == {
        "fastball": 18,
        "breaking": 18,
        "offspeed": 18,
        "other": 18,
    }

    state_path = tmp_path / "aux_state.json"
    builder.save(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["validation_year"] == 2022
    assert state["audit"]["label_fingerprint_sha256"] == labels.audit[
        "label_fingerprint_sha256"
    ]


def test_validation_main_rows_are_rejected() -> None:
    main, history = strict_fixture()
    with pytest.raises(TrackmanAuxiliaryBoundaryError, match="validation/future"):
        FoldTrainTrackmanAuxiliaryBuilder(validation_year=2022).fit_transform(
            main, history
        )


def test_strict_mismatch_yields_missing_labels_and_zero_targets() -> None:
    main, history = strict_fixture()
    fold_train = main.loc[main["season"].lt(2022)].copy()
    fold_train.loc[fold_train.index[1], "game_dayofweek"] = 3

    labels = FoldTrainTrackmanAuxiliaryBuilder(
        validation_year=2022
    ).fit_transform(fold_train, history)

    first_game = fold_train["season"].eq(2019).to_numpy()
    assert not labels.pitch_type_mask[first_game].any()
    assert (labels.pitch_type_targets[first_game] == -1).all()
    assert not labels.continuous_mask[first_game].any()
    assert (labels.continuous_targets[first_game] == 0.0).all()
    assert labels.audit["alignment_row_calendar_mismatch_games_rejected"] == 1


def test_history_shuffle_is_deterministic() -> None:
    main, history = strict_fixture()
    fold_train = main.loc[main["season"].lt(2022)].copy()
    first = FoldTrainTrackmanAuxiliaryBuilder(validation_year=2022).fit_transform(
        fold_train, history
    )
    second = FoldTrainTrackmanAuxiliaryBuilder(validation_year=2022).fit_transform(
        fold_train, history.sample(frac=1.0, random_state=91).reset_index(drop=True)
    )

    assert first.audit["label_fingerprint_sha256"] == second.audit[
        "label_fingerprint_sha256"
    ]
    np.testing.assert_array_equal(first.pitch_type_targets, second.pitch_type_targets)
    np.testing.assert_allclose(first.continuous_targets, second.continuous_targets)
    np.testing.assert_array_equal(first.continuous_mask, second.continuous_mask)
