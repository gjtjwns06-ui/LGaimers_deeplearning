from __future__ import annotations

import numpy as np

from lg_control.exclusive_outcome import (
    ExclusiveOutcomeGatedCrossModel,
    ExclusiveOutcomeHeadConfig,
)
from lg_control.rate_basis_features import RateBasisAugmentedPreprocessor
from lg_control.recent_regime_final import (
    LoadedRecentRegimeArtifact,
    predict_recent_regime,
    predict_recent_regime_heads,
    selected_model_config,
    selected_policy,
)
from tests.test_preprocessing import make_frame


def _artifact():
    train = make_frame()
    preprocessor = RateBasisAugmentedPreprocessor(min_category_frequency=1).fit(train)
    model = ExclusiveOutcomeGatedCrossModel(
        preprocessor.feature_layout(),
        selected_model_config(),
        ExclusiveOutcomeHeadConfig(
            hidden_dim=64, dropout=0.10, use_ball_strike_auxiliary=True
        ),
    ).eval()
    return LoadedRecentRegimeArtifact(
        directory="memory",
        preprocessor=preprocessor,
        models=(model,),
        manifest={"obsolete_pre_2023_f_training_rows": 0},
        training_record={},
    ), train.drop(columns="control_success")


def test_selected_policy_never_uses_old_f() -> None:
    policy = selected_policy()
    assert policy.history_policy == "stable_r_plus_recent"
    assert policy.regime_start_year == 2023
    assert policy.recent_decay == 1.0


def test_recent_artifact_prediction_is_batch_and_order_independent() -> None:
    artifact, frame = _artifact()
    full = predict_recent_regime(artifact, frame, batch_size=len(frame))
    small_batches = predict_recent_regime(artifact, frame, batch_size=2)
    singleton = np.concatenate(
        [
            predict_recent_regime(artifact, frame.iloc[index : index + 1], batch_size=1)
            for index in range(len(frame))
        ]
    )
    np.testing.assert_allclose(full, small_batches, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(full, singleton, rtol=1e-6, atol=1e-7)

    permutation = np.array([4, 0, 5, 2, 1, 3])
    shuffled = predict_recent_regime(artifact, frame.iloc[permutation], batch_size=3)
    restored = np.empty_like(shuffled)
    restored[permutation] = shuffled
    np.testing.assert_allclose(full, restored, rtol=1e-6, atol=1e-7)


def test_recent_heads_match_scalar_success_and_sum_to_one() -> None:
    artifact, frame = _artifact()
    scalar = predict_recent_regime(artifact, frame, batch_size=2)
    heads = predict_recent_regime_heads(artifact, frame, batch_size=3)
    np.testing.assert_allclose(
        scalar, heads["outcome_control_success"], rtol=1e-6, atol=1e-7
    )
    outcome = np.column_stack(
        [
            heads["outcome_control_success"],
            heads["outcome_reverse_only"],
            heads["outcome_middle_only"],
            heads["outcome_reverse_and_middle"],
            heads["outcome_other_failure"],
        ]
    )
    np.testing.assert_allclose(outcome.sum(axis=1), 1.0, atol=1e-6)
