from __future__ import annotations

import numpy as np
import torch

from lg_control.rate_basis_features import RateBasisAugmentedPreprocessor
from lg_control.r_specialist import RSpecialistConfig, RSpecialistTabM, predict_r_specialist
from tests.test_preprocessing import make_frame


def _fixture():
    frame = make_frame()
    preprocessor = RateBasisAugmentedPreprocessor(min_category_frequency=1).fit(frame)
    arrays = preprocessor.transform(frame)
    model = RSpecialistTabM(
        preprocessor.feature_layout(),
        RSpecialistConfig(hidden_dim=16, ensemble_size=2, hidden_layers=1, dropout=0.0),
    ).eval()
    return model, arrays


def test_r_specialist_outputs_are_valid_exclusive_probabilities() -> None:
    model, arrays = _fixture()
    output = predict_r_specialist(
        model, arrays["categorical"], arrays["numerical"], batch_size=3
    )
    outcome = np.column_stack(
        [
            output["outcome_control_success"],
            output["outcome_reverse_only"],
            output["outcome_middle_only"],
            output["outcome_reverse_and_middle"],
            output["outcome_other_failure"],
        ]
    )
    np.testing.assert_allclose(outcome.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(outcome).all()
    assert ((outcome >= 0.0) & (outcome <= 1.0)).all()


def test_r_specialist_is_batch_and_order_independent() -> None:
    model, arrays = _fixture()
    full = predict_r_specialist(
        model, arrays["categorical"], arrays["numerical"], batch_size=len(arrays["categorical"])
    )["outcome_control_success"]
    singleton = np.concatenate(
        [
            predict_r_specialist(
                model,
                arrays["categorical"][index : index + 1],
                arrays["numerical"][index : index + 1],
                batch_size=1,
            )["outcome_control_success"]
            for index in range(len(full))
        ]
    )
    np.testing.assert_allclose(full, singleton, atol=1e-7)
    permutation = np.array([4, 0, 5, 2, 1, 3])
    shuffled = predict_r_specialist(
        model,
        arrays["categorical"][permutation],
        arrays["numerical"][permutation],
        batch_size=2,
    )["outcome_control_success"]
    restored = np.empty_like(shuffled)
    restored[permutation] = shuffled
    np.testing.assert_allclose(full, restored, atol=1e-7)


def test_r_specialist_config_round_trip() -> None:
    model, _ = _fixture()
    restored = RSpecialistTabM.from_config_dict(model.to_config_dict())
    assert restored.config == model.config
    assert isinstance(restored, torch.nn.Module)
