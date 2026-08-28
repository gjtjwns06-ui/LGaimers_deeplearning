from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lg_control.combined_complement import (
    CombinedRComplementArtifact,
    predict_combined_r_complement,
)


class _Preprocessor:
    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "categorical": np.zeros((len(frame), 1), dtype=np.int64),
            "numerical": frame[["specialist_logit"]].to_numpy(np.float32),
        }


class _Model(torch.nn.Module):
    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        return numerical[:, 0]


def _artifact() -> CombinedRComplementArtifact:
    return CombinedRComplementArtifact(
        preprocessor=_Preprocessor(),  # type: ignore[arg-type]
        model=_Model(),  # type: ignore[arg-type]
        gate={
            "columns": ["uncertainty", "direction"],
            "cap": 0.3,
            "global_weight": 0.0,
            "groups": {"far|up": 0.3, "u050|down": 0.1},
        },
    )


def test_combined_complement_applies_only_selected_r_cells() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["R", "R", "R", "F"],
            "specialist_logit": [0.4, -0.4, 0.4, 0.4],
        }
    )
    base = np.array([0.56, 0.54, 0.51, 0.56])
    probability, specialist, weight = predict_combined_r_complement(
        _artifact(), frame, base, batch_size=4
    )
    assert weight.tolist() == [0.3, 0.1, 0.0, 0.0]
    assert probability[0] > base[0]
    assert probability[1] < base[1]
    assert probability[2] == base[2]
    assert probability[3] == base[3]
    assert np.isfinite(specialist).all()


def test_combined_complement_is_batch_and_order_independent() -> None:
    frame = pd.DataFrame(
        {"game_type": ["R", "F", "R"], "specialist_logit": [0.2, -0.2, 0.5]}
    )
    base = np.array([0.57, 0.43, 0.52])
    full = predict_combined_r_complement(_artifact(), frame, base, batch_size=3)[0]
    order = np.array([2, 0, 1])
    shuffled = predict_combined_r_complement(
        _artifact(), frame.iloc[order].reset_index(drop=True), base[order], batch_size=1
    )[0]
    assert np.allclose(shuffled, full[order])
