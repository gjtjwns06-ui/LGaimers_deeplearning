from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.lg_control.calibration import AffineLogitCalibrator
from src.lg_control.metrics import brier_score, brier_skill, competition_score


class CalibrationTests(unittest.TestCase):
    def test_fit_improves_biased_probabilities_and_round_trips(self) -> None:
        target = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        raw = np.array([0.30, 0.35, 0.40, 0.60, 0.65, 0.70], dtype=float)
        calibrator = AffineLogitCalibrator().fit(raw, target)
        calibrated = calibrator.transform(raw)
        self.assertLessEqual(brier_score(target, calibrated), brier_score(target, raw))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.json"
            calibrator.save(path)
            restored = AffineLogitCalibrator.load(path)
            np.testing.assert_array_equal(calibrated, restored.transform(raw))

    def test_competition_score_matches_brier_skill(self) -> None:
        target = np.array([0, 0, 1, 1], dtype=float)
        prediction = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        self.assertAlmostEqual(
            competition_score(target, prediction),
            max(0.0, 100_000.0 * brier_skill(target, prediction)),
        )


if __name__ == "__main__":
    unittest.main()

