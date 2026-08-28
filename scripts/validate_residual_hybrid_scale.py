#!/usr/bin/env python3
"""Run the v22 residual-hybrid ZIP at the official hidden-test row count."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import validate_complementary_hybrid_scale as scale  # noqa: E402


scale.SUBMISSION_ZIP = PROJECT_ROOT / "hybrid_v22.zip"
scale.SUBMISSION_DIR = PROJECT_ROOT / "submit_residual_hybrid_v22"
scale.REPORT = PROJECT_ROOT / "reports/residual_hybrid_v22_zip_scale.json"


if __name__ == "__main__":
    scale.main()
