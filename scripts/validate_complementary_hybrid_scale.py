#!/usr/bin/env python3
"""Validate the shipped conditional-hybrid ZIP at hidden-test row scale."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ZIP = PROJECT_ROOT / "submit_complementary_hybrid.zip"
SUBMISSION_DIR = PROJECT_ROOT / "submit_complementary_hybrid"
TRAIN = PROJECT_ROOT / "data/train.csv"
REPORT = PROJECT_ROOT / "reports/complementary_hybrid_zip_scale.json"
HIDDEN_ROWS = 245_789
SEED = 20260827


def _run(package: Path, test: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    (package / "data").mkdir(exist_ok=True)
    test.to_csv(package / "data/test.csv", index=False, encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=package,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"submission execution failed\n{completed.stdout}\n{completed.stderr[-4000:]}"
        )
    return (
        pd.read_csv(package / "output/submission.csv"),
        elapsed,
        completed.stdout.strip(),
    )


def main() -> None:
    for path in (SUBMISSION_ZIP, SUBMISSION_DIR, TRAIN):
        exists = path.is_file() if path.suffix else path.is_dir()
        if not exists:
            raise FileNotFoundError(path)
    rng = np.random.default_rng(SEED)
    raw = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False).drop(
        columns="control_success"
    )
    indices = rng.choice(len(raw), HIDDEN_ROWS, replace=True)
    test = raw.iloc[indices].reset_index(drop=True)
    test["row_id"] = [f"SCALE_{index:06d}" for index in range(HIDDEN_ROWS)]
    test["season"] = 2025

    with tempfile.TemporaryDirectory(prefix="conditional_hybrid_scale_") as temporary:
        root = Path(temporary)
        extracted = root / "zip"
        extracted.mkdir()
        with zipfile.ZipFile(SUBMISSION_ZIP) as archive:
            archive.extractall(extracted)
        output_zip, zip_seconds, stdout = _run(extracted, test)
        print(
            f"ZIP full-scale run: {zip_seconds:.2f}s for {HIDDEN_ROWS:,} rows",
            flush=True,
        )

        source = root / "source"
        shutil.copytree(
            SUBMISSION_DIR,
            source,
            ignore=shutil.ignore_patterns("__pycache__", "data", "output"),
        )
        parity_rows = 20_000
        output_source, _, _ = _run(source, test.iloc[:parity_rows].copy())

    prediction = output_zip["control_success"].to_numpy(dtype=np.float64)
    row_id_ok = output_zip["row_id"].astype(str).equals(test["row_id"].astype(str))
    finite = bool(
        np.isfinite(prediction).all()
        and np.all(prediction >= 0.0)
        and np.all(prediction <= 1.0)
    )
    parity = float(
        np.max(
            np.abs(
                output_source["control_success"].to_numpy(float)
                - prediction[:parity_rows]
            )
        )
    )
    gates = {
        "row_id_preserved": bool(row_id_ok),
        "row_count": bool(len(output_zip) == HIDDEN_ROWS),
        "finite_in_unit_interval": finite,
        # The DL predictor uses float32 batched matrix operations.  The full ZIP
        # and 20k source runs therefore differ slightly at batch boundaries.
        "source_zip_parity": bool(parity < 2e-7),
        "within_600s_budget": bool(zip_seconds < 600.0),
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hidden_rows": HIDDEN_ROWS,
        "zip_seconds": round(zip_seconds, 3),
        "rows_out": int(len(output_zip)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "source_zip_parity_max_abs": parity,
        "stdout": stdout,
        **gates,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
