#!/usr/bin/env python3
"""Inference entrypoint for the fixed post-2022 regime ensemble."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "model"
BUNDLE_ROOT = MODEL_ROOT / "bundle"
RUNTIME_SRC = MODEL_ROOT / "runtime" / "src"
DEPLOYMENT_MANIFEST = MODEL_ROOT / "deployment_manifest.json"
OUTPUT_PATH = Path("output") / "submission.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_deployment() -> None:
    payload = json.loads(DEPLOYMENT_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("kind") != "recent_regime_submission_v1":
        raise ValueError("deployment kind differs")
    expected = payload.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("deployment inventory is missing")
    observed = {"script.py", "requirements.txt"}
    for path in MODEL_ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"deployment contains a symlink: {path}")
        if path.is_file() and path != DEPLOYMENT_MANIFEST:
            observed.add(path.relative_to(PACKAGE_ROOT).as_posix())
    if observed != set(expected):
        raise ValueError("deployment inventory differs")
    for relative, digest in expected.items():
        path = (PACKAGE_ROOT / relative).resolve()
        if PACKAGE_ROOT not in path.parents or not path.is_file():
            raise ValueError(f"unsafe deployment file: {relative}")
        if _sha256(path) != digest:
            raise ValueError(f"deployment hash mismatch: {relative}")


verify_deployment()
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

import pandas as pd
import torch

from lg_control.constants import ID_COL, TARGET_COL
from lg_control.recent_regime_final import (
    load_recent_regime_artifact,
    predict_recent_regime,
)
from lg_control.schema import validate_input_frame, validate_predictions


def resolve_test_path() -> Path:
    for path in (Path("data/test.csv"), Path("open/test.csv")):
        if path.is_file():
            return path
    raise FileNotFoundError("test.csv was not found")


def write_submission(frame: pd.DataFrame, probabilities) -> None:
    values = validate_predictions(frame[ID_COL], probabilities)
    output = pd.DataFrame({ID_COL: frame[ID_COL].copy(), TARGET_COL: values})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix="recent_regime_",
            suffix=".csv.tmp",
            dir=OUTPUT_PATH.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            output.to_csv(temporary, index=False, lineterminator="\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, OUTPUT_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    frame = pd.read_csv(resolve_test_path(), encoding="utf-8-sig", low_memory=False)
    validate_input_frame(frame, require_target=False, allow_target=False)
    if frame.empty or not frame[ID_COL].is_unique:
        raise ValueError("test row IDs must be nonempty and unique")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact = load_recent_regime_artifact(BUNDLE_ROOT, device=device)
    probability = predict_recent_regime(
        artifact, frame, batch_size=32768, device=device
    )
    write_submission(frame, probability)
    print(
        f"Saved {OUTPUT_PATH} ({len(frame):,} rows, "
        f"{len(artifact.models)} recent-regime model(s))"
    )


if __name__ == "__main__":
    main()
