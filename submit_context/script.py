#!/usr/bin/env python3
"""Run the sealed single C2 context-gated model on the evaluation server."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_ROOT = PACKAGE_ROOT / "model"
RUNTIME_ROOT = LOCAL_MODEL_ROOT / "runtime"
RUNTIME_SRC_ROOT = RUNTIME_ROOT / "src"
DEPLOYMENT_MANIFEST_PATH = LOCAL_MODEL_ROOT / "deployment_manifest.json"
BUNDLE_DIRECTORY = LOCAL_MODEL_ROOT / "bundle"
DEPLOYMENT_MANIFEST_KIND = "final_context_c2_submission_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_deployment_files() -> None:
    if not DEPLOYMENT_MANIFEST_PATH.is_file() or DEPLOYMENT_MANIFEST_PATH.is_symlink():
        raise FileNotFoundError("sealed deployment manifest is missing")
    payload = json.loads(DEPLOYMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("deployment manifest root must be an object")
    observed_hash = payload.get("deployment_manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("deployment_manifest_sha256", None)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != DEPLOYMENT_MANIFEST_KIND
        or not isinstance(observed_hash, str)
        or observed_hash != _canonical_sha256(unhashed)
    ):
        raise ValueError("deployment manifest identity differs")
    expected = payload.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("deployment manifest file inventory is missing")

    observed_paths = {"script.py", "requirements.txt"}
    for path in LOCAL_MODEL_ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"deployment contains a symlink: {path}")
        if path.is_file() and path != DEPLOYMENT_MANIFEST_PATH:
            observed_paths.add(path.relative_to(PACKAGE_ROOT).as_posix())
    if set(expected) != observed_paths:
        raise ValueError("deployment file inventory differs")
    for relative, expected_hash in expected.items():
        candidate = (PACKAGE_ROOT / relative).resolve()
        if PACKAGE_ROOT not in candidate.parents or not candidate.is_file():
            raise ValueError(f"unsafe or missing deployment file: {relative}")
        if _sha256_file(candidate) != expected_hash:
            raise ValueError(f"deployment file hash differs: {relative}")


verify_deployment_files()
if str(RUNTIME_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC_ROOT))

import pandas as pd

from lg_control.constants import ID_COL, TARGET_COL
from lg_control.final_context_training import (
    load_final_context_artifact,
    predict_final_context,
)
from lg_control.schema import validate_input_frame, validate_predictions


OUTPUT_PATH = Path("output") / "submission.csv"
INFERENCE_BATCH_SIZE = 16_384


def resolve_test_path() -> Path:
    preferred = Path("data") / "test.csv"
    fallback = Path("open") / "test.csv"
    for path in (preferred, fallback):
        if path.exists():
            if not path.is_file():
                raise FileNotFoundError(f"test path is not a file: {path}")
            return path
    raise FileNotFoundError("test.csv was not found at ./data/test.csv or ./open/test.csv")


def load_test_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    validate_input_frame(frame, require_target=False, allow_target=False)
    if frame.empty or not frame[ID_COL].is_unique:
        raise ValueError("test row_id values must be nonempty and unique")
    return frame


def write_submission(frame: pd.DataFrame, probabilities) -> None:
    values = validate_predictions(frame[ID_COL], probabilities)
    submission = pd.DataFrame(
        {ID_COL: frame[ID_COL].copy(), TARGET_COL: values},
        columns=[ID_COL, TARGET_COL],
    )
    if not submission[ID_COL].equals(frame[ID_COL]):
        raise RuntimeError("submission row order differs from test.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix="submission_",
            suffix=".csv.tmp",
            dir=OUTPUT_PATH.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            submission.to_csv(temporary, index=False, lineterminator="\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, OUTPUT_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    test_path = resolve_test_path()
    frame = load_test_frame(test_path)
    loaded = load_final_context_artifact(BUNDLE_DIRECTORY, verify_code=True)
    probabilities = predict_final_context(
        loaded,
        frame,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    write_submission(frame, probabilities)
    print(f"Saved {OUTPUT_PATH} ({len(frame):,} rows, single C2 neutral model)")


if __name__ == "__main__":
    main()
