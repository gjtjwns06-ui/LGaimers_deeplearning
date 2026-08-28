#!/usr/bin/env python3
"""Package, verify, and smoke-test the post-2022 regime submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.recent_regime_final import load_recent_regime_artifact


SUBMIT_SOURCE = PROJECT_ROOT / "submit_recent_regime"
RUNTIME_FILES = (
    "src/lg_control/__init__.py",
    "src/lg_control/auxiliary_targets.py",
    "src/lg_control/constants.py",
    "src/lg_control/exclusive_outcome.py",
    "src/lg_control/metrics.py",
    "src/lg_control/model.py",
    "src/lg_control/preprocessing.py",
    "src/lg_control/rate_basis_features.py",
    "src/lg_control/recent_regime.py",
    "src/lg_control/recent_regime_final.py",
    "src/lg_control/regime_model.py",
    "src/lg_control/schema.py",
    "src/lg_control/training.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_names(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate paths")
    for name in names:
        path = PurePosixPath(name.rstrip("/"))
        if not name or name.startswith("/") or "\\" in name or ".." in path.parts:
            raise ValueError(f"unsafe ZIP path: {name}")


def _copy_tree(staging: Path, bundle: Path) -> None:
    shutil.copy2(SUBMIT_SOURCE / "script.py", staging / "script.py")
    shutil.copy2(SUBMIT_SOURCE / "requirements.txt", staging / "requirements.txt")
    model_root = staging / "model"
    model_root.mkdir()
    shutil.copytree(bundle, model_root / "bundle")
    for relative in RUNTIME_FILES:
        source = PROJECT_ROOT / relative
        target = model_root / "runtime" / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    inventory = {
        "script.py": _sha256(staging / "script.py"),
        "requirements.txt": _sha256(staging / "requirements.txt"),
    }
    for path in sorted(model_root.rglob("*")):
        if path.is_file():
            inventory[path.relative_to(staging).as_posix()] = _sha256(path)
    (model_root / "deployment_manifest.json").write_text(
        json.dumps(
            {"kind": "recent_regime_submission_v1", "files_sha256": inventory},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_zip(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging).as_posix())


def _validate_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError("submission ZIP is invalid")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _safe_names(names)
        top = {name.split("/", 1)[0] for name in names}
        if top != {"script.py", "requirements.txt", "model"}:
            raise ValueError(f"ZIP top-level entries differ: {sorted(top)}")
        if "model/deployment_manifest.json" not in names:
            raise ValueError("deployment manifest is missing")
        if "model/bundle/manifest.json" not in names:
            raise ValueError("bundle manifest is missing")
        if any(name.startswith(("data/", "open/", "output/")) for name in names):
            raise ValueError("ZIP contains input/output data")


def _smoke(staging: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="recent_regime_smoke_") as temporary:
        root = Path(temporary)
        shutil.copytree(staging, root, dirs_exist_ok=True)
        (root / "data").mkdir()
        shutil.copy2(PROJECT_ROOT / "data" / "test.csv", root / "data" / "test.csv")
        completed = subprocess.run(
            [sys.executable, "script.py"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"submission smoke failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        import pandas as pd

        test = pd.read_csv(root / "data" / "test.csv")
        output = pd.read_csv(root / "output" / "submission.csv")
        if output.columns.tolist() != ["row_id", "control_success"]:
            raise ValueError("smoke output columns differ")
        if not output["row_id"].equals(test["row_id"]):
            raise ValueError("smoke output row identity differs")
        values = output["control_success"].to_numpy(dtype=float)
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise ValueError("smoke probabilities are outside [0, 1]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=PROJECT_ROOT / "artifacts/final_recent_regime"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "submit_recent_regime.zip"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded = load_recent_regime_artifact(args.bundle)
    if not loaded.models:
        raise ValueError("bundle contains no models")
    with tempfile.TemporaryDirectory(prefix="recent_regime_package_") as temporary:
        staging = Path(temporary)
        _copy_tree(staging, args.bundle)
        _smoke(staging)
        _write_zip(staging, args.output)
    _validate_zip(args.output)
    digest = _sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "zip": str(args.output),
                "sha256": digest,
                "bytes": args.output.stat().st_size,
                "models": len(loaded.models),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
