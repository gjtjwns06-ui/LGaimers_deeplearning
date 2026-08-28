#!/usr/bin/env python3
"""Package and smoke-test the conditional v17 + recent-DL submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "scripts/complementary_submission_entrypoint.py"
RUNTIME_FILES = (
    "src/lg_control/__init__.py",
    "src/lg_control/auxiliary_targets.py",
    "src/lg_control/complementary_gate.py",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v17-dir",
        type=Path,
        default=PROJECT_ROOT.parent
        / "LG-aimers-v11-independent-research/submission_v17_count_hand",
    )
    parser.add_argument(
        "--dl-bundle",
        type=Path,
        default=PROJECT_ROOT / "artifacts/final_recent_regime",
    )
    parser.add_argument(
        "--gate-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/complementary_hybrid_v1/gate",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "artifacts/complementary_hybrid_v1/selection.json",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=PROJECT_ROOT / "submit_complementary_hybrid",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "submit_complementary_hybrid.zip",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_runtime(staging: Path) -> None:
    runtime_root = staging / "model/dl_runtime"
    for relative in RUNTIME_FILES:
        source = PROJECT_ROOT / relative
        target = runtime_root / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_tree(staging: Path, args: argparse.Namespace) -> None:
    shutil.copy2(ENTRYPOINT, staging / "script.py")
    (staging / "requirements.txt").write_text(
        "lightgbm==4.6.0\n"
        "catboost==1.2.10\n"
        "# torch, pandas, and numpy are supplied by the official image.\n",
        encoding="utf-8",
    )
    model_root = staging / "model"
    model_root.mkdir()
    shutil.copytree(
        args.v17_dir.resolve(),
        model_root / "v17",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data", "output"),
    )
    shutil.copytree(args.dl_bundle.resolve(), model_root / "dl_bundle")
    shutil.copytree(args.gate_dir.resolve(), model_root / "gate")
    _copy_runtime(staging)

    inventory = {
        "script.py": _sha256(staging / "script.py"),
        "requirements.txt": _sha256(staging / "requirements.txt"),
    }
    for path in sorted(model_root.rglob("*")):
        if path.is_file():
            inventory[path.relative_to(staging).as_posix()] = _sha256(path)
    (model_root / "deployment_manifest.json").write_text(
        json.dumps(
            {
                "kind": "v17_recent_dl_conditional_submission_v1",
                "files_sha256": inventory,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _safe_names(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate paths")
    for name in names:
        path = PurePosixPath(name.rstrip("/"))
        if not name or name.startswith("/") or "\\" in name or ".." in path.parts:
            raise ValueError(f"unsafe ZIP path: {name}")


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
        required = {
            "model/deployment_manifest.json",
            "model/gate/gate.csv",
            "model/gate/gate_config.json",
            "model/v17/script.py",
            "model/dl_bundle/manifest.json",
        }
        missing = required - set(names)
        if missing:
            raise ValueError(f"ZIP is missing files: {sorted(missing)}")
        if any(name.startswith(("data/", "open/", "output/")) for name in names):
            raise ValueError("ZIP contains input/output data")


def _smoke(staging: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="conditional_hybrid_smoke_") as temporary:
        root = Path(temporary)
        shutil.copytree(staging, root, dirs_exist_ok=True)
        (root / "data").mkdir()
        test = pd.read_csv(PROJECT_ROOT / "data/test.csv", nrows=64)
        test.to_csv(root / "data/test.csv", index=False)
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
                "submission smoke failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        output = pd.read_csv(root / "output/submission.csv")
        if output.columns.tolist() != ["row_id", "control_success"]:
            raise ValueError("smoke output columns differ")
        if not output["row_id"].equals(test["row_id"]):
            raise ValueError("smoke output row identity differs")
        values = output["control_success"].to_numpy(dtype=float)
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise ValueError("smoke probabilities are outside [0, 1]")
        return {
            "rows": int(len(output)),
            "prediction_mean": float(values.mean()),
            "stdout": completed.stdout.strip(),
        }


def main() -> None:
    args = parse_args()
    for path in (
        args.v17_dir,
        args.dl_bundle,
        args.gate_dir,
        args.selection,
        ENTRYPOINT,
        PROJECT_ROOT / "data/test.csv",
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    if selection.get("decision") != "PASS":
        raise RuntimeError("conditional gate did not pass its temporal audit")
    if args.submission_dir.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.submission_dir.resolve()}")
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output.resolve()}")

    with tempfile.TemporaryDirectory(prefix="conditional_hybrid_package_") as temporary:
        staging = Path(temporary)
        _copy_tree(staging, args)
        smoke = _smoke(staging)
        shutil.copytree(staging, args.submission_dir.resolve())
        _write_zip(staging, args.output.resolve())
    _validate_zip(args.output.resolve())
    digest = _sha256(args.output.resolve())
    hash_path = args.output.resolve().with_suffix(args.output.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "submission_dir": str(args.submission_dir.resolve()),
                "zip": str(args.output.resolve()),
                "sha256": digest,
                "bytes": args.output.resolve().stat().st_size,
                "smoke": smoke,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
