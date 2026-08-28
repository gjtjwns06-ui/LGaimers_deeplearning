#!/usr/bin/env python3
"""Package v21 plus the forward-audited combined-signal R complement."""

from __future__ import annotations

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
ENTRYPOINT = PROJECT_ROOT / "scripts/residual_hybrid_submission_entrypoint.py"
V17_DIR = (
    PROJECT_ROOT.parent
    / "LG-aimers-v11-independent-research/submission_v17_count_hand"
)
DL_BUNDLE = PROJECT_ROOT / "artifacts/final_recent_regime"
GATE_DIR = PROJECT_ROOT / "artifacts/complementary_hybrid_v1/gate"
RESIDUAL_DIR = PROJECT_ROOT / "artifacts/uncertain_residual_v1/final_bundle"
RESIDUAL_SELECTION = PROJECT_ROOT / "artifacts/uncertain_residual_v1/selection.json"
APPLICATION_CONFIG = PROJECT_ROOT / "artifacts/uncertain_application_v1/config.json"
F_SHAPE_DIR = PROJECT_ROOT / "artifacts/f_regime_deep_shape_v1"
COMBINED_COMPLEMENT_DIR = PROJECT_ROOT / "artifacts/final_combined_r_complement"
COMBINED_GATE_DIR = PROJECT_ROOT / "artifacts/final_combined_r_gate"
COMBINED_SELECTION = PROJECT_ROOT / "artifacts/combined_r_complement_v1/selection.json"
SUBMISSION_DIR = PROJECT_ROOT / "submit_residual_hybrid_v22"
OUTPUT_ZIP = PROJECT_ROOT / "hybrid_v22.zip"
RUNTIME_FILES = (
    "src/lg_control/__init__.py",
    "src/lg_control/auxiliary_targets.py",
    "src/lg_control/complementary_gate.py",
    "src/lg_control/combined_complement.py",
    "src/lg_control/combined_signal_features.py",
    "src/lg_control/constants.py",
    "src/lg_control/exclusive_outcome.py",
    "src/lg_control/f_regime_residual.py",
    "src/lg_control/f_regime_shape.py",
    "src/lg_control/hybrid_residual.py",
    "src/lg_control/metrics.py",
    "src/lg_control/model.py",
    "src/lg_control/preprocessing.py",
    "src/lg_control/rate_basis_features.py",
    "src/lg_control/recent_regime.py",
    "src/lg_control/recent_regime_final.py",
    "src/lg_control/regime_model.py",
    "src/lg_control/schema.py",
    "src/lg_control/training.py",
    "src/lg_control/temporal_target_features.py",
    "src/lg_control/uncertain_residual.py",
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


def _stage(staging: Path) -> None:
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
        V17_DIR,
        model_root / "v17",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data", "output"),
    )
    shutil.copytree(DL_BUNDLE, model_root / "dl_bundle")
    shutil.copytree(GATE_DIR, model_root / "gate")
    shutil.copytree(RESIDUAL_DIR, model_root / "uncertain_residual")
    shutil.copytree(F_SHAPE_DIR, model_root / "f_regime_shape")
    shutil.copytree(COMBINED_COMPLEMENT_DIR, model_root / "combined_complement")
    shutil.copytree(COMBINED_GATE_DIR, model_root / "combined_r_gate")
    application_root = model_root / "residual_application"
    application_root.mkdir()
    shutil.copy2(APPLICATION_CONFIG, application_root / "config.json")
    for relative in RUNTIME_FILES:
        source = PROJECT_ROOT / relative
        target = model_root / "dl_runtime" / PurePosixPath(relative)
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
            {
                "kind": "v17_recent_dl_uncertain_residual_f_shape_r_complement_submission_v1",
                "files_sha256": inventory,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _smoke(staging: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="residual_hybrid_smoke_") as temporary:
        root = Path(temporary)
        shutil.copytree(staging, root, dirs_exist_ok=True)
        (root / "data").mkdir()
        test = pd.read_csv(PROJECT_ROOT / "data/test.csv")
        test.to_csv(root / "data/test.csv", index=False)
        completed = subprocess.run(
            [sys.executable, "script.py"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"smoke failed\n{completed.stdout}\n{completed.stderr[-4000:]}"
            )
        output = pd.read_csv(root / "output/submission.csv")
        if output.columns.tolist() != ["row_id", "control_success"]:
            raise ValueError("smoke output columns differ")
        if not output["row_id"].equals(test["row_id"]):
            raise ValueError("smoke row IDs differ")
        values = output["control_success"].to_numpy(float)
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise ValueError("smoke probabilities are outside [0,1]")
        return {
            "rows": int(len(output)),
            "prediction_mean": float(values.mean()),
            "stdout": completed.stdout.strip(),
        }


def _write_zip(staging: Path) -> None:
    if OUTPUT_ZIP.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_ZIP}")
    with zipfile.ZipFile(
        OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging).as_posix())


def _validate_zip() -> None:
    if not zipfile.is_zipfile(OUTPUT_ZIP):
        raise ValueError("output is not a valid ZIP")
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        names = archive.namelist()
        _safe_names(names)
        if {name.split("/", 1)[0] for name in names} != {
            "script.py",
            "requirements.txt",
            "model",
        }:
            raise ValueError("ZIP top-level entries differ")
        required = {
            "model/deployment_manifest.json",
            "model/v17/script.py",
            "model/dl_bundle/manifest.json",
            "model/gate/gate_config.json",
            "model/uncertain_residual/manifest.json",
            "model/residual_application/config.json",
            "model/f_regime_shape/artifact.json",
            "model/combined_complement/manifest.json",
            "model/combined_r_gate/artifact.json",
        }
        missing = required - set(names)
        if missing:
            raise ValueError(f"ZIP is missing files: {sorted(missing)}")


def main() -> None:
    required = [
        ENTRYPOINT,
        V17_DIR,
        DL_BUNDLE,
        GATE_DIR,
        RESIDUAL_DIR,
        RESIDUAL_SELECTION,
        APPLICATION_CONFIG,
        F_SHAPE_DIR,
        COMBINED_COMPLEMENT_DIR,
        COMBINED_GATE_DIR,
        COMBINED_SELECTION,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    selection = json.loads(RESIDUAL_SELECTION.read_text(encoding="utf-8"))
    if selection.get("decision") != "PASS":
        raise RuntimeError("uncertain residual did not pass its forward audit")
    application = json.loads(APPLICATION_CONFIG.read_text(encoding="utf-8"))
    if application.get("decision") != "PASS":
        raise RuntimeError("uncertain residual application gate did not pass")
    f_shape = json.loads((F_SHAPE_DIR / "artifact.json").read_text(encoding="utf-8"))
    audit = f_shape.get("selection_contract", {})
    if (
        f_shape.get("kind") != "f_regime_deep_shape_v1"
        or not bool(audit.get("audit_months_all_improved"))
        or float(audit.get("audit_delta_brier", 0.0)) >= -1e-6
    ):
        raise RuntimeError("F regime shape correction did not pass its forward audit")
    complement = json.loads(COMBINED_SELECTION.read_text(encoding="utf-8"))
    if (
        complement.get("decision") != "PASS"
        or float(complement.get("audit_delta", 0.0)) >= -1e-6
        or not all(float(row["delta"]) < 0.0 for row in complement.get("audit_months", []))
    ):
        raise RuntimeError("combined R complement did not pass its forward audit")
    if SUBMISSION_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {SUBMISSION_DIR}")
    if OUTPUT_ZIP.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_ZIP}")

    with tempfile.TemporaryDirectory(prefix="residual_hybrid_package_") as temporary:
        staging = Path(temporary)
        _stage(staging)
        smoke = _smoke(staging)
        shutil.copytree(staging, SUBMISSION_DIR)
        _write_zip(staging)
    _validate_zip()
    digest = _sha256(OUTPUT_ZIP)
    OUTPUT_ZIP.with_suffix(OUTPUT_ZIP.suffix + ".sha256").write_text(
        f"{digest}  {OUTPUT_ZIP.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "zip": str(OUTPUT_ZIP),
                "filename_length": len(OUTPUT_ZIP.name),
                "bytes": OUTPUT_ZIP.stat().st_size,
                "sha256": digest,
                "smoke": smoke,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
