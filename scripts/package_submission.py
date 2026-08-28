#!/usr/bin/env python3
"""Build and validate the self-contained deep-learning submission archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SOURCE = PROJECT_ROOT / "submit"
PACKAGE_SOURCE = PROJECT_ROOT / "src" / "lg_control"
BUNDLE_SOURCE = PROJECT_ROOT / "artifacts" / "final_deploy_bundle"
ALLOWED_TOP_LEVEL = {"script.py", "requirements.txt", "model"}
REQUIRED_PACKAGE_FILES = {
    "__init__.py",
    "artifacts.py",
    "calibration.py",
    "constants.py",
    "features.py",
    "inference.py",
    "model.py",
    "schema.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def safe_relative_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative filename")
    if "\\" in value:
        raise ValueError(f"{label} must use portable '/' separators: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"{label} must be a filename without directories: {value!r}")
    return value


def validate_source_files() -> None:
    script_path = SUBMIT_SOURCE / "script.py"
    requirements_path = SUBMIT_SOURCE / "requirements.txt"
    for path in (script_path, requirements_path):
        if not path.is_file():
            raise FileNotFoundError(f"required submission source is missing: {path}")
        if path.is_symlink():
            raise ValueError(f"submission source must not be a symlink: {path}")

    source = script_path.read_text(encoding="utf-8")
    compile(source, str(script_path), "exec")
    if any(
        line.strip() and not line.lstrip().startswith("#")
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
    ):
        raise ValueError("submit/requirements.txt must contain comments only")

    if not PACKAGE_SOURCE.is_dir() or PACKAGE_SOURCE.is_symlink():
        raise FileNotFoundError(f"package source directory is missing: {PACKAGE_SOURCE}")
    present = {path.name for path in PACKAGE_SOURCE.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_PACKAGE_FILES.difference(present))
    if missing:
        raise FileNotFoundError(f"lg_control package files are missing: {missing}")
    for path in PACKAGE_SOURCE.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"package source must not contain symlinks: {path}")


def validate_manifest_payload(
    manifest: Mapping[str, Any],
) -> tuple[list[str], str, dict[str, str]]:
    if manifest.get("artifact_version") != 1:
        raise ValueError(
            f"unsupported artifact_version: {manifest.get('artifact_version')!r}"
        )

    raw_model_files = manifest.get("model_files")
    if not isinstance(raw_model_files, list):
        raise ValueError("manifest model_files must be a list")
    model_files = [
        safe_relative_name(value, label=f"model_files[{index}]")
        for index, value in enumerate(raw_model_files)
    ]
    if not model_files or len(set(model_files)) != len(model_files):
        raise ValueError("manifest must contain at least one unique model file")

    calibrator_file = safe_relative_name(
        manifest.get("calibrator_file"), label="calibrator_file"
    )
    raw_hashes = manifest.get("files_sha256")
    if not isinstance(raw_hashes, dict):
        raise ValueError("manifest files_sha256 must be an object")
    hashes: dict[str, str] = {}
    for raw_name, raw_digest in raw_hashes.items():
        name = safe_relative_name(raw_name, label="files_sha256 key")
        if not isinstance(raw_digest, str) or len(raw_digest) != 64:
            raise ValueError(f"invalid SHA-256 digest for {name}")
        try:
            int(raw_digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid SHA-256 digest for {name}") from exc
        hashes[name] = raw_digest.lower()

    required_hashed = {
        "preprocessor.json",
        "model_config.json",
        calibrator_file,
        *model_files,
    }
    missing_hashes = sorted(required_hashed.difference(hashes))
    if missing_hashes:
        raise ValueError(f"manifest is missing artifact hashes: {missing_hashes}")
    return model_files, calibrator_file, hashes


def validate_bundle(directory: Path) -> dict[str, Any]:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"model bundle directory is missing: {directory}")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"model bundle must not contain symlinks: {path}")

    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    model_files, calibrator_file, hashes = validate_manifest_payload(manifest)
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"artifact hash mismatch for {name}: expected {expected}, got {actual}"
            )

    # Parse the small state/config JSON files now so a malformed bundle fails
    # before the archive is written.
    read_json(directory / "preprocessor.json")
    read_json(directory / "model_config.json")
    read_json(directory / calibrator_file)
    for name in model_files:
        if (directory / name).stat().st_size == 0:
            raise ValueError(f"model artifact is empty: {name}")
    return manifest


def copy_submission_tree(staging: Path) -> None:
    shutil.copy2(SUBMIT_SOURCE / "script.py", staging / "script.py")
    shutil.copy2(SUBMIT_SOURCE / "requirements.txt", staging / "requirements.txt")

    model_root = staging / "model"
    model_root.mkdir()
    shutil.copytree(
        PACKAGE_SOURCE,
        model_root / "lg_control",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copytree(
        BUNDLE_SOURCE,
        model_root / "bundle",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def write_zip_member(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    is_directory = path.is_dir()
    normalized = arcname.rstrip("/") + ("/" if is_directory else "")
    info = zipfile.ZipInfo(normalized, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    if is_directory:
        info.external_attr = (0o40755 << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, b"")
    else:
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def build_archive(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}_",
            suffix=".zip.tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            paths = sorted(
                staging.rglob("*"),
                key=lambda path: path.relative_to(staging).as_posix(),
            )
            for path in paths:
                arcname = path.relative_to(staging).as_posix()
                write_zip_member(archive, path, arcname)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalized_zip_members(names: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for name in names:
        if not name or "\\" in name or name.startswith("/"):
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        trimmed = name.rstrip("/")
        path = PurePosixPath(trimmed)
        if (
            not trimmed
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        normalized.append(trimmed + ("/" if name.endswith("/") else ""))
    if len(normalized) != len(set(normalized)):
        raise ValueError("ZIP contains duplicate member names")
    return normalized


def validate_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"submission archive is missing: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a valid ZIP archive: {path}")

    with zipfile.ZipFile(path) as archive:
        members = normalized_zip_members(archive.namelist())
        files = {name for name in members if not name.endswith("/")}
        top_level = {name.rstrip("/").split("/", 1)[0] for name in members}
        if top_level != ALLOWED_TOP_LEVEL:
            raise ValueError(
                f"ZIP top-level entries must be {sorted(ALLOWED_TOP_LEVEL)}, "
                f"got {sorted(top_level)}"
            )
        required_root_files = {"script.py", "requirements.txt"}
        if not required_root_files.issubset(files):
            raise ValueError("ZIP root is missing script.py or requirements.txt")
        if any(name.startswith(("data/", "open/", "output/")) for name in members):
            raise ValueError("ZIP must not contain input data or generated output")

        required_code = {f"model/lg_control/{name}" for name in REQUIRED_PACKAGE_FILES}
        missing_code = sorted(required_code.difference(files))
        if missing_code:
            raise ValueError(f"ZIP is missing inference package files: {missing_code}")

        manifest_name = "model/bundle/manifest.json"
        if manifest_name not in files:
            raise ValueError("ZIP is missing model/bundle/manifest.json")
        try:
            manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("ZIP bundle manifest is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError("ZIP bundle manifest root must be an object")
        _, _, hashes = validate_manifest_payload(manifest)
        for name, expected in hashes.items():
            member = f"model/bundle/{name}"
            if member not in files:
                raise ValueError(f"ZIP is missing manifest artifact: {member}")
            actual = sha256_bytes(archive.read(member))
            if actual != expected:
                raise ValueError(
                    f"ZIP artifact hash mismatch for {name}: "
                    f"expected {expected}, got {actual}"
                )

        if archive.read("script.py") != (SUBMIT_SOURCE / "script.py").read_bytes():
            raise ValueError("ZIP script.py differs from submit/script.py")
        if archive.read("requirements.txt") != (
            SUBMIT_SOURCE / "requirements.txt"
        ).read_bytes():
            raise ValueError("ZIP requirements.txt differs from source")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_smoke_output(test_path: Path, output_path: Path) -> int:
    input_columns, input_rows = read_csv_rows(test_path)
    if "row_id" not in input_columns:
        raise ValueError("smoke test data is missing row_id")
    output_columns, output_rows = read_csv_rows(output_path)
    if output_columns != ["row_id", "control_success"]:
        raise ValueError(f"smoke output columns are invalid: {output_columns}")
    if len(output_rows) != len(input_rows):
        raise ValueError(
            f"smoke output row count mismatch: input={len(input_rows)}, "
            f"output={len(output_rows)}"
        )
    input_ids = [row["row_id"] for row in input_rows]
    output_ids = [row["row_id"] for row in output_rows]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("smoke output contains duplicate row_id values")
    if output_ids != input_ids:
        raise ValueError("smoke output row_id values or order differ from input")
    for index, row in enumerate(output_rows):
        try:
            probability = float(row["control_success"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"smoke probability at row {index} is not numeric") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"smoke probability at row {index} is outside [0, 1]")
    return len(output_rows)


def run_smoke_test(archive_path: Path, smoke_data: Path, timeout: int) -> int:
    if not smoke_data.is_file():
        raise FileNotFoundError(f"smoke data is missing: {smoke_data}")
    with tempfile.TemporaryDirectory(prefix="lg_submission_smoke_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            # validate_archive has already rejected traversal and duplicate paths.
            archive.extractall(root)
        data_directory = root / "data"
        data_directory.mkdir()
        copied_test = data_directory / "test.csv"
        shutil.copy2(smoke_data, copied_test)

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "script.py"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"smoke inference exceeded {timeout} seconds") from exc
        if result.returncode != 0:
            raise RuntimeError(
                "smoke inference failed with exit code "
                f"{result.returncode}\n--- stdout ---\n{result.stdout}"
                f"\n--- stderr ---\n{result.stderr}"
            )
        row_count = validate_smoke_output(copied_test, root / "output" / "submission.csv")
        print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        return row_count


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submit.zip"),
        help="archive path (default: ./submit.zip)",
    )
    parser.add_argument(
        "--smoke-data",
        type=Path,
        default=None,
        help="optional test.csv used in an isolated extracted-archive smoke run",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=positive_int,
        default=600,
        help="smoke inference timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate sources, bundle, and an existing --output without rebuilding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    validate_source_files()
    manifest = validate_bundle(BUNDLE_SOURCE)
    print(
        f"Bundle OK: {len(manifest['model_files'])} models, all manifest hashes match",
        flush=True,
    )

    if args.self_check:
        validate_archive(output)
        print(f"Archive self-check OK: {output}", flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="lg_submission_stage_") as temporary:
            staging = Path(temporary)
            copy_submission_tree(staging)
            validate_bundle(staging / "model" / "bundle")
            build_archive(staging, output)
        validate_archive(output)
        print(f"Created and validated: {output} ({output.stat().st_size:,} bytes)", flush=True)

    if args.smoke_data is not None:
        rows = run_smoke_test(output, args.smoke_data.resolve(), args.smoke_timeout)
        print(f"Isolated smoke test OK: {rows:,} rows", flush=True)


if __name__ == "__main__":
    main()
