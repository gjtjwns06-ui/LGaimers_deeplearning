#!/usr/bin/env python3
"""Build and smoke-test the sealed single-model submission ZIP."""

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
import time
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.final_context_training import (
    FINAL_CODE_RELATIVE_FILES,
    FINAL_TRACKED_FILES,
    load_final_context_artifact,
)
from lg_control.regime_tuning import (
    _payload_with_hash,
    _serialize_json,
    _verify_embedded_hash,
)


SUBMIT_SOURCE = PROJECT_ROOT / "submit_context"
BUNDLE_SOURCE = PROJECT_ROOT / "artifacts" / "final_context_c2"
ALLOWED_TOP_LEVEL = {"script.py", "requirements.txt", "model"}
DEPLOYMENT_MANIFEST_KIND = "final_context_c2_submission_v1"
MAX_ZIP_BYTES = 10_000_000_000
MAX_UNCOMPRESSED_BYTES = 32_000_000_000
OFFICIAL_EVALUATION_ROWS = 245_789


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources() -> None:
    script = SUBMIT_SOURCE / "script.py"
    requirements = SUBMIT_SOURCE / "requirements.txt"
    for path in (script, requirements):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"submission source is missing or unsafe: {path}")
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    if any(
        line.strip() and not line.lstrip().startswith("#")
        for line in requirements.read_text(encoding="utf-8").splitlines()
    ):
        raise ValueError("requirements.txt may contain comments only")
    for relative in FINAL_CODE_RELATIVE_FILES:
        candidate = (PROJECT_ROOT / relative).resolve()
        if (
            PROJECT_ROOT not in candidate.parents
            or not candidate.is_file()
            or candidate.is_symlink()
        ):
            raise FileNotFoundError(
                f"sealed runtime source is missing or unsafe: {relative}"
            )


def validate_bundle(directory: Path, *, verify_code: bool) -> dict:
    loaded = load_final_context_artifact(directory, verify_code=verify_code)
    if loaded.manifest.get("model_count") != 1:
        raise ValueError("submission bundle must contain exactly one model")
    return loaded.manifest


def copy_submission_tree(staging: Path) -> None:
    shutil.copy2(SUBMIT_SOURCE / "script.py", staging / "script.py")
    shutil.copy2(SUBMIT_SOURCE / "requirements.txt", staging / "requirements.txt")
    model_root = staging / "model"
    model_root.mkdir()
    runtime_root = model_root / "runtime"
    for relative in FINAL_CODE_RELATIVE_FILES:
        source = PROJECT_ROOT / relative
        destination = runtime_root / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copytree(BUNDLE_SOURCE, model_root / "bundle")
    tracked: dict[str, str] = {
        "script.py": sha256_file(staging / "script.py"),
        "requirements.txt": sha256_file(staging / "requirements.txt"),
    }
    for path in sorted(model_root.rglob("*")):
        if path.is_file():
            tracked[path.relative_to(staging).as_posix()] = sha256_file(path)
    deployment = _payload_with_hash(
        {
            "schema_version": 1,
            "kind": DEPLOYMENT_MANIFEST_KIND,
            "files_sha256": tracked,
        },
        "deployment_manifest_sha256",
    )
    (model_root / "deployment_manifest.json").write_bytes(
        _serialize_json(deployment)
    )


def write_zip_member(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    directory = path.is_dir()
    name = arcname.rstrip("/") + ("/" if directory else "")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    if directory:
        info.external_attr = (0o40755 << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, b"")
    else:
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def build_archive(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite submission archive: {output}")
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
            for path in sorted(
                staging.rglob("*"),
                key=lambda item: item.relative_to(staging).as_posix(),
            ):
                write_zip_member(
                    archive,
                    path,
                    path.relative_to(staging).as_posix(),
                )
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_zip_names(names: list[str]) -> list[str]:
    normalized: list[str] = []
    for name in names:
        if not name or name.startswith("/") or "\\" in name:
            raise ValueError(f"unsafe ZIP path: {name!r}")
        trimmed = name.rstrip("/")
        path = PurePosixPath(trimmed)
        if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"unsafe ZIP path: {name!r}")
        normalized.append(trimmed + ("/" if name.endswith("/") else ""))
    if len(normalized) != len(set(normalized)):
        raise ValueError("ZIP contains duplicate paths")
    return normalized


def validate_archive(path: Path) -> None:
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"submission archive is invalid: {path}")
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("submission archive exceeds the 10GB limit")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("submission archive exceeds the 32GB extracted limit")
        for item in archive.infolist():
            mode = (item.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"ZIP contains a symlink: {item.filename}")
        names = _safe_zip_names(archive.namelist())
        files = {name for name in names if not name.endswith("/")}
        top = {name.rstrip("/").split("/", 1)[0] for name in names}
        if top != ALLOWED_TOP_LEVEL:
            raise ValueError(f"ZIP top-level entries differ: {sorted(top)}")
        if not {"script.py", "requirements.txt"}.issubset(files):
            raise ValueError("ZIP root lacks script.py or requirements.txt")
        if any(name.startswith(("data/", "open/", "output/")) for name in names):
            raise ValueError("ZIP must not contain input or output data")
        manifest_name = "model/bundle/manifest.json"
        deployment_name = "model/deployment_manifest.json"
        if manifest_name not in files:
            raise ValueError("ZIP lacks the final artifact manifest")
        if deployment_name not in files:
            raise ValueError("ZIP lacks the deployment manifest")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        _verify_embedded_hash(manifest, "artifact_manifest_sha256")
        hashes = manifest.get("files_sha256")
        if not isinstance(hashes, dict) or set(hashes) != FINAL_TRACKED_FILES:
            raise ValueError("ZIP artifact hash inventory differs")
        for name, expected in hashes.items():
            member = f"model/bundle/{name}"
            if member not in files:
                raise ValueError(f"ZIP lacks artifact file: {member}")
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            if actual != expected:
                raise ValueError(f"ZIP artifact hash differs: {member}")
        bundle_files = {
            name for name in files if name.startswith("model/bundle/")
        }
        expected_bundle_files = {manifest_name}.union(
            {f"model/bundle/{name}" for name in FINAL_TRACKED_FILES}
        )
        if bundle_files != expected_bundle_files:
            raise ValueError("ZIP artifact file inventory differs")
        if archive.read("script.py") != (SUBMIT_SOURCE / "script.py").read_bytes():
            raise ValueError("ZIP script.py differs from source")
        if archive.read("requirements.txt") != (
            SUBMIT_SOURCE / "requirements.txt"
        ).read_bytes():
            raise ValueError("ZIP requirements.txt differs from source")
        code_hashes = manifest.get("code_files_sha256")
        if not isinstance(code_hashes, dict) or set(code_hashes) != set(
            FINAL_CODE_RELATIVE_FILES
        ):
            raise ValueError("ZIP sealed runtime code inventory differs")
        expected_runtime_files = {
            f"model/runtime/{relative}" for relative in FINAL_CODE_RELATIVE_FILES
        }
        runtime_files = {
            name for name in files if name.startswith("model/runtime/")
        }
        if runtime_files != expected_runtime_files:
            raise ValueError("ZIP runtime file inventory differs")
        for relative, expected in code_hashes.items():
            member = f"model/runtime/{relative}"
            if hashlib.sha256(archive.read(member)).hexdigest() != expected:
                raise ValueError(f"ZIP runtime code hash differs: {member}")

        deployment = json.loads(archive.read(deployment_name).decode("utf-8"))
        _verify_embedded_hash(deployment, "deployment_manifest_sha256")
        if (
            deployment.get("schema_version") != 1
            or deployment.get("kind") != DEPLOYMENT_MANIFEST_KIND
        ):
            raise ValueError("ZIP deployment manifest contract differs")
        deployment_hashes = deployment.get("files_sha256")
        deployed_files = files.difference({deployment_name})
        if not isinstance(deployment_hashes, dict) or set(deployment_hashes) != deployed_files:
            raise ValueError("ZIP deployment file inventory differs")
        for member, expected in deployment_hashes.items():
            if hashlib.sha256(archive.read(member)).hexdigest() != expected:
                raise ValueError(f"ZIP deployment file hash differs: {member}")

        with tempfile.TemporaryDirectory(
            prefix="lg_final_archive_bundle_check_"
        ) as temporary:
            bundle_root = Path(temporary) / "bundle"
            bundle_root.mkdir()
            for name in sorted(FINAL_TRACKED_FILES.union({"manifest.json"})):
                (bundle_root / name).write_bytes(
                    archive.read(f"model/bundle/{name}")
                )
            load_final_context_artifact(bundle_root, verify_code=True)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV lacks a header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_smoke_output(test_path: Path, output_path: Path) -> int:
    with test_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        with output_path.open("r", encoding="utf-8-sig", newline="") as output_handle:
            input_reader = csv.DictReader(input_handle)
            output_reader = csv.DictReader(output_handle)
            input_columns = list(input_reader.fieldnames or [])
            output_columns = list(output_reader.fieldnames or [])
            if "row_id" not in input_columns:
                raise ValueError("smoke input lacks row_id")
            if output_columns != ["row_id", "control_success"]:
                raise ValueError(f"smoke output columns differ: {output_columns}")
            seen: set[str] = set()
            rows = 0
            for input_row in input_reader:
                output_row = next(output_reader, None)
                if output_row is None:
                    raise ValueError("smoke output has too few rows")
                row_id = input_row["row_id"]
                if output_row["row_id"] != row_id or row_id in seen:
                    raise ValueError("smoke output row identity/order differs")
                seen.add(row_id)
                probability = float(output_row["control_success"])
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("smoke output probability is invalid")
                rows += 1
            if next(output_reader, None) is not None:
                raise ValueError("smoke output has too many rows")
    if rows < 1:
        raise ValueError("smoke output is empty")
    return rows


def smoke_test(archive_path: Path, smoke_data: Path, timeout: int) -> int:
    if not smoke_data.is_file():
        raise FileNotFoundError(f"smoke test data is missing: {smoke_data}")
    with tempfile.TemporaryDirectory(prefix="lg_final_submission_smoke_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
        (root / "data").mkdir()
        copied = root / "data" / "test.csv"
        shutil.copy2(smoke_data, copied)
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
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"smoke inference exceeded {timeout} seconds") from error
        if result.returncode != 0:
            raise RuntimeError(
                f"smoke inference failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        if sys.platform.startswith("linux"):
            import resource

            peak_rss_mib = (
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
            )
            print(f"Isolated child peak RSS: {peak_rss_mib:.2f} MiB", flush=True)
        return validate_smoke_output(copied, root / "output" / "submission.csv")


def build_scaled_smoke_data(source: Path, destination: Path, rows: int) -> None:
    columns, records = read_csv(source)
    if not records or "row_id" not in columns or "control_success" in columns:
        raise ValueError("scale smoke source must be nonempty, target-free test data")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for index in range(rows):
            record = dict(records[index % len(records)])
            record["row_id"] = f"FULL_SCALE_{index:06d}"
            writer.writerow(record)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submit_final_context.zip"),
    )
    parser.add_argument("--smoke-data", type=Path, default=None)
    parser.add_argument("--smoke-timeout", type=positive_int, default=600)
    parser.add_argument("--full-scale-smoke", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    validate_sources()
    manifest = validate_bundle(BUNDLE_SOURCE, verify_code=True)
    print(
        "Bundle OK: "
        f"model_count={manifest['model_count']}, "
        f"manifest={manifest['artifact_manifest_sha256']}",
        flush=True,
    )
    if args.self_check:
        validate_archive(output)
    else:
        with tempfile.TemporaryDirectory(prefix="lg_final_submission_stage_") as temporary:
            staging = Path(temporary)
            copy_submission_tree(staging)
            validate_bundle(staging / "model" / "bundle", verify_code=False)
            build_archive(staging, output)
        validate_archive(output)
        print(f"Created: {output} ({output.stat().st_size:,} bytes)", flush=True)
    if args.smoke_data is not None:
        smoke_source = args.smoke_data.resolve()
        if args.full_scale_smoke:
            with tempfile.TemporaryDirectory(
                prefix="lg_final_full_scale_input_"
            ) as temporary:
                scaled = Path(temporary) / "test.csv"
                build_scaled_smoke_data(
                    smoke_source, scaled, OFFICIAL_EVALUATION_ROWS
                )
                started = time.perf_counter()
                rows = smoke_test(output, scaled, args.smoke_timeout)
                elapsed = time.perf_counter() - started
            print(
                f"Full-scale isolated smoke OK: {rows:,} rows in {elapsed:.2f}s",
                flush=True,
            )
        else:
            rows = smoke_test(output, smoke_source, args.smoke_timeout)
            print(f"Isolated smoke test OK: {rows:,} rows", flush=True)
    elif args.full_scale_smoke:
        raise ValueError("--full-scale-smoke requires --smoke-data")


if __name__ == "__main__":
    main()
