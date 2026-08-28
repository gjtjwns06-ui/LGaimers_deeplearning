"""Create the sealed 2019--2023 development input without parsing 2024 labels.

The source CSV is streamed as bytes.  Only the unquoted ASCII ``row_id`` and
``season`` prefix is decoded before a row is either copied byte-for-byte or
discarded.  In particular, the suffix containing the target of excluded 2024
rows is never decoded, CSV-tokenized, converted, or validated.  The complete
source bytes are hashed only for integrity/provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.constants import TRAIN_COLUMNS


DEFAULT_INPUT = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "development_2019_2023.csv"
DEVELOPMENT_SEASONS = (2019, 2020, 2021, 2022, 2023)
EXCLUDED_SEASON = 2024


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy() -> dict[str, Any]:
    return {
        "included_seasons": list(DEVELOPMENT_SEASONS),
        "excluded_seasons": [EXCLUDED_SEASON],
        "excluded_row_fields_lexically_parsed": ["row_id", "season"],
        "excluded_target_lexically_parsed": False,
        "excluded_target_semantically_parsed": False,
        "byte_dialect_contract": (
            "official header order; unquoted ASCII row_id/season prefix; "
            "one official record per physical line"
        ),
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"snapshot build lock already exists: {path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_existing(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any] | None:
    if not output_path.exists() and not manifest_path.exists():
        return None
    if output_path.is_file() and not manifest_path.exists():
        # A manifest is the commit marker, but an arbitrary pre-existing file
        # must never be mistaken for recoverable staging and overwritten.
        raise RuntimeError(
            "snapshot output exists without its manifest; refusing to overwrite it"
        )
    if not output_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "development snapshot is partially present; remove or repair both files"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("existing snapshot manifest is malformed") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("existing snapshot manifest must be a JSON object")
    recorded_contract_hash = manifest.get("contract_sha256")
    unsigned = dict(manifest)
    unsigned.pop("contract_sha256", None)
    if recorded_contract_hash != _canonical_sha256(unsigned):
        raise RuntimeError("existing snapshot contract hash is invalid")
    if manifest.get("artifact") != "development_snapshot_v2":
        raise RuntimeError("existing snapshot manifest has an unsupported contract")
    if manifest.get("policy") != _policy():
        raise RuntimeError("existing snapshot policy differs from the fixed contract")
    source_record = manifest.get("source")
    output_record = manifest.get("output")
    if not isinstance(source_record, dict) or not isinstance(output_record, dict):
        raise RuntimeError("existing snapshot source/output records are malformed")
    if source_record.get("path") != str(input_path):
        raise RuntimeError("existing snapshot records a different source path")
    if output_record.get("path") != str(output_path):
        raise RuntimeError("existing snapshot records a different output path")
    if source_record.get("sha256") != _sha256_file(input_path):
        raise RuntimeError("existing snapshot was built from different source bytes")
    if manifest.get("output", {}).get("sha256") != _sha256_file(output_path):
        raise RuntimeError("existing development snapshot hash does not match its manifest")
    season_counts = output_record.get("season_counts")
    if not isinstance(season_counts, dict) or set(season_counts) != {
        str(season) for season in DEVELOPMENT_SEASONS
    }:
        raise RuntimeError("existing snapshot season counts are malformed")
    if any(type(value) is not int or value <= 0 for value in season_counts.values()):
        raise RuntimeError("existing snapshot season counts must be positive integers")
    included_rows = output_record.get("rows")
    excluded_rows = manifest.get("excluded_rows")
    source_rows = source_record.get("rows")
    if (
        type(included_rows) is not int
        or type(excluded_rows) is not int
        or type(source_rows) is not int
        or included_rows != sum(season_counts.values())
        or source_rows != included_rows + excluded_rows
    ):
        raise RuntimeError("existing snapshot row counts do not reconcile")
    return manifest


def build_snapshot(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = input_path.resolve(strict=True)
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    named_paths = (input_path, output_path, manifest_path, lock_path)
    if len(set(named_paths)) != len(named_paths):
        raise ValueError("source, output, manifest, and lock paths must be distinct")
    for candidate in (output_path, manifest_path, lock_path):
        if candidate.exists() and os.path.samefile(input_path, candidate):
            raise ValueError("snapshot artifacts must not alias the source file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock_path):
        existing = _validate_existing(input_path, output_path, manifest_path)
        if existing is not None:
            return existing

        source_sha256 = _sha256_file(input_path)
        season_counts: Counter[int] = Counter()
        included_rows = 0
        excluded_rows = 0
        seen_row_ids: set[str] = set()

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            with input_path.open("rb") as source, temporary_path.open("wb") as destination:
                try:
                    raw_header = next(source)
                except StopIteration as error:
                    raise ValueError("source CSV is empty") from error
                try:
                    header_text = raw_header.decode("utf-8-sig")
                    header = next(csv.reader([header_text], strict=True))
                except (UnicodeDecodeError, csv.Error) as error:
                    raise ValueError("source CSV header is malformed") from error
                if header != TRAIN_COLUMNS:
                    raise ValueError(
                        "source CSV column order differs from the official training schema"
                    )
                destination.write(raw_header)

                for line_number, raw_line in enumerate(source, start=2):
                    # Byte-level dialect contract: the official first two fields
                    # are unquoted ASCII row_id and season, and each official row
                    # occupies one physical line.  Inspect just that prefix before
                    # deciding whether the suffix may be decoded or tokenized.
                    first_comma = raw_line.find(b",")
                    second_comma = raw_line.find(b",", first_comma + 1)
                    if first_comma <= 0 or second_comma <= first_comma + 1:
                        raise ValueError(
                            f"line {line_number} has a malformed identity prefix"
                        )
                    raw_row_id = raw_line[:first_comma]
                    raw_season_bytes = raw_line[first_comma + 1 : second_comma]
                    if b'"' in raw_row_id or b'"' in raw_season_bytes:
                        raise ValueError(
                            f"line {line_number} quotes an official identity prefix field"
                        )
                    try:
                        row_id = raw_row_id.decode("ascii")
                        raw_season = raw_season_bytes.decode("ascii")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            f"line {line_number} has a non-ASCII identity prefix"
                        ) from error
                    if not row_id or row_id in seen_row_ids:
                        raise ValueError(
                            f"line {line_number} has a missing or duplicate row_id"
                        )
                    seen_row_ids.add(row_id)
                    try:
                        season = int(raw_season)
                    except ValueError as error:
                        raise ValueError(
                            f"line {line_number} has non-integer season {raw_season!r}"
                        ) from error
                    if str(season) != raw_season.strip():
                        raise ValueError(
                            f"line {line_number} has non-canonical season {raw_season!r}"
                        )
                    season_counts[season] += 1
                    if season in DEVELOPMENT_SEASONS:
                        try:
                            decoded_line = raw_line.decode("utf-8")
                            parsed = next(csv.reader([decoded_line], strict=True))
                        except (UnicodeDecodeError, csv.Error) as error:
                            raise ValueError(
                                f"development line {line_number} is malformed CSV"
                            ) from error
                        if len(parsed) != len(header):
                            raise ValueError(
                                f"development line {line_number} has {len(parsed)} fields; "
                                f"expected {len(header)}"
                            )
                        destination.write(raw_line)
                        if not raw_line.endswith((b"\n", b"\r")):
                            destination.write(b"\n")
                        included_rows += 1
                    elif season == EXCLUDED_SEASON:
                        # Deliberately do not decode, tokenize, or inspect the
                        # suffix.  The target stays an uninterpreted byte tail.
                        excluded_rows += 1
                    else:
                        raise ValueError(
                            f"unexpected season {season} at line {line_number}"
                        )

            observed = tuple(sorted(season_counts))
            expected = (*DEVELOPMENT_SEASONS, EXCLUDED_SEASON)
            if observed != expected:
                raise ValueError(
                    f"source season coverage differs from {expected}: observed={observed}"
                )
            if included_rows == 0 or excluded_rows == 0:
                raise ValueError("source must contain both development and excluded rows")
            if _sha256_file(input_path) != source_sha256:
                raise RuntimeError("source CSV changed while the snapshot was being built")

            output_sha256 = _sha256_file(temporary_path)
            contract: dict[str, Any] = {
                "artifact": "development_snapshot_v2",
                "policy": _policy(),
                "source": {
                    "path": str(input_path),
                    "sha256": source_sha256,
                    "rows": included_rows + excluded_rows,
                },
                "output": {
                    "path": str(output_path),
                    "sha256": output_sha256,
                    "rows": included_rows,
                    "season_counts": {
                        str(season): season_counts[season]
                        for season in DEVELOPMENT_SEASONS
                    },
                },
                "excluded_rows": excluded_rows,
            }
            contract["contract_sha256"] = _canonical_sha256(contract)
            os.replace(temporary_path, output_path)
            _write_json_atomic(manifest_path, contract)
            return contract
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_snapshot(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
