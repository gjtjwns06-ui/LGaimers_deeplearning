from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_development_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_development_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _official_row(row_id: str, season: int, target: str) -> list[str]:
    values = [""] * len(MODULE.TRAIN_COLUMNS)
    values[MODULE.TRAIN_COLUMNS.index("row_id")] = row_id
    values[MODULE.TRAIN_COLUMNS.index("season")] = str(season)
    values[MODULE.TRAIN_COLUMNS.index("control_success")] = target
    return values


def _write_source(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MODULE.TRAIN_COLUMNS)
        writer.writerows(rows)


def test_snapshot_excludes_2024_without_parsing_its_target(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    output = tmp_path / "development.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, str(season % 2))
        for season in MODULE.DEVELOPMENT_SEASONS
    ]
    _write_source(source, rows)
    # Invalid UTF-8 in the target tail proves the 2024 suffix is not decoded,
    # tokenized, or validated.  Its ASCII identity prefix remains valid.
    with source.open("ab") as handle:
        handle.write(b"TRAIN_2024,2024," + b"\xff\xfeDO_NOT_PARSE\n")

    manifest = MODULE.build_snapshot(source, output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert [int(row["season"]) for row in saved] == list(
        MODULE.DEVELOPMENT_SEASONS
    )
    assert all(row["control_success"] in {"0", "1"} for row in saved)
    assert manifest["policy"]["excluded_target_lexically_parsed"] is False
    assert manifest["policy"]["excluded_target_semantically_parsed"] is False
    assert manifest["excluded_rows"] == 1


def test_malformed_development_tail_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    output = tmp_path / "development.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, "0")
        for season in (*MODULE.DEVELOPMENT_SEASONS[1:], MODULE.EXCLUDED_SEASON)
    ]
    _write_source(source, rows)
    with source.open("ab") as handle:
        handle.write(b"TRAIN_2019,2019," + b"\xff\xfeMALFORMED\n")
    with pytest.raises(ValueError, match="development line .* malformed CSV"):
        MODULE.build_snapshot(source, output)


def test_snapshot_resume_verifies_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    output = tmp_path / "development.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, "0")
        for season in (*MODULE.DEVELOPMENT_SEASONS, MODULE.EXCLUDED_SEASON)
    ]
    _write_source(source, rows)
    first = MODULE.build_snapshot(source, output)
    second = MODULE.build_snapshot(source, output)
    assert first == second

    rows[0][-1] = "1"
    _write_source(source, rows)
    with pytest.raises(RuntimeError, match="different source bytes"):
        MODULE.build_snapshot(source, output)


def test_snapshot_contract_tampering_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    output = tmp_path / "development.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, "0")
        for season in (*MODULE.DEVELOPMENT_SEASONS, MODULE.EXCLUDED_SEASON)
    ]
    _write_source(source, rows)
    MODULE.build_snapshot(source, output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["excluded_rows"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract hash"):
        MODULE.build_snapshot(source, output)


def test_snapshot_roundtrips_into_development_plan(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    output = tmp_path / "development.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, str(season % 2))
        for season in (*MODULE.DEVELOPMENT_SEASONS, MODULE.EXCLUDED_SEASON)
    ]
    _write_source(source, rows)
    MODULE.build_snapshot(source, output)

    from lg_control.validation import build_development_plan

    plan = build_development_plan(pd.read_csv(output, low_memory=False))
    assert [fold.validation_season for fold in plan.development_folds] == [2022, 2023]


def test_snapshot_refuses_to_overwrite_its_source(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, "0")
        for season in (*MODULE.DEVELOPMENT_SEASONS, MODULE.EXCLUDED_SEASON)
    ]
    _write_source(source, rows)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must be distinct"):
        MODULE.build_snapshot(source, source)
    assert source.read_bytes() == before


def test_snapshot_refuses_uncommitted_or_unrelated_existing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.csv"
    output = tmp_path / "important.csv"
    rows = [
        _official_row(f"TRAIN_{season}", season, "0")
        for season in (*MODULE.DEVELOPMENT_SEASONS, MODULE.EXCLUDED_SEASON)
    ]
    _write_source(source, rows)
    output.write_bytes(b"important-user-data\n")
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        MODULE.build_snapshot(source, output)
    assert output.read_bytes() == before
