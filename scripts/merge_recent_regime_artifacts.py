#!/usr/bin/env python3
"""Merge independently trained recent-regime seeds into one strict ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.recent_regime_final import (
    ARTIFACT_VERSION,
    load_recent_regime_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(sources: list[Path], output: Path) -> dict:
    if len(sources) < 2:
        raise ValueError("at least two seed artifacts are required")
    loaded = [load_recent_regime_artifact(path) for path in sources]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reference = sources[0]
    for path in sources[1:]:
        for common in ("preprocessor.json", "model_config.json"):
            if (reference / common).read_bytes() != (path / common).read_bytes():
                raise ValueError(f"seed artifacts differ in {common}")
    invariant_records = []
    for artifact in loaded:
        record = dict(artifact.training_record)
        record.pop("seeds", None)
        record.pop("seed_records", None)
        invariant_records.append(record)
    if any(record != invariant_records[0] for record in invariant_records[1:]):
        raise ValueError("seed training records differ outside seed-specific fields")

    seed_records: list[dict] = []
    model_sources: dict[int, Path] = {}
    for source, artifact in zip(sources, loaded, strict=True):
        records = artifact.training_record.get("seed_records")
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("each source must contain exactly one seed record")
        seed = int(records[0]["seed"])
        if seed in model_sources:
            raise ValueError(f"duplicate seed artifact: {seed}")
        filenames = artifact.manifest.get("model_files")
        if not isinstance(filenames, list) or len(filenames) != 1:
            raise ValueError("each source must contain exactly one model file")
        model_sources[seed] = source / filenames[0]
        seed_records.append(records[0])
    seeds = sorted(model_sources)
    seed_records.sort(key=lambda record: int(record["seed"]))

    output.mkdir(parents=True)
    shutil.copy2(reference / "preprocessor.json", output / "preprocessor.json")
    shutil.copy2(reference / "model_config.json", output / "model_config.json")
    model_files: list[str] = []
    for seed in seeds:
        filename = f"model_seed_{seed}.npz"
        shutil.copy2(model_sources[seed], output / filename)
        model_files.append(filename)
    training_record = dict(invariant_records[0])
    training_record["seeds"] = seeds
    training_record["seed_records"] = seed_records
    (output / "training_record.json").write_text(
        json.dumps(training_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tracked = ["preprocessor.json", "model_config.json", "training_record.json", *model_files]
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "architecture": "recent_regime_rate_basis_exclusive_ensemble_v1",
        "model_files": model_files,
        "files_sha256": {name: _sha256(output / name) for name in tracked},
        "row_independent_inference": True,
        "obsolete_pre_2023_f_training_rows": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verified = load_recent_regime_artifact(output)
    if len(verified.models) != len(seeds):
        raise RuntimeError("merged model count differs")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/final_recent_regime")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = merge(args.sources, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "models": manifest["model_files"],
                "obsolete_pre_2023_f_training_rows": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
