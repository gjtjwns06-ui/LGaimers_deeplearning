#!/usr/bin/env python3
"""Inference entrypoint for v17 + gated DL + complementary residual heads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "model"
V17_SCRIPT = MODEL_ROOT / "v17" / "script.py"
DL_BUNDLE = MODEL_ROOT / "dl_bundle"
DL_RUNTIME_SRC = MODEL_ROOT / "dl_runtime" / "src"
GATE_ROOT = MODEL_ROOT / "gate"
RESIDUAL_ROOT = MODEL_ROOT / "uncertain_residual"
APPLICATION_CONFIG = MODEL_ROOT / "residual_application" / "config.json"
F_SHAPE_ROOT = MODEL_ROOT / "f_regime_shape"
COMBINED_COMPLEMENT_ROOT = MODEL_ROOT / "combined_complement"
COMBINED_GATE_PATH = MODEL_ROOT / "combined_r_gate" / "artifact.json"
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
    if payload.get("kind") != "v17_recent_dl_uncertain_residual_f_shape_r_complement_submission_v1":
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


def _load_v17_runtime():
    spec = importlib.util.spec_from_file_location("frozen_v17_runtime", V17_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the frozen v17 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_test_path() -> Path:
    for path in (Path("data/test.csv"), Path("open/test.csv")):
        if path.is_file():
            return path
    raise FileNotFoundError("test.csv was not found")


def write_submission(frame, probabilities) -> None:
    import pandas as pd

    from lg_control.constants import ID_COL, TARGET_COL
    from lg_control.schema import validate_predictions

    values = validate_predictions(frame[ID_COL], probabilities)
    output = pd.DataFrame({ID_COL: frame[ID_COL].copy(), TARGET_COL: values})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix="residual_hybrid_",
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
    verify_deployment()
    if str(DL_RUNTIME_SRC) not in sys.path:
        sys.path.insert(0, str(DL_RUNTIME_SRC))

    import numpy as np
    import pandas as pd
    import torch

    from lg_control.complementary_gate import (
        load_complementary_gate,
        predict_complementary_hybrid,
    )
    from lg_control.combined_complement import (
        load_combined_r_complement,
        predict_combined_r_complement,
    )
    from lg_control.constants import ID_COL
    from lg_control.recent_regime_final import (
        load_recent_regime_artifact,
        predict_recent_regime_heads,
    )
    from lg_control.f_regime_shape import (
        load_f_regime_shape,
        predict_f_regime_shape,
    )
    from lg_control.schema import validate_input_frame
    from lg_control.uncertain_residual import (
        load_uncertain_residual,
        predict_uncertain_region,
    )

    frame = pd.read_csv(resolve_test_path(), encoding="utf-8-sig", low_memory=False)
    validate_input_frame(frame, require_target=False, allow_target=False)
    if frame.empty or not frame[ID_COL].is_unique:
        raise ValueError("test row IDs must be nonempty and unique")

    v17 = _load_v17_runtime()
    meta, boosters, cat_models = v17.load_predictor()
    prediction_v17 = v17.predict(frame, meta, boosters, cat_models)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_artifact = load_recent_regime_artifact(DL_BUNDLE, device=device)
    dl_heads = predict_recent_regime_heads(
        dl_artifact, frame, batch_size=32768, device=device
    )
    prediction_dl = dl_heads["outcome_control_success"]
    gate = load_complementary_gate(GATE_ROOT)
    probability, gate_weights = predict_complementary_hybrid(
        frame, prediction_v17, prediction_dl, gate
    )

    residual = load_uncertain_residual(RESIDUAL_ROOT, device=device)
    uncertain_probability, corrections, uncertain_mask = predict_uncertain_region(
        residual,
        frame,
        prediction_v17,
        prediction_dl,
        batch_size=32768,
        device=device,
    )
    application = json.loads(APPLICATION_CONFIG.read_text(encoding="utf-8"))
    if (
        application.get("kind") != "uncertain_residual_application_v1"
        or application.get("decision") != "PASS"
    ):
        raise ValueError("uncertain residual application config is invalid")
    scope = str(application["chosen"]["scope"])
    strength = float(application["chosen"]["strength"])
    if scope == "R":
        scope_mask = frame["game_type"].astype(str).str.upper().eq("R").to_numpy()
    elif scope == "F":
        scope_mask = frame["game_type"].astype(str).str.upper().eq("F").to_numpy()
    elif scope == "all":
        scope_mask = np.ones(len(frame), dtype=bool)
    else:
        raise ValueError("unknown uncertain residual application scope")
    application_mask = uncertain_mask & scope_mask
    blended = prediction_v17 + strength * (
        uncertain_probability - prediction_v17
    )
    probability[application_mask] = blended[application_mask]
    f_shape = load_f_regime_shape(F_SHAPE_ROOT)
    probability, f_correction, f_mask = predict_f_regime_shape(
        f_shape, frame, probability, dl_heads
    )
    combined = load_combined_r_complement(
        COMBINED_COMPLEMENT_ROOT, COMBINED_GATE_PATH, device=device
    )
    probability, combined_probability, combined_weights = predict_combined_r_complement(
        combined,
        frame,
        probability,
        batch_size=32768,
        device=device,
    )
    if not np.isfinite(probability).all():
        raise RuntimeError("final residual hybrid probabilities are non-finite")
    write_submission(frame, probability)
    print(
        f"Saved {OUTPUT_PATH} ({len(frame):,} rows, "
        f"uncertain_rows={int(uncertain_mask.sum()):,}, "
        f"applied_rows={int(application_mask.sum()):,}, "
        f"f_shape_rows={int(f_mask.sum()):,}, "
        f"f_shape_mean={f_correction[f_mask].mean() if f_mask.any() else 0.0:.8f}, "
        f"r_complement_rows={int((combined_weights > 0).sum()):,}, "
        f"r_complement_weight={combined_weights.mean():.6f}, "
        f"r_complement_mean={combined_probability.mean():.6f}, "
        f"mean_gate={gate_weights.mean():.6f}, "
        f"mean_residual={corrections[uncertain_mask].mean() if uncertain_mask.any() else 0.0:.6f})"
    )


if __name__ == "__main__":
    main()
