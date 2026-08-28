#!/usr/bin/env python3
"""Train and forward-audit an F-only reverse-state residual on top of v20."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_complementary_hybrid as complementary  # noqa: E402
from lg_control.auxiliary_targets import recover_fold_train_auxiliary_targets  # noqa: E402
from lg_control.complementary_gate import (  # noqa: E402
    load_complementary_gate,
    predict_complementary_hybrid,
)
from lg_control.f_regime_residual import (  # noqa: E402
    FRegimeResidualPreprocessor,
    HEAD_COLUMNS,
)
from lg_control.hybrid_residual import (  # noqa: E402
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    HybridResidualTrainConfig,
    predict_bounded_residual,
    save_model_npz,
    train_bounded_residual,
    train_bounded_residual_fixed_epochs,
)


OUTPUT = ROOT / "artifacts" / "f_regime_reverse_residual_v2"
SEEDS = (42, 2026, 3407)
HEAD_FILES = (
    ROOT / "artifacts/recent_regime_outer_stable_r_seed42_polish4/predictions.csv",
    ROOT / "artifacts/recent_regime_outer_stable_r_seed2026_polish4/predictions.csv",
    ROOT / "artifacts/recent_regime_outer_stable_r_seed3407_polish4/predictions.csv",
)
GATE_KEYS = ["game_type", "count", "direction"]
GATE_SHRINK_ROWS = 250
GATE_CAP = 0.15


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    y = frame["control_success"].to_numpy(float)
    p = np.asarray(probability, dtype=np.float64)
    rate = float(y.mean())
    brier = float(np.mean(np.square(y - p)))
    return {
        "rows": int(len(frame)),
        "target_rate": rate,
        "prediction_mean": float(p.mean()),
        "brier": brier,
        "competition_score": max(0.0, 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def from_correction(base: np.ndarray, correction: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(logit(base) + correction)))


def head_mapping(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: frame[f"dlh_{name}"].to_numpy(float) for name in HEAD_COLUMNS}


def load_head_ensemble() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    source_columns = ["row_id", *(f"head_{name}" for name in HEAD_COLUMNS)]
    for index, path in enumerate(HEAD_FILES):
        piece = pd.read_csv(path, usecols=source_columns)
        if piece.row_id.duplicated().any():
            raise RuntimeError(f"duplicate row IDs in {path}")
        piece = piece.rename(
            columns={f"head_{name}": f"seed{index}_{name}" for name in HEAD_COLUMNS}
        )
        pieces.append(piece)
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="row_id", validate="one_to_one")
    output = merged[["row_id"]].copy()
    for name in HEAD_COLUMNS:
        output[f"dlh_{name}"] = np.mean(
            np.column_stack([merged[f"seed{index}_{name}"] for index in range(len(pieces))]),
            axis=1,
        )
    outcome = output[[f"dlh_{name}" for name in HEAD_COLUMNS[:5]]].sum(axis=1)
    if not np.allclose(outcome, 1.0, atol=2e-5):
        raise RuntimeError("ensemble outcome heads do not sum to one")
    return output


def load_frame(*, all_game_types: bool = False) -> pd.DataFrame:
    args = SimpleNamespace(
        train_csv=ROOT / "data/train.csv",
        research_root=ROOT.parent / "LG-aimers-v11-independent-research",
        dl_oof=ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    frame = complementary._load_aligned_oof(args)
    raw = pd.read_csv(args.train_csv, encoding="utf-8-sig", low_memory=False)
    recovered = recover_fold_train_auxiliary_targets(raw, partition="fold_train")
    raw["aux_reverse_true"] = recovered["aux_reverse"].to_numpy()
    raw["aux_middle_true"] = recovered["aux_middle"].to_numpy()
    duplicate = [column for column in raw if column in frame and column != "row_id"]
    raw = raw.drop(columns=duplicate)
    frame = frame.merge(raw, on="row_id", validate="one_to_one")
    frame = frame.merge(load_head_ensemble(), on="row_id", validate="one_to_one")
    if not all_game_types:
        frame = frame[frame.game_type.astype(str).str.upper().eq("F")].reset_index(drop=True)
        if len(frame) != 30010 or set(frame.season.astype(int)) != {2024}:
            raise RuntimeError("F reverse residual must use the exact 2024 F OOF")
    elif len(frame) != 253507 or set(frame.season.astype(int)) != {2024}:
        raise RuntimeError("F reverse residual base must use the exact 2024 full OOF")
    return frame


def split_base(fit: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gate = complementary._fit_gate(fit, GATE_KEYS, GATE_SHRINK_ROWS, GATE_CAP)
    fit_base, _ = complementary._gate_predictions(fit, gate)
    eval_base, _ = complementary._gate_predictions(evaluation, gate)
    return fit_base, eval_base


def model(input_dim: int, cap: float, seed: int) -> BoundedLogitResidualNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return BoundedLogitResidualNet(
            HybridResidualModelConfig(
                input_dim=input_dim,
                correction_cap=cap,
                hidden_dim=32,
                hidden_layers=2,
                dropout=0.10,
            )
        )


def train_config(seed: int, epochs: int = 30) -> HybridResidualTrainConfig:
    return HybridResidualTrainConfig(
        epochs=epochs,
        scheduler_epochs=30,
        batch_size=2048,
        eval_batch_size=32768,
        learning_rate=5e-4,
        weight_decay=5e-3,
        correction_l2=0.15,
        mean_correction_penalty=0.10,
        patience=6,
        min_delta=2e-8,
        seed=seed,
        gradient_clip_norm=2.0,
        verbose=False,
    )


def develop(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_base: np.ndarray,
    validation_base: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for include_heads in (False, True):
        mode = "reverse_heads" if include_heads else "no_reverse_heads"
        preprocessor = FRegimeResidualPreprocessor(
            include_reverse_heads=include_heads
        ).fit(train, train_base, head_mapping(train))
        train_x = preprocessor.transform(train, train_base, head_mapping(train))
        validation_x = preprocessor.transform(
            validation, validation_base, head_mapping(validation)
        )
        write_json(OUTPUT / f"development_preprocessor_{mode}.json", preprocessor.to_state_dict())
        for cap in (0.03, 0.05, 0.08):
            print(f"develop mode={mode} cap={cap:.2f}", flush=True)
            net = model(train_x.shape[1], cap, 42)
            result = train_bounded_residual(
                net,
                train_x,
                train_base.astype(np.float32),
                train.control_success.to_numpy(np.float32),
                validation_x,
                validation_base.astype(np.float32),
                validation.control_success.to_numpy(np.float32),
                config=train_config(42),
                device="cpu",
            )
            _, train_correction = predict_bounded_residual(
                net, train_x, train_base, batch_size=32768, device="cpu"
            )
            train_center = float(train_correction.mean())
            for center_on_fit in (False, True):
                center = train_center if center_on_fit else 0.0
                for strength in (0.25, 0.50, 0.75, 1.00):
                    applied = strength * (result.corrections - center)
                    probability = from_correction(validation_base, applied)
                    candidate = {
                        "mode": mode,
                        "include_reverse_heads": include_heads,
                        "center_on_fit": center_on_fit,
                        "development_correction_center": center,
                        "cap": cap,
                        "strength": strength,
                        "selected_epoch": int(result.best_epoch),
                        "feature_count": int(train_x.shape[1]),
                        "base": metrics(validation, validation_base),
                        "candidate": metrics(validation, probability),
                        "mean_logit_correction": float(applied.mean()),
                    }
                    candidate["delta_brier"] = float(
                        candidate["candidate"]["brier"] - candidate["base"]["brier"]
                    )
                    candidates.append(candidate)
    selected = min(candidates, key=lambda item: item["candidate"]["brier"])
    return selected, candidates


def fit_fixed(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    fit_base: np.ndarray,
    evaluation_base: np.ndarray,
    selected: dict[str, object],
) -> tuple[FRegimeResidualPreprocessor, list[BoundedLogitResidualNet], np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    preprocessor = FRegimeResidualPreprocessor(
        include_reverse_heads=bool(selected["include_reverse_heads"])
    ).fit(fit, fit_base, head_mapping(fit))
    fit_x = preprocessor.transform(fit, fit_base, head_mapping(fit))
    evaluation_x = preprocessor.transform(
        evaluation, evaluation_base, head_mapping(evaluation)
    )
    probabilities: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    fit_corrections: list[np.ndarray] = []
    models: list[BoundedLogitResidualNet] = []
    histories: list[dict[str, object]] = []
    for seed in SEEDS:
        print(f"fixed seed={seed} epochs={selected['selected_epoch']}", flush=True)
        net = model(fit_x.shape[1], float(selected["cap"]), seed)
        history = train_bounded_residual_fixed_epochs(
            net,
            fit_x,
            fit_base.astype(np.float32),
            fit.control_success.to_numpy(np.float32),
            config=train_config(seed, int(selected["selected_epoch"])),
            device="cpu",
        )
        probability, correction = predict_bounded_residual(
            net,
            evaluation_x,
            evaluation_base,
            batch_size=32768,
            device="cpu",
        )
        _, fit_correction = predict_bounded_residual(
            net, fit_x, fit_base, batch_size=32768, device="cpu"
        )
        models.append(net)
        probabilities.append(probability)
        corrections.append(correction)
        fit_corrections.append(fit_correction)
        histories.append({"seed": seed, "history": history})
    del probabilities
    correction_center = (
        float(np.mean(np.concatenate(fit_corrections)))
        if bool(selected["center_on_fit"])
        else 0.0
    )
    correction = float(selected["strength"]) * (
        np.mean(corrections, axis=0) - correction_center
    )
    probability = from_correction(evaluation_base, correction)
    return preprocessor, models, probability, correction, correction_center, histories


def main() -> None:
    torch.set_num_threads(8)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    frame = load_frame()
    dev_train = frame[frame.game_month.le(5)].reset_index(drop=True)
    dev_validation = frame[frame.game_month.between(6, 7)].reset_index(drop=True)
    audit_train = frame[frame.game_month.le(7)].reset_index(drop=True)
    audit = frame[frame.game_month.ge(8)].reset_index(drop=True)

    dev_train_base, dev_validation_base = split_base(dev_train, dev_validation)
    selected, candidates = develop(
        dev_train, dev_validation, dev_train_base, dev_validation_base
    )
    print(
        f"selected mode={selected['mode']} cap={selected['cap']} "
        f"strength={selected['strength']} delta={selected['delta_brier']:+.10f}",
        flush=True,
    )

    audit_train_base, audit_base = split_base(audit_train, audit)
    audit_pre, audit_models, audit_probability, audit_correction, audit_center, audit_histories = fit_fixed(
        audit_train, audit, audit_train_base, audit_base, selected
    )
    audit_payload = {
        "base": metrics(audit, audit_base),
        "candidate": metrics(audit, audit_probability),
        "delta_brier": float(metrics(audit, audit_probability)["brier"] - metrics(audit, audit_base)["brier"]),
        "mean_logit_correction": float(audit_correction.mean()),
        "correction_center": audit_center,
        "histories": audit_histories,
    }
    pass_gate = bool(
        selected["include_reverse_heads"]
        and selected["delta_brier"] < -1e-6
        and audit_payload["delta_brier"] < -1e-6
    )

    reverse_probability = (
        frame.dlh_outcome_reverse_only.to_numpy(float)
        + frame.dlh_outcome_reverse_and_middle.to_numpy(float)
    )
    valid_reverse = frame.aux_reverse_true.notna().to_numpy()
    reverse_brier = float(
        np.mean(
            np.square(
                frame.loc[valid_reverse, "aux_reverse_true"].to_numpy(float)
                - reverse_probability[valid_reverse]
            )
        )
    )
    decile = pd.qcut(reverse_probability, 10, labels=False, duplicates="drop")
    reverse_calibration = (
        pd.DataFrame(
            {
                "decile": decile,
                "actual_reverse": frame.aux_reverse_true,
                "predicted_reverse": reverse_probability,
            }
        )
        .dropna()
        .groupby("decile")
        .agg(rows=("actual_reverse", "size"), actual=("actual_reverse", "mean"), predicted=("predicted_reverse", "mean"))
        .reset_index()
    )
    reverse_calibration.to_csv(OUTPUT / "reverse_head_calibration.csv", index=False)

    audit_output = audit[["row_id", "game_month", "control_success"]].copy()
    audit_output["prediction_v20_base"] = audit_base
    audit_output["prediction_f_reverse_residual"] = audit_probability
    audit_output["logit_correction"] = audit_correction
    audit_output.to_parquet(OUTPUT / "audit_predictions.parquet", index=False)

    final_payload: dict[str, object] | None = None
    if pass_gate:
        print("audit PASS; fitting final F reverse residual", flush=True)
        deployed_gate = load_complementary_gate(
            ROOT / "artifacts/complementary_hybrid_v1/gate"
        )
        final_base, _ = predict_complementary_hybrid(
            frame,
            frame.prediction_v17.to_numpy(float),
            frame.prediction_dl.to_numpy(float),
            deployed_gate,
        )
        # A one-row evaluation is used only to reuse the fixed-fit helper.
        final_pre, final_models, _, _, final_center, final_histories = fit_fixed(
            frame, frame.iloc[:1], final_base, final_base[:1], selected
        )
        bundle = OUTPUT / "final_bundle"
        bundle.mkdir()
        write_json(bundle / "preprocessor.json", final_pre.to_state_dict())
        model_files: list[str] = []
        for seed, net in zip(SEEDS, final_models):
            filename = f"model_seed_{seed}.npz"
            save_model_npz(net, bundle / filename)
            model_files.append(filename)
        final_payload = {
            "kind": "f_regime_reverse_residual_v1",
            "scope": "F",
            "training_season": 2024,
            "training_rows": int(len(frame)),
            "uses_privileged_labels_at_inference": False,
            "include_reverse_heads": True,
            "strength": float(selected["strength"]),
            "correction_center": final_center,
            "model_config": final_models[0].config.to_dict(),
            "model_files": model_files,
            "selected_epoch": int(selected["selected_epoch"]),
            "histories": final_histories,
        }
        write_json(bundle / "manifest.json", final_payload)

    payload = {
        "decision": "PASS" if pass_gate else "FAIL",
        "cause_hypothesis": "post-2022 F reverse-label/coordinate regime unification",
        "selected": selected,
        "development_candidates": candidates,
        "audit": audit_payload,
        "reverse_head_brier": reverse_brier,
        "final_bundle": final_payload,
    }
    write_json(OUTPUT / "selection.json", payload)
    report = [
        "# F post-2022 reverse-state residual",
        "",
        f"- decision: **{payload['decision']}**",
        f"- selected mode: `{selected['mode']}`",
        f"- validation delta Brier: `{selected['delta_brier']:+.10f}`",
        f"- forward-audit delta Brier: `{audit_payload['delta_brier']:+.10f}`",
        f"- reverse-head Brier: `{reverse_brier:.10f}`",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"decision={payload['decision']} audit_delta={audit_payload['delta_brier']:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
