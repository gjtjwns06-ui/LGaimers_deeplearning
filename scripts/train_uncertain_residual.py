#!/usr/bin/env python3
"""Train and forward-audit a v17 residual net only near probability 0.5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_complementary_hybrid as complementary  # noqa: E402
from lg_control.hybrid_residual import (  # noqa: E402
    BoundedLogitResidualNet,
    HybridResidualModelConfig,
    HybridResidualTrainConfig,
    predict_bounded_residual,
    save_model_npz,
    train_bounded_residual,
    train_bounded_residual_fixed_epochs,
)
from lg_control.uncertain_residual import UncertainResidualPreprocessor  # noqa: E402


SEEDS = (42, 2026, 3407)
THRESHOLD = 0.025
GATE_KEYS = ["game_type", "count", "direction"]
GATE_SHRINK_ROWS = 250
GATE_CAP = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "data/train.csv")
    parser.add_argument(
        "--research-root",
        type=Path,
        default=PROJECT_ROOT.parent / "LG-aimers-v11-independent-research",
    )
    parser.add_argument(
        "--dl-oof",
        type=Path,
        default=PROJECT_ROOT / "artifacts/recent_regime_outer_ensemble_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/uncertain_residual_v1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y_true = frame["control_success"].to_numpy(dtype=np.float64)
    prediction = np.asarray(probability, dtype=np.float64)
    rate = float(y_true.mean())
    denominator = rate * (1.0 - rate)
    brier = float(np.mean(np.square(y_true - prediction)))
    return {
        "rows": int(len(frame)),
        "target_rate": rate,
        "prediction_mean": float(prediction.mean()),
        "brier": brier,
        "competition_score": max(0.0, 100000.0 * (1.0 - brier / denominator)),
    }


def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    base_args = SimpleNamespace(
        train_csv=args.train_csv,
        research_root=args.research_root,
        dl_oof=args.dl_oof,
    )
    frame = complementary._load_aligned_oof(base_args)
    raw = pd.read_csv(args.train_csv, encoding="utf-8-sig", low_memory=False)
    duplicate = [column for column in raw.columns if column in frame.columns and column != "row_id"]
    raw = raw.drop(columns=duplicate)
    frame = frame.merge(raw, on="row_id", validate="one_to_one")
    frame["near_half"] = np.abs(frame["prediction_v17"] - 0.5) <= THRESHOLD
    return frame


def _model(input_dim: int, cap: float, seed: int) -> BoundedLogitResidualNet:
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


def _train_config(seed: int, epochs: int = 24) -> HybridResidualTrainConfig:
    return HybridResidualTrainConfig(
        epochs=epochs,
        scheduler_epochs=24,
        batch_size=4096,
        eval_batch_size=32768,
        learning_rate=5e-4,
        weight_decay=5e-3,
        correction_l2=0.10,
        mean_correction_penalty=0.10,
        patience=5,
        min_delta=2e-8,
        seed=seed,
        gradient_clip_norm=2.0,
        verbose=False,
    )


def _conditional_prediction(
    fit: pd.DataFrame, validation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    gate = complementary._fit_gate(
        fit, GATE_KEYS, GATE_SHRINK_ROWS, GATE_CAP
    )
    return complementary._gate_predictions(validation, gate)


def _develop_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_focus = train.loc[train["near_half"]].copy()
    validation_focus = validation.loc[validation["near_half"]].copy()
    base_full, _ = _conditional_prediction(train, validation)
    base_focus_v17 = validation_focus["prediction_v17"].to_numpy(float)
    y_train = train_focus["control_success"].to_numpy(dtype=np.float32)
    y_validation = validation_focus["control_success"].to_numpy(dtype=np.float32)
    candidates: list[dict[str, object]] = []

    for include_count in (False, True):
        mode = "count_context" if include_count else "basic"
        preprocessor = UncertainResidualPreprocessor(
            include_count_context=include_count
        ).fit(
            train_focus,
            train_focus["prediction_v17"].to_numpy(float),
            train_focus["prediction_dl"].to_numpy(float),
        )
        train_features = preprocessor.transform(
            train_focus,
            train_focus["prediction_v17"].to_numpy(float),
            train_focus["prediction_dl"].to_numpy(float),
        )
        validation_features = preprocessor.transform(
            validation_focus,
            validation_focus["prediction_v17"].to_numpy(float),
            validation_focus["prediction_dl"].to_numpy(float),
        )
        _write_json(
            output_dir / f"development_preprocessor_{mode}.json",
            preprocessor.to_state_dict(),
        )
        for cap in (0.03, 0.05, 0.08):
            probabilities: list[np.ndarray] = []
            corrections: list[np.ndarray] = []
            seed_rows: list[dict[str, object]] = []
            for seed in SEEDS:
                print(f"develop mode={mode} cap={cap:.2f} seed={seed}", flush=True)
                model = _model(train_features.shape[1], cap, seed)
                config = _train_config(seed)
                result = train_bounded_residual(
                    model,
                    train_features,
                    train_focus["prediction_v17"].to_numpy(np.float32),
                    y_train,
                    validation_features,
                    base_focus_v17.astype(np.float32),
                    y_validation,
                    config=config,
                    device=device,
                )
                probabilities.append(result.probabilities)
                corrections.append(result.corrections)
                seed_rows.append(
                    {
                        "seed": seed,
                        "selected_epoch": result.best_epoch,
                        "metrics": result.best_metrics,
                        "mean_logit_correction": float(result.corrections.mean()),
                        "history": result.history,
                    }
                )
            focus_probability = np.mean(probabilities, axis=0)
            full_probability = base_full.copy()
            full_probability[validation["near_half"].to_numpy()] = focus_probability
            candidate = {
                "mode": mode,
                "include_count_context": include_count,
                "cap": cap,
                "feature_count": int(train_features.shape[1]),
                "selected_epochs": {
                    str(row["seed"]): int(row["selected_epoch"]) for row in seed_rows
                },
                "focus_v17": _metric(validation_focus, base_focus_v17),
                "focus_candidate": _metric(validation_focus, focus_probability),
                "full_conditional_base": _metric(validation, base_full),
                "full_candidate": _metric(validation, full_probability),
                "mean_logit_correction": float(np.mean(corrections)),
                "seeds": seed_rows,
            }
            candidate["delta_focus_vs_v17"] = float(
                candidate["focus_candidate"]["brier"]
                - candidate["focus_v17"]["brier"]
            )
            candidate["delta_full_vs_conditional"] = float(
                candidate["full_candidate"]["brier"]
                - candidate["full_conditional_base"]["brier"]
            )
            candidates.append(candidate)
            print(
                f"candidate mode={mode} cap={cap:.2f} "
                f"delta_focus={candidate['delta_focus_vs_v17']:+.10f} "
                f"delta_full={candidate['delta_full_vs_conditional']:+.10f}",
                flush=True,
            )
    selected = min(candidates, key=lambda item: item["full_candidate"]["brier"])
    return selected, candidates


def _fit_fixed_bundle(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    selected: dict[str, object],
    device: str,
) -> tuple[
    UncertainResidualPreprocessor,
    list[BoundedLogitResidualNet],
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    fit_focus = fit.loc[fit["near_half"]].copy()
    validation_focus = validation.loc[validation["near_half"]].copy()
    preprocessor = UncertainResidualPreprocessor(
        include_count_context=bool(selected["include_count_context"])
    ).fit(
        fit_focus,
        fit_focus["prediction_v17"].to_numpy(float),
        fit_focus["prediction_dl"].to_numpy(float),
    )
    fit_features = preprocessor.transform(
        fit_focus,
        fit_focus["prediction_v17"].to_numpy(float),
        fit_focus["prediction_dl"].to_numpy(float),
    )
    validation_features = preprocessor.transform(
        validation_focus,
        validation_focus["prediction_v17"].to_numpy(float),
        validation_focus["prediction_dl"].to_numpy(float),
    )
    models: list[BoundedLogitResidualNet] = []
    probabilities: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    histories: list[dict[str, object]] = []
    cap = float(selected["cap"])
    for seed in SEEDS:
        epoch = int(selected["selected_epochs"][str(seed)])
        print(f"fixed cap={cap:.2f} seed={seed} epochs={epoch}", flush=True)
        model = _model(fit_features.shape[1], cap, seed)
        config = _train_config(seed, epochs=epoch)
        history = train_bounded_residual_fixed_epochs(
            model,
            fit_features,
            fit_focus["prediction_v17"].to_numpy(np.float32),
            fit_focus["control_success"].to_numpy(np.float32),
            config=config,
            device=device,
        )
        probability, correction = predict_bounded_residual(
            model,
            validation_features,
            validation_focus["prediction_v17"].to_numpy(np.float32),
            batch_size=32768,
            device=device,
        )
        models.append(model)
        probabilities.append(probability)
        corrections.append(correction)
        histories.append({"seed": seed, "epochs": epoch, "history": history})
    return (
        preprocessor,
        models,
        np.mean(probabilities, axis=0),
        np.mean(corrections, axis=0),
        histories,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(args)
    development_train = frame.loc[frame["game_month"].le(5)].copy()
    development_validation = frame.loc[frame["game_month"].between(6, 7)].copy()
    audit_train = frame.loc[frame["game_month"].le(7)].copy()
    audit = frame.loc[frame["game_month"].ge(8)].copy()

    selected, candidates = _develop_candidates(
        development_train, development_validation, output_dir, args.device
    )
    dev_pass = bool(
        selected["delta_focus_vs_v17"] < -1e-6
        and selected["delta_full_vs_conditional"] < -1e-6
    )

    audit_preprocessor, audit_models, audit_focus_probability, audit_correction, audit_histories = _fit_fixed_bundle(
        audit_train, audit, selected, args.device
    )
    audit_base, _ = _conditional_prediction(audit_train, audit)
    audit_probability = audit_base.copy()
    audit_mask = audit["near_half"].to_numpy()
    audit_probability[audit_mask] = audit_focus_probability
    audit_focus = audit.loc[audit["near_half"]].copy()
    audit_payload = {
        "focus_v17": _metric(
            audit_focus, audit_focus["prediction_v17"].to_numpy(float)
        ),
        "focus_candidate": _metric(audit_focus, audit_focus_probability),
        "full_conditional_base": _metric(audit, audit_base),
        "full_candidate": _metric(audit, audit_probability),
        "mean_logit_correction": float(audit_correction.mean()),
        "histories": audit_histories,
    }
    audit_payload["delta_focus_vs_v17"] = float(
        audit_payload["focus_candidate"]["brier"]
        - audit_payload["focus_v17"]["brier"]
    )
    audit_payload["delta_full_vs_conditional"] = float(
        audit_payload["full_candidate"]["brier"]
        - audit_payload["full_conditional_base"]["brier"]
    )
    pass_gate = bool(
        dev_pass
        and audit_payload["delta_focus_vs_v17"] < -1e-6
        and audit_payload["delta_full_vs_conditional"] < -1e-6
    )

    audit_predictions = audit[
        [
            "row_id",
            "game_month",
            "game_type",
            "control_success",
            "_game_cluster",
            "prediction_v17",
            "prediction_dl",
        ]
    ].copy()
    audit_predictions["prediction_conditional_base"] = audit_base
    audit_predictions["near_half"] = audit_mask
    audit_predictions["residual_logit_correction"] = 0.0
    audit_predictions.loc[audit_mask, "residual_logit_correction"] = audit_correction
    audit_predictions["prediction_uncertain_residual"] = audit_probability
    audit_predictions.to_parquet(output_dir / "audit_predictions.parquet", index=False)

    if pass_gate:
        print("audit gate PASS; fitting final full-2024 residual bundle", flush=True)
        empty_validation = frame.loc[frame["near_half"]].iloc[:1].copy()
        # Reuse the fixed fitting helper; the one-row output is discarded.
        final_preprocessor, final_models, _, _, final_histories = _fit_fixed_bundle(
            frame, empty_validation, selected, args.device
        )
        bundle = output_dir / "final_bundle"
        bundle.mkdir()
        _write_json(bundle / "preprocessor.json", final_preprocessor.to_state_dict())
        model_files: list[str] = []
        for seed, model in zip(SEEDS, final_models):
            filename = f"model_seed_{seed}.npz"
            save_model_npz(model, bundle / filename)
            model_files.append(filename)
        model_config = final_models[0].config.to_dict()
        _write_json(
            bundle / "manifest.json",
            {
                "kind": "v17_uncertain_residual_v1",
                "threshold": THRESHOLD,
                "training_season": 2024,
                "training_rows": int(frame["near_half"].sum()),
                "feature_mode": selected["mode"],
                "model_config": model_config,
                "model_files": model_files,
                "selected_epochs": selected["selected_epochs"],
                "histories": final_histories,
            },
        )
    else:
        print("audit gate FAIL; no final residual bundle was fitted", flush=True)

    payload = {
        "decision": "PASS" if pass_gate else "FAIL",
        "threshold": THRESHOLD,
        "development_pass": dev_pass,
        "selected": selected,
        "development_candidates": candidates,
        "audit": audit_payload,
    }
    _write_json(output_dir / "selection.json", payload)
    report = [
        "# Low-confidence v17 residual network",
        "",
        f"- decision: **{'PASS' if pass_gate else 'FAIL'}**",
        f"- selected feature mode: `{selected['mode']}`",
        f"- selected logit correction cap: `{selected['cap']}`",
        f"- development full delta Brier: `{selected['delta_full_vs_conditional']:+.10f}`",
        f"- audit full delta Brier: `{audit_payload['delta_full_vs_conditional']:+.10f}`",
        f"- audit focus delta Brier: `{audit_payload['delta_focus_vs_v17']:+.10f}`",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"decision={'PASS' if pass_gate else 'FAIL'} mode={selected['mode']} "
        f"cap={selected['cap']:.2f} "
        f"audit_delta={audit_payload['delta_full_vs_conditional']:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
