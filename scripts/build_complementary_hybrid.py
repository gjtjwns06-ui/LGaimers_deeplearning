#!/usr/bin/env python3
"""Build a forward-validated conditional v17 + recent-DL fusion gate.

The stronger v17 model remains the anchor.  The learned surface only chooses
how far to move toward the recent-regime deep model for a small, row-local
state.  Candidate surfaces are selected on 2024 months 6-7 after fitting
months 3-5, then refitted through month 7 and audited on months 8-10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_control.complementary_gate import (  # noqa: E402
    ComplementaryGateArtifact,
    predict_complementary_hybrid,
    save_complementary_gate,
)


SURFACE_KEYS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
CANDIDATE_KEYSETS = [
    ["game_type"],
    ["game_type", "direction"],
    ["game_type", "count"],
    ["game_type", "count", "direction"],
]


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
        default=PROJECT_ROOT / "artifacts/complementary_hybrid_v1",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _official_score(y_true: np.ndarray, prediction: np.ndarray) -> float:
    target_rate = float(np.mean(y_true))
    denominator = target_rate * (1.0 - target_rate)
    brier = float(np.mean(np.square(y_true - prediction)))
    return max(0.0, 100000.0 * (1.0 - brier / denominator))


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    y_true = frame["control_success"].to_numpy(dtype=np.float64)
    probability = np.asarray(prediction, dtype=np.float64)
    return {
        "rows": int(len(frame)),
        "target_rate": float(y_true.mean()),
        "prediction_mean": float(probability.mean()),
        "brier": float(np.mean(np.square(y_true - probability))),
        "competition_score": _official_score(y_true, probability),
    }


def _fit_v17_forward_surface(fit: pd.DataFrame) -> dict[tuple[int, ...], float]:
    frame = fit.copy()
    frame["residual"] = frame["control_success"] - frame["prediction_v16"]
    frame["cell"] = list(
        zip(*[frame[column].astype(int) for column in SURFACE_KEYS])
    )
    pieces: list[pd.DataFrame] = []
    for _, season_frame in frame.groupby("season", sort=True):
        season_mean = float(season_frame["residual"].mean())
        statistics = season_frame.groupby("cell")["residual"].agg(["mean", "size"])
        statistics["deviation"] = statistics["mean"] - season_mean
        pieces.append(statistics[["deviation", "size"]])
    pooled = pd.concat(pieces)
    deviation = pooled.groupby(level=0).apply(
        lambda values: float(
            np.average(values["deviation"], weights=values["size"])
        ),
        include_groups=False,
    )
    size = pooled.groupby(level=0)["size"].sum().astype(float)
    return (size / (size + 100.0) * deviation).to_dict()


def _load_aligned_oof(args: argparse.Namespace) -> pd.DataFrame:
    reports = args.research_root.resolve() / "reports"
    discovery_path = reports / "stage22_trackman_hidden_interactions_discovery_oof.parquet"
    audit_path = reports / "stage22_trackman_hidden_interactions_audit_oof.parquet"
    for path in (args.train_csv, args.dl_oof, discovery_path, audit_path):
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())

    raw_columns = ["row_id", "game_month", "game_type", *SURFACE_KEYS]
    raw = pd.read_csv(
        args.train_csv.resolve(),
        usecols=raw_columns,
        encoding="utf-8-sig",
        low_memory=False,
    )
    discovery = pd.read_parquet(
        discovery_path,
        columns=["row_id", "season", "control_success", "prediction_v16"],
    ).merge(raw[["row_id", *SURFACE_KEYS]], on="row_id", validate="one_to_one")
    if set(discovery["season"].astype(int)) != {2021, 2022}:
        raise RuntimeError("v17 forward surface must be fitted on 2021 and 2022 OOF")
    surface = _fit_v17_forward_surface(discovery)

    audit = pd.read_parquet(
        audit_path,
        columns=[
            "row_id",
            "season",
            "control_success",
            "_game_cluster",
            "prediction_v16",
        ],
    ).merge(raw, on="row_id", validate="one_to_one")
    if set(audit["season"].astype(int)) != {2024}:
        raise RuntimeError("conditional hybrid audit must contain only 2024 OOF")
    audit["cell"] = list(
        zip(*[audit[column].astype(int) for column in SURFACE_KEYS])
    )
    audit["prediction_v17"] = np.clip(
        audit["prediction_v16"]
        + 0.5 * audit["cell"].map(surface).fillna(0.0),
        1e-6,
        1.0 - 1e-6,
    )

    dl = pd.read_csv(
        args.dl_oof.resolve(), usecols=["row_id", "prediction"]
    ).rename(columns={"prediction": "prediction_dl"})
    if dl["row_id"].duplicated().any():
        raise RuntimeError("recent DL OOF row IDs must be unique")
    frame = audit.merge(dl, on="row_id", how="inner", validate="one_to_one")
    if len(frame) != len(audit):
        raise RuntimeError("recent DL OOF does not cover the v17 audit rows")
    frame["game_type"] = frame["game_type"].astype(str).str.upper()
    frame["count"] = (
        frame["balls_before"].astype(int).astype(str)
        + "-"
        + frame["strikes_before"].astype(int).astype(str)
    )
    frame["direction"] = np.where(
        frame["prediction_dl"] >= frame["prediction_v17"], "up", "down"
    )
    return frame


def _oracle_weight(frame: pd.DataFrame) -> float:
    y_true = frame["control_success"].to_numpy(dtype=np.float64)
    anchor = frame["prediction_v17"].to_numpy(dtype=np.float64)
    direction = frame["prediction_dl"].to_numpy(dtype=np.float64) - anchor
    denominator = float(np.dot(direction, direction))
    if denominator <= 0.0:
        return 0.0
    return float(-np.dot(anchor - y_true, direction) / denominator)


def _fit_gate(
    frame: pd.DataFrame,
    keys: list[str],
    shrink_rows: int,
    maximum_weight: float,
) -> ComplementaryGateArtifact:
    fallback = float(np.clip(_oracle_weight(frame), 0.0, maximum_weight))
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = keys[0] if len(keys) == 1 else keys
    for group_key, group in frame.groupby(grouper, observed=True, sort=True):
        key_values = (group_key,) if len(keys) == 1 else tuple(group_key)
        raw_weight = _oracle_weight(group)
        shrinkage = len(group) / (len(group) + shrink_rows)
        weight = float(
            np.clip(
                fallback + shrinkage * (raw_weight - fallback),
                0.0,
                maximum_weight,
            )
        )
        row = dict(zip(keys, key_values))
        row.update(
            {
                "weight": weight,
                "fit_rows": int(len(group)),
                "raw_oracle_weight": float(raw_weight),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    # Runtime has one frozen key contract.  Less detailed development candidates
    # are evaluated locally but only the full selected contract is deployable.
    if keys != ["game_type", "count", "direction"]:
        return ComplementaryGateArtifact(
            table=table,
            fallback_weight=fallback,
            maximum_weight=maximum_weight,
            keys=tuple(keys),
        )
    return ComplementaryGateArtifact(
        table=table,
        fallback_weight=fallback,
        maximum_weight=maximum_weight,
    )


def _gate_predictions(
    frame: pd.DataFrame, artifact: ComplementaryGateArtifact
) -> tuple[np.ndarray, np.ndarray]:
    # The runtime helper enforces the final key contract.  Generic development
    # candidates use the same stable positional merge directly.
    if tuple(artifact.keys) == ("game_type", "count", "direction"):
        return predict_complementary_hybrid(
            frame,
            frame["prediction_v17"].to_numpy(float),
            frame["prediction_dl"].to_numpy(float),
            artifact,
        )
    query = frame[[*artifact.keys]].copy()
    query["_row_position"] = np.arange(len(query), dtype=np.int64)
    merged = query.merge(
        artifact.table[[*artifact.keys, "weight"]],
        on=list(artifact.keys),
        how="left",
        sort=False,
    ).sort_values("_row_position", kind="stable")
    weights = merged["weight"].fillna(artifact.fallback_weight).to_numpy(float)
    anchor = frame["prediction_v17"].to_numpy(float)
    challenger = frame["prediction_dl"].to_numpy(float)
    return np.clip(anchor + weights * (challenger - anchor), 1e-6, 1.0 - 1e-6), weights


def _fixed_predictions(frame: pd.DataFrame, dl_weight: float) -> np.ndarray:
    anchor = frame["prediction_v17"].to_numpy(float)
    challenger = frame["prediction_dl"].to_numpy(float)
    return anchor + dl_weight * (challenger - anchor)


def _cluster_bootstrap_score_delta(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    reference: np.ndarray,
    replicates: int,
) -> dict[str, object]:
    y_true = frame["control_success"].to_numpy(dtype=np.float64)
    loss_delta = np.square(y_true - candidate) - np.square(y_true - reference)
    grouped = pd.DataFrame(
        {
            "cluster": frame["_game_cluster"].to_numpy(),
            "loss_delta": loss_delta,
        }
    ).groupby("cluster")["loss_delta"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(20260827)
    samples = np.empty(replicates, dtype=np.float64)
    cluster_count = len(grouped)
    for index in range(replicates):
        choice = rng.integers(0, cluster_count, size=cluster_count)
        samples[index] = sums[choice].sum() / sizes[choice].sum()
    denominator = float(y_true.mean() * (1.0 - y_true.mean()))
    score_delta = -100000.0 * float(loss_delta.mean()) / denominator
    score_samples = -100000.0 * samples / denominator
    return {
        "clusters": int(cluster_count),
        "replicates": int(replicates),
        "score_delta": score_delta,
        "score_delta_ci90": [
            float(np.quantile(score_samples, 0.05)),
            float(np.quantile(score_samples, 0.95)),
        ],
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_aligned_oof(args)

    development_train = frame.loc[frame["game_month"].le(5)].copy()
    development_validation = frame.loc[
        frame["game_month"].between(6, 7)
    ].copy()
    audit_train = frame.loc[frame["game_month"].le(7)].copy()
    audit = frame.loc[frame["game_month"].ge(8)].copy()
    if min(map(len, (development_train, development_validation, audit_train, audit))) == 0:
        raise RuntimeError("temporal development split is empty")

    candidate_rows: list[dict[str, object]] = []
    for keys in CANDIDATE_KEYSETS:
        for shrink_rows in (250, 500, 1000, 2000, 5000, 10000, 20000):
            for maximum_weight in (0.10, 0.15, 0.20):
                artifact = _fit_gate(
                    development_train, keys, shrink_rows, maximum_weight
                )
                probability, weights = _gate_predictions(
                    development_validation, artifact
                )
                metrics = _metrics(development_validation, probability)
                candidate_rows.append(
                    {
                        "keys": keys,
                        "shrink_rows": shrink_rows,
                        "maximum_weight": maximum_weight,
                        "fallback_weight": artifact.fallback_weight,
                        "mean_validation_weight": float(weights.mean()),
                        "validation": metrics,
                    }
                )
    selected = min(candidate_rows, key=lambda item: item["validation"]["brier"])
    selected_keys = list(selected["keys"])
    selected_shrink = int(selected["shrink_rows"])
    selected_cap = float(selected["maximum_weight"])
    if selected_keys != ["game_type", "count", "direction"]:
        raise RuntimeError(
            "development selected a non-deployable key contract; inspect candidates"
        )

    audit_artifact = _fit_gate(
        audit_train, selected_keys, selected_shrink, selected_cap
    )
    audit_probability, audit_weights = _gate_predictions(audit, audit_artifact)
    audit_v17 = audit["prediction_v17"].to_numpy(float)
    audit_fixed = _fixed_predictions(audit, 0.05)
    audit_metrics = {
        "v17": _metrics(audit, audit_v17),
        "fixed_v17_95_dl_05": _metrics(audit, audit_fixed),
        "conditional_gate": _metrics(audit, audit_probability),
        "mean_gate_weight": float(audit_weights.mean()),
        "gate_weight_min": float(audit_weights.min()),
        "gate_weight_max": float(audit_weights.max()),
    }
    audit_metrics["delta_brier_gate_vs_v17"] = float(
        audit_metrics["conditional_gate"]["brier"] - audit_metrics["v17"]["brier"]
    )
    audit_metrics["delta_brier_gate_vs_fixed"] = float(
        audit_metrics["conditional_gate"]["brier"]
        - audit_metrics["fixed_v17_95_dl_05"]["brier"]
    )
    audit_metrics["bootstrap_vs_v17"] = _cluster_bootstrap_score_delta(
        audit, audit_probability, audit_v17, args.bootstrap_replicates
    )
    audit_metrics["bootstrap_vs_fixed"] = _cluster_bootstrap_score_delta(
        audit, audit_probability, audit_fixed, args.bootstrap_replicates
    )
    pass_gate = bool(
        audit_metrics["delta_brier_gate_vs_v17"] < -1e-6
        and audit_metrics["delta_brier_gate_vs_fixed"] < -1e-6
    )

    final_artifact = _fit_gate(frame, selected_keys, selected_shrink, selected_cap)
    final_probability, final_weights = _gate_predictions(frame, final_artifact)
    save_complementary_gate(
        output_dir / "gate",
        final_artifact,
        metadata={
            "fit_season": 2024,
            "fit_source": "aligned leakage-safe v17 and recent-DL OOF",
            "selection_train_months": [3, 4, 5],
            "selection_validation_months": [6, 7],
            "forward_audit_train_months": [3, 4, 5, 6, 7],
            "forward_audit_months": [8, 9, 10],
            "shrink_rows": selected_shrink,
            "decision": "PASS" if pass_gate else "FAIL",
        },
    )
    audit_output = audit[
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
    audit_output["gate_weight"] = audit_weights
    audit_output["prediction_conditional"] = audit_probability
    audit_output.to_parquet(output_dir / "temporal_audit_predictions.parquet", index=False)

    payload = {
        "decision": "PASS" if pass_gate else "FAIL",
        "formula": "p_v17 + weight(game_type,count,direction) * (p_dl - p_v17)",
        "selection": selected,
        "development_candidates": candidate_rows,
        "audit": audit_metrics,
        "full_2024_fit_diagnostic_only": {
            "v17": _metrics(frame, frame["prediction_v17"].to_numpy(float)),
            "recent_dl": _metrics(frame, frame["prediction_dl"].to_numpy(float)),
            "fixed_v17_95_dl_05": _metrics(frame, _fixed_predictions(frame, 0.05)),
            "conditional_gate_in_sample": _metrics(frame, final_probability),
            "mean_gate_weight": float(final_weights.mean()),
        },
        "rows": int(len(frame)),
        "selected_gate_rows": int(len(final_artifact.table)),
    }
    _write_json(output_dir / "selection.json", payload)

    audit_gate = audit_metrics["conditional_gate"]
    audit_base = audit_metrics["v17"]
    audit_simple = audit_metrics["fixed_v17_95_dl_05"]
    report = [
        "# v17 + recent DL conditional complementarity gate",
        "",
        f"- decision: **{'PASS' if pass_gate else 'FAIL'}**",
        "- anchor: v17",
        "- challenger: post-2022 recent-regime DL ensemble",
        "- formula: `p_v17 + w(state) * (p_dl - p_v17)`",
        f"- state: `{'+'.join(selected_keys)}`",
        f"- selected cap / row shrink: `{selected_cap:.2f}` / `{selected_shrink}`",
        "",
        "## Forward audit (2024 months 8-10)",
        "",
        "| Candidate | Brier | Score |",
        "|---|---:|---:|",
        f"| v17 | {audit_base['brier']:.10f} | {audit_base['competition_score']:.2f} |",
        f"| fixed 95:5 | {audit_simple['brier']:.10f} | {audit_simple['competition_score']:.2f} |",
        f"| conditional gate | **{audit_gate['brier']:.10f}** | **{audit_gate['competition_score']:.2f}** |",
        "",
        f"- delta Brier vs v17: `{audit_metrics['delta_brier_gate_vs_v17']:+.10f}`",
        f"- delta Brier vs fixed 95:5: `{audit_metrics['delta_brier_gate_vs_fixed']:+.10f}`",
        f"- mean audit DL weight: `{audit_metrics['mean_gate_weight']:.4f}`",
        "",
        "The final table is refitted on all leakage-safe 2024 OOF rows only after",
        "the state, cap, and shrinkage were frozen by the forward audit protocol.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"decision={'PASS' if pass_gate else 'FAIL'} "
        f"keys={'+'.join(selected_keys)} cap={selected_cap:.2f} shrink={selected_shrink} "
        f"audit_delta_v17={audit_metrics['delta_brier_gate_vs_v17']:+.10f} "
        f"audit_delta_fixed={audit_metrics['delta_brier_gate_vs_fixed']:+.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
