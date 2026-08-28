"""Row-independent inference for v16 = Stage15 cause-shrink plus a second latent pass."""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TEST_PATH = os.path.join(HERE, "data", "test.csv")
OUT_DIR = os.path.join(HERE, "output")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")
ID = "row_id"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.02, 0.98)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_features(df, prev_cols, cat_maps):
    df = df.copy()
    for column, mapping in cat_maps.items():
        df[column] = df[column].map(mapping).astype("float32")
    previous_form = df[prev_cols].mean(axis=1, skipna=True)
    previous_form = previous_form.fillna(
        pd.to_numeric(df["asof_pitcher_success_rate"], errors="coerce")
    ).fillna(0.5)
    df["prev_form"] = previous_form.clip(0.05, 0.95)
    df["rel_career_vs_recent"] = (
        pd.to_numeric(df["asof_pitcher_success_rate"], errors="coerce") - df["prev_form"]
    )
    df["rel_batter_vs_pitcher"] = (
        pd.to_numeric(df["asof_batter_success_rate"], errors="coerce") - df["prev_form"]
    )
    df["rel_prev1_vs_prev5"] = (
        pd.to_numeric(df["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(df["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    pitcher_hand = pd.to_numeric(df["pitcher_hand"], errors="coerce")
    batter_hand = pd.to_numeric(df["batter_hand"], errors="coerce")
    df["hand_matchup"] = pitcher_hand * 4 + batter_hand
    return df


def apply_encodings(df, enc_specs):
    df = df.copy()
    df["inning_b"] = pd.to_numeric(df["inning"], errors="coerce").clip(upper=10)
    row_count = len(df)
    for name, keys in enc_specs:
        table = pd.read_csv(os.path.join(MODEL_DIR, "enc_%s.csv" % name), encoding="utf-8")
        df = df.merge(table, on=list(keys), how="left", sort=False)
        if len(df) != row_count:
            raise RuntimeError("encoding lookup changed row count: %s" % name)
        df[name] = df[name].fillna(0.0)
    return df


def apply_season_state(df, config, entity):
    table = pd.read_csv(os.path.join(MODEL_DIR, config["snapshot_file"]), encoding="utf-8")
    entity_id = "%s_id" % entity
    if table[entity_id].duplicated().any():
        raise RuntimeError("duplicate %s snapshot key" % entity)
    row_count = len(df)
    df = df.merge(table, on=entity_id, how="left", sort=False)
    if len(df) != row_count:
        raise RuntimeError("%s snapshot lookup changed row count" % entity)
    career_n = pd.to_numeric(df["asof_%s_n" % entity], errors="coerce")
    career_rate = pd.to_numeric(df["asof_%s_success_rate" % entity], errors="coerce")
    prior_n_name = "prior_%s_n" % entity
    prior_success_name = "prior_%s_success" % entity
    prior_n = pd.to_numeric(df[prior_n_name], errors="coerce")
    prior_success = pd.to_numeric(df[prior_success_name], errors="coerce")
    career_success = np.rint(career_n * career_rate)
    career_success = np.where(career_n == 0.0, 0.0, career_success)
    season_n = career_n - prior_n
    season_success = career_success - prior_success
    valid = (
        np.isfinite(prior_n) & (prior_n > 0.0) & np.isfinite(season_n)
        & (season_n >= 0.0) & np.isfinite(season_success)
        & (season_success >= 0.0) & (season_success <= season_n)
    )
    season_rate = np.divide(
        season_success, season_n, out=np.zeros(len(df), dtype="float64"),
        where=valid & (season_n > 0.0),
    )
    prior_rate = np.divide(
        prior_success, prior_n, out=np.zeros(len(df), dtype="float64"), where=valid
    )
    smoothing = float(config["smoothing_k"])
    delta = np.where(
        valid, season_n / (season_n + smoothing) * (season_rate - prior_rate), np.nan
    )
    delta = np.where(valid & (season_n == 0.0), 0.0, delta)
    df[config["count_feature"]] = np.where(valid, season_n, np.nan).astype("float32")
    df[config["delta_feature"]] = delta.astype("float32")
    return df.drop(columns=[prior_n_name, prior_success_name])


def load_predictor():
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as file:
        meta = json.load(file)
    boosters = [
        lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_%d.txt" % index))
        for index in range(int(meta["n_models"]))
    ]
    cat_models = []
    for index in meta["ordered_catboost"]["seeds"]:
        model = CatBoostClassifier()
        model.load_model(os.path.join(MODEL_DIR, "catboost_%d.cbm" % index))
        cat_models.append(model)
    return meta, boosters, cat_models


def add_catboost_columns(features, raw, meta):
    result = features.copy()
    def integer_token(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.round().astype("Int64").astype("string")

    pitcher_hand = integer_token(raw["pitcher_hand"])
    batter_hand = integer_token(raw["batter_hand"])
    balls = integer_token(raw["balls_before"])
    strikes = integer_token(raw["strikes_before"])
    outs = integer_token(raw["outs_before"])
    pitcher_id = integer_token(raw["pitcher_id"])
    # Positional assignment is required for single-row/shuffled inference.
    result["hand_matchup_cat"] = (
        pitcher_hand + ":" + batter_hand
    ).to_numpy()
    result["count_state_cat"] = (
        balls + ":" + strikes + ":" + outs
    ).to_numpy()
    result["pitcher_count_cat"] = (
        pitcher_id + ":" + balls + ":" + strikes
    ).to_numpy()
    result["pitcher_batter_hand_cat"] = (
        pitcher_id + ":" + batter_hand
    ).to_numpy()
    config = meta["ordered_catboost"]
    mapped = set(meta["cat_maps"])
    composites = {
        "hand_matchup_cat", "count_state_cat", "pitcher_count_cat",
        "pitcher_batter_hand_cat",
    }
    for column in config["categorical_columns"]:
        if column not in mapped and column not in composites:
            result[column] = integer_token(result[column])
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    missing = [column for column in config["features"] if column not in result]
    if missing:
        raise RuntimeError("missing CatBoost features: %s" % missing)
    return result[config["features"]]


def latent_score(raw, config):
    """Row-local lookup over a frozen pitcher x context surface.

    Serves both latent passes; the caller supplies the config block. Row order is
    restored explicitly so a shuffled or single-row test frame gives identical output.
    """
    table = pd.read_csv(os.path.join(MODEL_DIR, config["lookup_file"]), encoding="utf-8")
    keys = config["keys"]
    if table.duplicated(keys).any():
        raise RuntimeError("duplicate latent lookup key")
    query = raw[keys].copy()
    for key in keys:
        query[key] = pd.to_numeric(query[key], errors="coerce")
    query["_row_position"] = np.arange(len(query), dtype="int64")
    merged = query.merge(table, on=keys, how="left", sort=False)
    if len(merged) != len(raw):
        raise RuntimeError("latent lookup changed row count")
    merged = merged.sort_values("_row_position", kind="stable")
    return pd.to_numeric(merged[config["score_column"]], errors="coerce").fillna(0.0).to_numpy(dtype="float64")


def _snapshot_for_rows(raw, filename, key):
    table = pd.read_csv(os.path.join(MODEL_DIR, filename), encoding="utf-8")
    if table[key].duplicated().any():
        raise RuntimeError("duplicate Stage15 snapshot key: %s" % key)
    query = raw[[key]].copy()
    query["_row_position"] = np.arange(len(query), dtype="int64")
    merged = query.merge(table, on=key, how="left", sort=False)
    if len(merged) != len(raw):
        raise RuntimeError("Stage15 snapshot changed row count: %s" % key)
    return merged.sort_values("_row_position", kind="stable").reset_index(drop=True)


def _rate_state(current_n, current_rate, prior_n, prior_rate, k=20.0):
    current_count = np.rint(current_n * current_rate)
    prior_count = np.rint(prior_n * prior_rate)
    n = current_n - prior_n
    event = current_count - prior_count
    valid = (
        np.isfinite(current_n) & np.isfinite(current_rate)
        & np.isfinite(prior_n) & np.isfinite(prior_rate)
        & (prior_n >= 0.0) & (n >= 0.0) & (event >= 0.0) & (event <= n)
    )
    p0 = np.clip(prior_rate, 1e-4, 1.0 - 1e-4)
    with np.errstate(invalid="ignore", divide="ignore"):
        posterior = np.divide(event + k * p0, n + k, out=np.zeros_like(event), where=valid)
        delta = np.where(valid, posterior - p0, 0.0)
        a = event + k * p0
        b = n - event + k * (1.0 - p0)
        variance = np.where(
            valid,
            (a * b) / np.maximum((a + b) ** 2 * (a + b + 1.0), 1e-12),
            0.0,
        )
        surprise = np.where(
            valid,
            (event - n * p0) / np.sqrt(n * p0 * (1.0 - p0) + 1.0),
            0.0,
        )
    return delta, variance, np.clip(surprise, -5.0, 5.0), valid.astype(float)


def stage15_cause_features(raw, meta):
    config = meta["stage15_cause_shrink"]
    pitcher = _snapshot_for_rows(raw, config["pitcher_snapshot_file"], "pitcher_id")
    batter = _snapshot_for_rows(raw, config["batter_snapshot_file"], "batter_id")
    output = {}

    pitcher_rates = {
        "reverse": "asof_pitcher_reverse_rate",
        "middle": "asof_pitcher_middle_rate",
        "ball": "asof_pitcher_ball_rate",
        "strike": "asof_pitcher_strike_rate",
        "fastball": "asof_pitcher_fastball_rate",
        "breaking": "asof_pitcher_breaking_rate",
        "offspeed": "asof_pitcher_offspeed_rate",
    }
    cause_names = ["reverse", "middle", "ball", "strike"]
    mix_names = ["fastball", "breaking", "offspeed"]
    current_cause_n = pd.to_numeric(raw["asof_pitcher_n"], errors="coerce").to_numpy(float)
    prior_cause_n = pd.to_numeric(pitcher["prior_asof_pitcher_n"], errors="coerce").to_numpy(float)
    current_mix_n = pd.to_numeric(raw["asof_pitcher_pitchmix_n"], errors="coerce").to_numpy(float)
    prior_mix_n = pd.to_numeric(pitcher["prior_asof_pitcher_pitchmix_n"], errors="coerce").to_numpy(float)
    for name, rate_column in pitcher_rates.items():
        current_n = current_cause_n if name in cause_names else current_mix_n
        prior_n = prior_cause_n if name in cause_names else prior_mix_n
        delta, variance, surprise, valid = _rate_state(
            current_n,
            pd.to_numeric(raw[rate_column], errors="coerce").to_numpy(float),
            prior_n,
            pd.to_numeric(pitcher["prior_" + rate_column], errors="coerce").to_numpy(float),
        )
        output[name + "_delta"] = delta
        output[name + "_variance"] = variance
        output[name + "_surprise"] = surprise
        output[name + "_valid"] = valid

    output["season_pitch_count"] = np.log1p(
        np.where(np.isfinite(current_cause_n - prior_cause_n), np.maximum(current_cause_n - prior_cause_n, 0.0), 0.0)
    )
    output["pitchmix_season_count"] = np.log1p(
        np.where(np.isfinite(current_mix_n - prior_mix_n), np.maximum(current_mix_n - prior_mix_n, 0.0), 0.0)
    )
    bdelta, bvariance, bsurprise, bvalid = _rate_state(
        pd.to_numeric(raw["asof_batter_n"], errors="coerce").to_numpy(float),
        pd.to_numeric(raw["asof_batter_middle_rate"], errors="coerce").to_numpy(float),
        pd.to_numeric(batter["prior_asof_batter_n"], errors="coerce").to_numpy(float),
        pd.to_numeric(batter["prior_asof_batter_middle_rate"], errors="coerce").to_numpy(float),
    )
    output["batter_middle_delta"] = bdelta
    output["batter_middle_variance"] = bvariance
    output["batter_middle_surprise"] = bsurprise
    output["batter_middle_valid"] = bvalid
    frame = pd.DataFrame(output)
    frame["cause_delta_l1"] = frame[[name + "_delta" for name in cause_names]].abs().sum(axis=1)
    frame["pitchmix_delta_l1"] = frame[[name + "_delta" for name in mix_names]].abs().sum(axis=1)
    # Research artifacts were fitted on the exact float32 cause-feature frame.
    return frame[config["feature_columns"]].fillna(0.0).astype("float32")


def stage15_correction(raw, p_v8, cat_sleeve, latent_probability, meta):
    config = meta["stage15_cause_shrink"]
    features = stage15_cause_features(raw, meta).to_numpy(dtype="float64")
    mean = np.asarray(config["feature_mean"], dtype="float64")
    scale = np.asarray(config["feature_scale"], dtype="float64")
    standardized = np.nan_to_num((features - mean) / scale)
    ridge = standardized @ np.asarray(config["ridge_coef"], dtype="float64")
    axes = np.column_stack([
        np.ones(len(raw), dtype="float64"), p_v8, cat_sleeve, latent_probability,
    ])
    orthogonal = ridge - axes @ np.asarray(config["projection"], dtype="float64")
    correction = float(config["prediction_coefficient"]) * orthogonal
    bound = float(config["correction_clip"])
    return np.clip(correction, -bound, bound)


def predict(test, meta, boosters, cat_models):
    features = build_features(test, meta["prev_cols"], meta["cat_maps"])
    features = apply_encodings(features, meta.get("enc_specs", []))
    features = apply_season_state(features, meta["season_success"], "pitcher")
    features = apply_season_state(features, meta["batter_season"], "batter")
    model_matrix = features[meta["features"]].apply(pd.to_numeric, errors="coerce")
    initial = logit(features["prev_form"].to_numpy())
    raw_scores = np.vstack(
        [booster.predict(model_matrix, raw_score=True) for booster in boosters]
    )
    intercept = float(meta["intercept"])
    shrink = float(meta["shrink"])
    ensemble_linear = initial + intercept + shrink * raw_scores.mean(axis=0)
    p_v6 = sigmoid(ensemble_linear)
    pitcher_delta = pd.to_numeric(
        features[meta["season_success"]["delta_feature"]], errors="coerce"
    ).to_numpy(dtype="float64")
    batter_delta = pd.to_numeric(
        features[meta["batter_season"]["delta_feature"]], errors="coerce"
    ).to_numpy(dtype="float64")
    correction = np.nan_to_num(
        float(meta["season_delta_logit_beta"]) * pitcher_delta,
        nan=0.0, posinf=0.0, neginf=0.0,
    ) + np.nan_to_num(
        float(meta["batter_delta_logit_beta"]) * batter_delta,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    p_v8 = sigmoid(ensemble_linear + correction)

    cat_matrix = add_catboost_columns(features, test, meta)
    cat_indices = meta["ordered_catboost"]["categorical_indices"]
    cat_pool = Pool(cat_matrix, cat_features=cat_indices, baseline=initial)
    cat_corrections = []
    for seed, model in zip(meta["ordered_catboost"]["seeds"], cat_models):
        raw_with_baseline = np.asarray(model.predict(cat_pool, prediction_type="RawFormulaVal"), dtype="float64")
        residual = raw_with_baseline - initial
        p_cat = sigmoid(initial + intercept + shrink * residual)
        p_v6_seed = sigmoid(initial + intercept + shrink * raw_scores[int(seed)])
        cat_corrections.append(p_cat - p_v6_seed)
    sleeve = float(meta["ordered_catboost"]["sleeve_weight"]) * np.mean(cat_corrections, axis=0)
    latent = latent_score(test, meta["latent_command_style"])
    latent_probability = float(meta["latent_command_style"]["coefficient"]) * latent
    prediction_v10 = np.clip(
        np.asarray(p_v8 + sleeve + latent_probability, dtype="float64"),
        1e-6, 1.0 - 1e-6,
    )
    correction15 = stage15_correction(test, p_v8, sleeve, latent_probability, meta)
    prediction_v15 = np.clip(prediction_v10 + correction15, 1e-6, 1.0 - 1e-6)
    # Second latent pass: the same pitcher x context surface refitted on what the
    # v10 residual still carries. Rank 4 / K=100 / 0.2 were chosen on 2022 alone.
    latent2_config = meta["latent2_pitcher_context"]
    correction2 = float(latent2_config["coefficient"]) * latent_score(test, latent2_config)
    prediction_v16 = np.clip(prediction_v15 + correction2, 1e-6, 1.0 - 1e-6)
    # v17: population count x hand surface. Same row-local lookup helper; the cell
    # offsets are per-season centred so only the shape transfers, never the fit
    # season's global residual level.
    surface_config = meta["count_hand_surface"]
    correction3 = float(surface_config["coefficient"]) * latent_score(test, surface_config)
    return np.clip(prediction_v16 + correction3, 1e-6, 1.0 - 1e-6)


def main():
    print("Load model artifacts...")
    meta, boosters, cat_models = load_predictor()
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    prediction = predict(test, meta, boosters, cat_models)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite prediction")
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({ID: test[ID].to_numpy(), "control_success": prediction}).to_csv(
        OUT_PATH, index=False, encoding="utf-8"
    )
    print("Saved %s rows=%d mean=%.6f min=%.6f max=%.6f" % (
        OUT_PATH, len(prediction), prediction.mean(), prediction.min(), prediction.max()
    ))


if __name__ == "__main__":
    main()
