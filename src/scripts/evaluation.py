"""Evaluate fantasy-point models on a held-out IPL season.

Follows the protocol in docs/REPORT.md (Section 6):
  * train on seasons before (test - 1), validate on (test - 1) for early stopping,
    refit on every season before the test season, predict the test season
  * player-level metrics: MAE, RMSE, R2, mean within-match Spearman
  * team-level metrics: for each test match the ILP picks an XI from the players
    who took part, using each model's predictions; the Dream Team is the ILP
    pick using actual points

Usage (from the repo root):
    python src/scripts/evaluation.py [--test-season 2025] [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from LPBaselineModel import compute_baseline_score, select_best_11

DATA_PATH = "src/data/player_match_features_with_external_data.csv"
OUTPUT_DIR = Path("src/model_artifacts/")

NUMERIC_FEATURES = [
    "runs_form_5", "wickets_form_5", "fantasy_points_form_5",
    "strike_rate_form_5", "economy_form_5",
    "matches_played",
    "runs_career_avg", "wickets_career_avg", "fantasy_points_career_avg",
    "fantasy_points_venue_avg", "fantasy_points_vs_opponent_avg",
    "temp_max", "temp_min", "precipitation",
]
CATEGORICAL_FEATURES = ["venue", "team", "opponent", "role", "weather_code"]
TARGET = "fantasy_points"

# Stats-derived role thresholds (per-match averages over earlier matches)
AR_MIN_OVERS = 0.8
AR_MIN_BALLS_FACED = 6


# Data

def fill_missing_teams(df: pd.DataFrame) -> pd.DataFrame:
    # Fielding-only rows have no team: use the player's most common team that season,
    # then the opponent is the other side in the match. Unresolvable rows are dropped.
    season_team = (
        df.dropna(subset=["team"])
        .groupby(["player", "season"])["team"]
        .agg(lambda s: s.mode().iloc[0])
    )
    missing = df["team"].isna()
    keys = list(zip(df.loc[missing, "player"], df.loc[missing, "season"]))
    df.loc[missing, "team"] = [season_team.get(k) for k in keys]

    match_teams = df.dropna(subset=["team"]).groupby("match_id")["team"].unique()

    def other_side(row):
        teams = match_teams.get(row["match_id"], [])
        others = [t for t in teams if t != row["team"]]
        return others[0] if len(others) == 1 else np.nan

    missing_opp = df["opponent"].isna() & df["team"].notna()
    df.loc[missing_opp, "opponent"] = df[missing_opp].apply(other_side, axis=1)

    n_before = len(df)
    df = df.dropna(subset=["team", "opponent"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows whose team could not be resolved.")
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    # matches_played and the stats-derived role, both from earlier matches only
    df = df.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("player")

    df["matches_played"] = g.cumcount()
    prior = lambda col: g[col].transform(lambda s: s.fillna(0).cumsum().shift(1))
    overs_pm = prior("overs_bowled") / df["matches_played"]
    balls_pm = prior("balls_faced") / df["matches_played"]
    stumped = prior("stumpings") > 0
    scraped_wk = df["player_role"].fillna("").str.lower().str.contains("keeper")

    bowls = overs_pm >= AR_MIN_OVERS
    bats = balls_pm >= AR_MIN_BALLS_FACED
    df["role"] = np.select(
        [stumped | scraped_wk, bowls & bats, bowls],
        ["wicketkeeper", "allrounder", "bowler"],
        default="batter",
    )
    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[TARGET])
    df = fill_missing_teams(df)
    df = add_history_features(df)

    for col in CATEGORICAL_FEATURES:
        # Categories come from the whole table so train and test encode identically
        df[col] = pd.Categorical(df[col].astype(str))
    return df


# Models: each returns a prediction for every row of `test`

def predict_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train[TARGET].mean())


def predict_career_avg(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test["fantasy_points_career_avg"].fillna(train[TARGET].mean()).to_numpy()


def predict_lp_heuristic(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test.apply(compute_baseline_score, axis=1).to_numpy()


def make_xgb(n_estimators: int, early_stopping_rounds=None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="mae",
        random_state=42,
    )


def predict_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    val_season = train["season"].max()
    fit, val = train[train["season"] < val_season], train[train["season"] == val_season]

    model = make_xgb(2000, early_stopping_rounds=50)
    model.fit(fit[features], fit[TARGET], eval_set=[(val[features], val[TARGET])], verbose=False)
    n_trees = model.best_iteration + 1
    print(f"  XGBoost: early stopping on {val_season} picked {n_trees} trees; refitting on all training seasons")

    model = make_xgb(n_trees)
    model.fit(train[features], train[TARGET], verbose=False)
    return model.predict(test[features])


MODELS = {
    "Mean baseline": predict_mean,
    "Career average": predict_career_avg,
    "LP heuristic": predict_lp_heuristic,
    "XGBoost": predict_xgboost,
}


# Metrics

def player_metrics(test: pd.DataFrame, preds: np.ndarray) -> dict:
    y = test[TARGET].to_numpy()
    err = preds - y
    by_match = pd.DataFrame({"match_id": test["match_id"].to_numpy(), "pred": preds, "actual": y})
    spearman = by_match.groupby("match_id").apply(
        lambda m: m["pred"].corr(m["actual"], method="spearman") if m["pred"].nunique() > 1 else np.nan,
        include_groups=False,
    )
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "R2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "Spearman": spearman.mean(),  # NaN for constant predictions
    }


def pick_xi(match: pd.DataFrame, score_col: str) -> pd.DataFrame:
    # select_best_11 maximises the "baseline_score" column
    pool = match[["player", "team", TARGET]].assign(baseline_score=match[score_col])
    roles = match[["player", "role"]].astype({"role": str}).drop_duplicates("player")
    xi = select_best_11(pool.reset_index(drop=True), roles)
    if len(xi) != 11:
        raise RuntimeError(f"ILP returned {len(xi)} players for match {match['match_id'].iloc[0]}")
    return xi


def team_results(test: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    rows = []
    for match_id, match in test.groupby("match_id", sort=False):
        if len(match) < 11:  # e.g. abandoned matches
            print(f"Skipping match {match_id}: only {len(match)} players took part.")
            continue
        dream = pick_xi(match, TARGET)
        dream_points = dream[TARGET].sum()
        team_1, team_2 = sorted(match["team"].astype(str).unique())

        for name in predictions:
            xi = pick_xi(match, f"pred_{name}")
            predicted, actual = float(xi["baseline_score"].sum()), xi[TARGET].sum()
            rows.append({
                "Model": name,
                "Match ID": match_id,
                "Match Date": match["date"].iloc[0].date(),
                "Team 1": team_1,
                "Team 2": team_2,
                "Predicted XI": "; ".join(xi["player"]),
                "Dream Team XI": "; ".join(dream["player"]),
                "Predicted Points": round(predicted, 1),
                "Actual Points": actual,
                "Dream Team Points": dream_points,
                "Pct of Dream Team": round(100 * actual / dream_points, 1),
                "Overlap with Dream XI": len(set(xi["player"]) & set(dream["player"])),
                "MAE": round(abs(predicted - actual), 1),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--out", type=Path, help="per-match CSV (default: src/model_artifacts/evaluation_<season>.csv)")
    args = parser.parse_args()

    df = load_data(args.data)
    train = df[df["season"] < args.test_season]
    test = df[df["season"] == args.test_season].copy()
    if test.empty:
        raise SystemExit(f"No rows for season {args.test_season}.")
    print(f"Train: {len(train)} rows ({train['season'].min()}-{train['season'].max()})")
    print(f"Test:  {len(test)} rows, {test['match_id'].nunique()} matches ({args.test_season})\n")

    player_rows = []
    for name, predict in MODELS.items():
        preds = predict(train, test)
        test[f"pred_{name}"] = preds
        player_rows.append({"Model": name, **player_metrics(test, preds)})

    print(f"\nPlayer-level ({len(test)} rows)")
    print(pd.DataFrame(player_rows).set_index("Model").round(3).to_string())

    matches = team_results(test, MODELS)
    summary = matches.groupby("Model", sort=False).agg(**{
        "Actual points of XI": ("Actual Points", "mean"),
        "% of Dream Team": ("Pct of Dream Team", "mean"),
        "Overlap with Dream XI": ("Overlap with Dream XI", "mean"),
        "Team total MAE": ("MAE", "mean"),
    })
    print(f"\nTeam-level ({matches['Match ID'].nunique()} matches; "
          f"Dream Team averages {matches['Dream Team Points'].mean():.0f} points)")
    print(summary.round(1).to_string())

    out = args.out or OUTPUT_DIR / f"evaluation_{args.test_season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out, index=False)
    print(f"\nPer-match results -> {out}")


if __name__ == "__main__":
    main()
