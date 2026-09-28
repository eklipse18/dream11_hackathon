"""TabFM-based fantasy-points predictor.

One TabFM model, no per-player training: historical player-match rows are
fed as in-context "training" examples, the upcoming players are unlabeled
query rows. See TabFMSolution.md for the design.

TabFM itself is only ever touched inside `_checkpoint`/`_run_tabfm` so this
module can later be swapped for an XGBoost implementation without changing
optimize_team.py or run.py.
"""

import numpy as np
import pandas as pd

DATA_PATH = "src/data/all_years_player_match_agg_with_external_data.csv"
# batting_team/bowling_team in DATA_PATH are match-level (identical for every player in a
# match), not per-player -- pull the real team/opponent + playing style from this sibling
# feature file instead, keyed on (player, match_id).
STYLE_PATH = "src/data/player_match_features_with_external_data.csv"

NUMERIC_FEATURES = [
    "avg_score", "avg_balls_faced", "avg_outs", "avg_runs_conceded",
    "avg_balls_bowled", "avg_wickets_taken", "avg_fantasy_points",
    "avg_venue_fantasy_points", "avg_against_opposition_fantasy_points",
    "avg_score_5", "avg_balls_faced_5", "avg_outs_5", "avg_runs_conceded_5",
    "avg_balls_bowled_5", "avg_wickets_taken_5", "avg_fantasy_points_5",
    "matches_played", "avg_fours", "avg_sixes", "avg_catches", "avg_runouts",
    "temp_max", "temp_min", "precipitation",
]
CATEGORICAL_FEATURES = ["venue", "team", "opponent", "role", "weather_code"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "fantasy_points"

STRATEGIES = ("player", "global", "player_similar", "hybrid")
HOLDOUT_YEAR = 2026  # hackathon reserves this year for final testing; never touch it here


# Data

def load_data(path: str = DATA_PATH, style_path: str = STYLE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    style = pd.read_csv(style_path, usecols=["player", "match_id", "team", "opponent", "batting_style", "bowling_style"])
    style = style.drop_duplicates(["player", "match_id"])
    df = df.drop(columns=["batting_team", "bowling_team"]).merge(style, on=["player", "match_id"], how="left")
    df = df.dropna(subset=["team", "opponent"]).reset_index(drop=True)  # join is 100% coverage today; guard future data changes

    df["season"] = df["date"].dt.year
    df = df[df["season"] < HOLDOUT_YEAR].reset_index(drop=True)

    # NaNs mean "no history yet" (e.g. a player's first match) -> 0, not a leaky global mean
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0.0)
    for col in CATEGORICAL_FEATURES + ["batting_style", "bowling_style"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df


def _player_role(df: pd.DataFrame, player: str, before: pd.Timestamp) -> str | None:
    rows = df[(df["player"] == player) & (df["date"] < before)]
    if rows.empty:
        rows = df[df["player"] == player]  # brand-new player: role only, no stats used
    if rows.empty:
        return None
    return rows.sort_values("date").iloc[-1]["role"]


def find_similar_players(df: pd.DataFrame, player: str, role: str, before: pd.Timestamp, k: int = 15) -> list[str]:
    """Same role (+ same playing style when known), ranked by z-scored NUMERIC_FEATURES distance."""
    history = df[df["date"] < before]
    latest = history.sort_values("date").groupby("player").tail(1)
    latest = latest[latest["role"] == role].reset_index(drop=True)
    is_player = latest["player"] == player
    if not is_player.any() or len(latest) < 2:
        return []

    style_col = "bowling_style" if role == "bowler" else "batting_style"
    target_style = latest.loc[is_player, style_col].iloc[0]
    if target_style != "Unknown":
        styled = latest[latest[style_col] == target_style]
        if len(styled) >= 2:
            latest = styled.reset_index(drop=True)
            is_player = latest["player"] == player

    stats = latest[NUMERIC_FEATURES]
    z = (stats - stats.mean()) / stats.std().replace(0, 1)
    target = z[is_player.to_numpy()].iloc[0]
    dist = np.sqrt(((z - target) ** 2).sum(axis=1))

    ranked = latest.assign(_dist=dist)[~is_player].sort_values(["_dist", "player"])
    return ranked["player"].head(k).tolist()


def build_context(
    match_date, players, strategy, context_size,
    player_ratio: float = 0.4, similar_ratio: float = 0.3, global_ratio: float = 0.3,
    seed: int = 42, df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Labeled context rows (date < match_date) for the given strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}, expected one of {STRATEGIES}")
    if df is None:
        df = load_data()

    before = pd.to_datetime(match_date)
    history = df[df["date"] < before]
    players = list(players)
    rng = np.random.default_rng(seed)

    def sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(pool) <= n:
            return pool
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        return pool.loc[idx]

    def player_pool() -> pd.DataFrame:
        return history[history["player"].isin(players)]

    def similar_pool() -> pd.DataFrame:
        similar: set[str] = set()
        for p in players:
            role = _player_role(df, p, before)
            if role:
                similar.update(find_similar_players(df, p, role, before))
        return history[history["player"].isin(similar)]

    if strategy == "player":
        result = sample(player_pool(), context_size)
    elif strategy == "global":
        result = sample(history, context_size)
    elif strategy == "player_similar":
        pool = pd.concat([player_pool(), similar_pool()]).drop_duplicates()
        result = sample(pool, context_size)
    else:  # hybrid
        n_player = int(context_size * player_ratio)
        n_similar = int(context_size * similar_ratio)
        n_global = context_size - n_player - n_similar
        result = pd.concat([
            sample(player_pool(), n_player),
            sample(similar_pool(), n_similar),
            sample(history, n_global),
        ]).drop_duplicates()

    return result if not result.empty else sample(history, context_size)  # never hand TabFM an empty context


def build_query_rows(match_date, team_a: str, team_b: str, venue: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One unlabeled row per candidate: every player who's played for either team, at their latest known stats."""
    if df is None:
        df = load_data()
    before = pd.to_datetime(match_date)
    history = df[df["date"] < before]

    def latest_for_team(team: str) -> pd.DataFrame:
        return history[history["team"] == team].sort_values("date").groupby("player").tail(1)

    pool = pd.concat([latest_for_team(team_a), latest_for_team(team_b)])
    pool = pool.sort_values("date").drop_duplicates("player", keep="last").copy()  # a player who's switched teams is only a candidate for their most recent one
    pool["date"] = before
    pool.loc[pool["team"] == team_a, "opponent"] = team_b
    pool.loc[pool["team"] == team_b, "opponent"] = team_a
    if venue is not None:
        pool["venue"] = venue
    pool[TARGET] = np.nan
    return pool.reset_index(drop=True)


# TabFM — the only functions that touch the model

_CHECKPOINT = None


def _checkpoint():
    global _CHECKPOINT
    if _CHECKPOINT is None:
        from tabfm import tabfm_v1_0_0_pytorch
        _CHECKPOINT = tabfm_v1_0_0_pytorch.load(model_type="regression")
    return _CHECKPOINT


def _run_tabfm(context: pd.DataFrame, query_row: pd.Series) -> float:
    from tabfm import TabFMRegressor
    reg = TabFMRegressor(model=_checkpoint(), max_num_rows=len(context))
    reg.fit(context[FEATURES], context[TARGET])
    return float(reg.predict(query_row[FEATURES].to_frame().T)[0])


def predict_match(
    match_date, team_a: str, team_b: str, strategy: str = "hybrid", context_size: int = 100,
    venue: str | None = None, seed: int = 42,
    player_ratio: float = 0.4, similar_ratio: float = 0.3, global_ratio: float = 0.3,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df is None:
        df = load_data()
    query = build_query_rows(match_date, team_a, team_b, venue=venue, df=df)

    predictions = []
    for _, row in query.iterrows():
        context = build_context(
            match_date, [row["player"]], strategy, context_size,
            player_ratio=player_ratio, similar_ratio=similar_ratio, global_ratio=global_ratio,
            seed=seed, df=df,
        )
        predictions.append({
            "player": row["player"],
            "team": row["team"],
            "role": row["role"],
            "predicted_fantasy_points": _run_tabfm(context, row),
        })
    return pd.DataFrame(predictions)


def demo():
    df = load_data()
    teams = sorted(df["team"].unique())
    assert len(teams) >= 2
    last_date = df["date"].max()
    context = build_context(last_date, [df["player"].iloc[0]], "global", context_size=50, df=df)
    assert not context.empty and (context["date"] < last_date).all()
    query = build_query_rows(last_date, teams[0], teams[1], df=df)
    assert query[TARGET].isna().all()
    assert (query["team"].isin([teams[0], teams[1]])).all()
    print(f"OK: {len(teams)} teams, context {len(context)} rows, {len(query)} candidate players")


if __name__ == "__main__":
    demo()
