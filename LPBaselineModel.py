
import pandas as pd
import pulp

FEATURES_PATH = "aggregate_player_match_features.csv"

#Roles: Batsman, Bowler, All-Rounder, Wicket-Keeper
ROLES_PATH = "roles.csv"


def get_latest_stats(df: pd.DataFrame, team: str, date: str) -> pd.DataFrame:
    #return each player's most recent known rolling stats strictly before `date`
     
    cutoff = pd.to_datetime(date)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    sub = df[(df["team"] == team) & (df["date"] < cutoff)]
    latest = sub.sort_values("date").groupby("player").tail(1)
    return latest


def compute_baseline_score(row: pd.Series) -> float:
    #average of whichever historical averages are available for this player. NaNs are skipped.

    candidates = [
        row.get("fantasy_points_form_5"),
        row.get("fantasy_points_career_avg"),
        row.get("fantasy_points_venue_avg"),
        row.get("fantasy_points_vs_opponent_avg"),
    ]
    valid = [v for v in candidates if pd.notna(v)]
    return sum(valid) / len(valid) if valid else 0.0


def build_candidate_pool(df: pd.DataFrame, team_a: str, team_b: str, match_date: str) -> pd.DataFrame:
    pool = pd.concat([
        get_latest_stats(df, team_a, match_date),
        get_latest_stats(df, team_b, match_date),
    ])
    pool["baseline_score"] = pool.apply(compute_baseline_score, axis=1)
    return pool[["player", "team", "baseline_score"]].reset_index(drop=True)


def select_best_11(pool: pd.DataFrame, roles: pd.DataFrame | None = None) -> pd.DataFrame:

    prob = pulp.LpProblem("Fantasy_XI_Selection", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}

    #maximize total predicted (baseline) score
    prob += pulp.lpSum(x[i] * pool.loc[i, "baseline_score"] for i in pool.index)

    #exactly 11 players
    prob += pulp.lpSum(x[i] for i in pool.index) == 11

    #at least 1 player from each team
    for team in pool["team"].unique():
        team_idx = pool[pool["team"] == team].index
        prob += pulp.lpSum(x[i] for i in team_idx) >= 1

    #role caps 
    if roles is not None:
        pool = pool.merge(roles, on="player", how="left")
        pool["role"] = pool["role"].fillna("Unknown")
        for role in pool["role"].unique():
            role_idx = pool[pool["role"] == role].index
            if role == "Unknown":
                continue
            prob += pulp.lpSum(x[i] for i in role_idx) >= 1
            prob += pulp.lpSum(x[i] for i in role_idx) <= 8

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_idx = [i for i in pool.index if x[i].value() == 1]
    result = pool.loc[selected_idx].sort_values("baseline_score", ascending=False)
    return result


def recommend_team(df: pd.DataFrame, team_a: str, team_b: str, match_date: str,
                    roles_path: str | None = ROLES_PATH) -> pd.DataFrame:
    pool = build_candidate_pool(df, team_a, team_b, match_date)
    roles = pd.read_csv(roles_path) if roles_path else None
    team = select_best_11(pool, roles)
    return team


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)

    TEAM_A, TEAM_B, MATCH_DATE = "CSK", "PBKS", "2025-04-08"

    print(f"Building baseline recommendation: {TEAM_A} vs {TEAM_B} on {MATCH_DATE}\n")
    best_11 = recommend_team(df, TEAM_A, TEAM_B, MATCH_DATE)

    print("Recommended XI (baseline max-average model):")
    print(best_11.to_string(index=False))
    print(f"\nTotal predicted score: {best_11['baseline_score'].sum():.1f}")
