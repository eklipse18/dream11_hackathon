"""End-to-end interface: match -> TabFM context -> predictions -> optimized XI.

Usage (from the repo root):
    python src/scripts/run.py
"""

import pandas as pd

import tabfm_model
from optimize_team import select_final_11


def predict_match(team_a: str, team_b: str, match_date, strategy: str = "hybrid", **kwargs) -> pd.DataFrame:
    predictions = tabfm_model.predict_match(match_date, team_a, team_b, strategy=strategy, **kwargs)
    xi = select_final_11(predictions)

    print(f"\n{team_a} vs {team_b} on {match_date} (strategy={strategy})")
    print(predictions.sort_values("predicted_fantasy_points", ascending=False).to_string(index=False))
    print(f"\nSelected XI (predicted total {xi['baseline_score'].sum():.1f}):")
    print(xi.to_string(index=False))
    return xi


def run_matches(matches: list[dict], **kwargs) -> dict[tuple, pd.DataFrame]:
    return {
        (m["team_a"], m["team_b"], m["match_date"]): predict_match(**{**m, **kwargs})
        for m in matches
    }


if __name__ == "__main__":
    # TabFMSolution.md's placeholder teams/date aren't in the dataset; using real ones instead.
    predict_match(team_a="CSK", team_b="PBKS", match_date="2025-04-08")
