"""Select the final 11 from tabfm_model.predict_match's output.

Just a wrapper around LPBaselineModel's existing ILP -- no new solver logic.
"""

import pandas as pd

from LPBaselineModel import select_best_11


def select_final_11(predictions: pd.DataFrame) -> pd.DataFrame:
    pool = predictions[["player", "team"]].assign(baseline_score=predictions["predicted_fantasy_points"])
    roles = predictions[["player", "role"]].drop_duplicates("player")
    return select_best_11(pool.reset_index(drop=True), roles)


def demo():
    predictions = pd.DataFrame({
        "player": [f"p{i}" for i in range(16)],
        "team": ["A"] * 8 + ["B"] * 8,
        "role": (["batter", "bowler", "all-rounder", "wicketkeeper"] * 4),
        "predicted_fantasy_points": [50, 40, 30, 20, 45, 35, 25, 15, 48, 38, 28, 18, 44, 34, 24, 14],
    })
    xi = select_final_11(predictions)
    assert len(xi) == 11
    assert set(xi["team"]) == {"A", "B"}
    print(f"OK: selected {len(xi)} players, total score {xi['baseline_score'].sum():.0f}")


if __name__ == "__main__":
    demo()
