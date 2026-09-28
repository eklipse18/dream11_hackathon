TabFM Solution Pipeline: 
"""
takes care of feature engineering, so we don't really have to think too much about manually engineering complex features from the raw data
ex: if we want to capture a player's performance against a specific opponent or another specific player, we don't necessarily have to
    manually design and formulate a feature to represent that interaction. TabFM can potentially learn such relationships from the contextual data itself.
"""

1. Predict each candidate player's fantasy points for the upcoming match.
2. Rank players.
3. Solve the constrained 11-player selection problem.

We do not train a separate model for every player, rather train one TabFM model across all player-match examples.

For a given query (say, team A vs. team B on date x/y/z), the data we feed our TabFM model is all the rows till before/on x/y/z 
date [abiding the rules] with label as context + ~30 rows corresponding to each player with no labels to represent the upcoming matches. 
For a given query, the output is the predicted fantasy points for each player.

Fantasy points are sorted and then solved by a linear optimizer.

Now, we have to decide what all historical data we will be giving to the model. We can choose to:
- give the full historical data of all teams that happened before it
- give the historical data of only the matches played by these teams (bad for new players, have very less data on them)
- give the historical data of only the matches played in that venue
...
- or a combination of these.

For new players, choosing the 2nd option is a poor choice, since they barely have matches played. So, we could find players with similar 
play styles/statistics as the new player.

We want our model to learn the fantasy score of this type of player under these conditions, so in that view, giving the full historical 
data makes the most sense. But the latency in inference is a factor to consider.

We propose a context structure to be in 3 parts:
- player-specific history [ex: Kohli vs CSK, Kohli vs GT, Kohli vs MI],
- similar-player history,
- global ipl history.

To do this, for every upcoming player P, we can construct a candidate historical pool:
    C_p = C_player UNION C_similar UNION C_global
then sample/select from it using a split [ex: 0.4, 0.3, 0.3] [optimal split can be found through validation]


Comparing:
Experiment 1:
    - Only upcoming players' history

Experiment 2:
    - All IPL history

Experiment 3:
    - Upcoming players + similar players

Experiment 4:
    - Upcoming players + similar players + global context

...

then evaluate all four on our validation.

----------------------------------------------------------------------------------------------------------------------

So our architecture should be:

- Team A vs. Team B
- Candidate players (~30)
- Build query features
-     Historical database           Query rows
          2022-2025              RCB + MI players
-     Context Selection
-  exact   similar   global
   player  players     IPL   
-                       TabFM
- Predicted FP / player
- Team optimizer
- Best 11

The data will be coming from @all_years_player_match_agg_with_external_data.csv.

I require:
- `tabfm_model.py`
    - `build_context(match_date,players,strategy,context_size)` 
        - Construct the labeled context and unlabeled query rows for an upcoming match
        - The query consists of one row per candidate player from Team A and Team B
        - Support these context strategies: "player", "global", "player_similar", "hybrid"
            - player: historical matches of the upcoming players
            - global: sampled historical IPL player-match rows
            - player_similar: upcoming players + similar-player history
            - hybrid: player-specific + similar-player + global IPL history [configurable proportions: player_ratio=0.4, similar_ratio=0.3, global_ratio=0.3]
    -  `predict_match(match_date,team_a,team_b,strategy="hybrid",...)`
        - For every candidate player:
            - Construct the context.
            - Construct the query rows.
            - Run TabFM inference.
            - Return:
                - player
                - team
                - role
                - predicted_fantasy_points
- `optimize_team.py` [Responsible only for selecting the final 11 players]
- `run.py` [Provide a simple end-to-end interface for testing a single match]
    - predict_match(
        team_a="Team A",
        team_b="Team B",
        match_date="2026-04-15"
      )
    - Pipeline:
        match
        ↓
        construct TabFM context
        ↓
        predict fantasy points for every candidate
        ↓
        optimize 11-player team
        ↓
        print predictions + selected team
    - Also provide a function to run predictions over a list of matches
- `ablations.ipynb`
    - Use a time-based split
    - 2026 must remain completely untouched until final evaluation because the hackathon explicitly reserves 2026 for testing
    - Run these experiments on the validation period:
        - Experiment 1 [Player-specific historical context only]
        - Experiment 2 [Global IPL historical context]
        - Experiment 3 [Player-specific + similar-player context]
        - Experiment 4 [Player-specific + similar-player + global context]
        - For each experiment evaluate:
            - player-level MAE
            - predicted vs actual fantasy points
            - ranking quality / top-player overlap
            - selected-team predicted points vs actual points
        - Also test a few hybrid context ratios
        - Only after choosing the best strategy on validation should the final model be evaluated on 2026

Important design principles
- One TabFM model, not one model per player.
- Each historical row represents a player in a match.
- The target is that player's actual fantasy points in that match.
- Query rows represent the players in the upcoming match and have no target.
- All query features must be information available before the upcoming match.
- Do not train/fine-tune TabFM on 2026.
- Keep context construction deterministic with fixed random seeds.
- Do not over-engineer the system. This is a prototype to determine whether TabFM works well enough to justify building the full solution.
- Keep the TabFM interface isolated so that tabfm_model.py can later be replaced by an XGBoost implementation without changing the rest of the - pipeline.