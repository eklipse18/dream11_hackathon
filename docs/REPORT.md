# Next-Gen Team Builder with Predictive AI — Project Report

**Task:** Recommend an 11-player IPL fantasy team for an upcoming match by predicting each player's fantasy points, with explanations for every pick.
**Training data:** Cricsheet ball-by-ball data, Domestic → Men's → IPL, seasons 2022–2025 only. No 2026 data is used anywhere.
**Report date:** 21 September 2026

---

## 1. Summary

The solution has two stages:

1. **Player points model.** A gradient-boosted tree regressor (XGBoost) predicts each player's fantasy points for a match. It uses rolling form, career averages, venue and opponent history, a stats-derived player role, and match-day weather.
2. **Team optimiser.** An integer linear program (ILP) chooses the 11 players with the highest total predicted points, subject to the hackathon rules on team composition (1–8 per role, at least one player from each side).

Each recommended player comes with a SHAP breakdown of the features that raised or lowered their prediction.

**Headline results.** Evaluated on the full 2025 season (74 matches, 1,712 player-match rows), with the model trained on 2022–2024:

| | Player-level MAE | XI points as % of best possible XI |
|---|---|---|
| Predict the training mean for everyone | 24.66 | 62.6 % |
| Career average | 24.59 | 66.1 % |
| XGBoost (stats-derived roles) | 24.27 | 66.2 % |
| **XGBoost (all features)** | **24.28** | **67.1 %** |

The second XGBoost model includes extra strike-rate and economy form features as well.

Per-player fantasy points in T20 are very noisy, and the model is only slightly better than simple baselines. Section 8 discusses why and what is most likely to help.

---

## 2. Problem and constraints

| Requirement (from brief) | How it is handled |
|---|---|
| Input: two team names (Cricsheet spelling) and a match date | Candidate pool = both squads. Features are taken as each player's latest values strictly before the match date. |
| 11 players, 1–8 each of batter / bowler / all-rounder / wicketkeeper, at least 1 from each team | Hard constraints in the ILP (Section 6.2) |
| Train only on IPL 2022–2025, never on 2026 | Every feature is computed from 2022–2025 deliveries. Evaluation holds out 2025 in the same way 2026 will be held out. |
| Toss not known, no playing XI or batting order | No feature uses toss, XI or batting position. Only pre-match history and context are used. |
| Features must be available for future matches | All features are lagged (prior matches only). See Section 4.3 for the one exception (weather). |
| Recommendation in under 10 s | ILP solve takes 0.05 s on average and 0.08 s at worst; model inference takes milliseconds |

---

## 3. Data

### 3.1 Source and scope

- **Cricsheet IPL ball-by-ball**, 2022–2025: 291 matches, 18 venues, 10 franchises.
- Deliveries are aggregated into one row per player per match. After cleaning this gives **6,713 player-match rows** and 358 players (about 1,620–1,720 rows per season).

### 3.2 Per-match aggregation

For each player in each match, the delivery data is rolled up into:

- **Batting:** runs, balls faced (wides excluded), fours, sixes, whether dismissed, how dismissed, strike rate
- **Bowling:** legal balls, runs conceded (bat runs, wides and no-balls), wickets (run-outs, retired and obstructing dismissals excluded), overs, economy, maidens
- **Fielding:** catches, stumpings, run-outs (credited from the `fielder` column)

### 3.3 Cleaning

The raw `fielder` column contains substitute fielders (`(sub)Anukul Roy`) and combined entries (`Tilak Varma/Ishan Kishan`). These create fake "players" with a few catches each. The cleaned dataset drops them, which takes the table from 6,813 to 6,713 rows.

**Remaining known issue.** 167 rows belong to players who only fielded in that match (took a catch without batting or bowling), and these rows have no `team`. For evaluation we fill in the team from that player's most common team in the same season. The 29 rows that still can't be resolved are dropped.

### 3.4 External data (fetched programmatically)

| Source | What | Coverage |
|---|---|---|
| Open-Meteo historical archive API | Daily max/min temperature, precipitation and weather code at each venue's coordinates on match day | 100 % of matches |
| Wikipedia infobox scrape | Playing role, batting style, bowling style, nationality | About 35 % of players. Surname-only Cricsheet names (e.g. "Kohli") often don't match a Wikipedia page. |

Because the Wikipedia roles cover only about a third of players and use around 30 different spellings (`Battingall-rounder`, `Wicket-keeperbatsman`, `Bowler[2][3]`, …), we don't use them directly as the role feature (see Section 4.2).

---

## 4. Features

### 4.1 Target: fantasy points

The target is the fantasy points a player scores in a match. We reconstructed the scoring rules below from the data, and they reproduce the target exactly for **all 6,713 rows**:

| Event | Points |
|---|---|
| Run | +1 |
| Boundary bonus (four / six) | +1 / +2 |
| 30 / 50 / 100 runs (highest milestone only) | +4 / +8 / +16 |
| Dismissed for a duck | −2 |
| Wicket (excluding run-out) | +25 |
| 3 or 4 wickets / 5+ wickets bonus | +4 / +8 |
| Maiden over | +12 |
| Catch | +8 |
| Stumping / run-out | +12 |

Across the dataset the target averages 34.6 points, with a standard deviation of 31.4, a median of 25 and a maximum of 222. It is heavily right-skewed: a few big innings or wicket hauls dominate.

### 4.2 Feature set

Every history feature is computed **only from matches before the current one** (lagged by one match). We checked this independently: recomputing each feature with a lagged rolling window matches the stored values for more than 99.8 % of rows. The small percentage comes from the days with two matches.

| Group | Features | Notes |
|---|---|---|
| Recent form (last 5 matches) | `runs_form_5`, `wickets_form_5`, `fantasy_points_form_5`, `strike_rate_form_5`, `economy_form_5` | Missing for a player's first IPL match (5.5 % of rows) |
| Career | `runs_career_avg`, `wickets_career_avg`, `fantasy_points_career_avg`, `matches_played` | Expanding mean over all earlier matches |
| Venue history | `fantasy_points_venue_avg` | Player's average at this venue. Missing for 40 % of rows (first visit). |
| Opponent history | `fantasy_points_vs_opponent_avg` | Player's average against this opponent. Missing for 37 % of rows. |
| Match context | `venue`, `team`, `opponent` | Categorical, handled natively by XGBoost |
| Weather | `temp_max`, `temp_min`, `precipitation`, `weather_code` | Open-Meteo, match day at venue |
| **Player role (stats-derived)** | `role` belongs to {batter, bowler, allrounder, wicketkeeper} | See below. Also stored as four one-hot columns `is_batter`, `is_bowler`, `is_allrounder`, `is_wicketkeeper`. |

**Stats-derived player roles.** The role is needed twice: as a model feature and for the team-composition constraints. We derive it from how each player is actually used in the IPL, not from scraped text:

- **Wicketkeeper:** at least one stumping, or the scraped role mentions keeper
- **All-rounder:** averages ≥ 0.8 overs bowled **and** ≥ 6 balls faced per match
- **Bowler:** meets the overs threshold only
- **Batter:** everyone else

We chose the thresholds by comparing against the players whose Wikipedia role is known. This gives every player exactly one role: 118 batters, 169 bowlers, 34 all-rounders and 37 wicketkeepers. Mapping the scraped text instead leaves 186 of 358 players with no role (`UNKNOWN`).

The main disagreement with Wikipedia is players it calls "batting all-rounders" who barely bowl in the IPL. Tilak Varma and Shivam Dube, for example, bowl about 0.1 overs per match, and we classify them as batters. For fantasy scoring, how a player is used in the IPL is what counts.

### 4.3 Feature availability for future matches

Every history feature can be computed on the day before a match from past data alone. **Weather is the exception.** In training it is the *observed* weather on match day. For a genuinely upcoming match it would come from a forecast. This is a small train/serve mismatch, and in practice weather contributes very little (Section 7).

---

## 5. Model architecture

```mermaid
flowchart LR
    A[Cricsheet deliveries<br/>IPL 2022-2025] --> B[Per player-match<br/>aggregation]
    B --> C[Fantasy points<br/>target]
    B --> D[Lagged form / career /<br/>venue / opponent features]
    E[Open-Meteo weather] --> F[Feature table]
    G[Stats-derived roles] --> F
    D --> F
    C --> F
    F --> H[XGBoost regressor<br/>predicted points per player]
    H --> I[ILP team optimiser<br/>PuLP + CBC]
    I --> J[Recommended XI<br/>+ SHAP explanations]
```

### 5.1 Points model: XGBoost regressor

| Setting | Value |
|---|---|
| Objective | Squared error (reported metric: MAE) |
| Trees | Up to 2,000, with early stopping (50 rounds) on the validation season. **34–42 trees selected.** |
| Depth / learning rate | 4 / 0.03 |
| Row / column subsampling | 0.8 / 0.8 |
| `min_child_weight`, `reg_lambda` | 5, 1.0 |
| Categoricals | Native (`enable_categorical=True`, `tree_method="hist"`) |
| Missing values | Native. Tree splits learn a default direction for missing values, e.g. a player's first appearance at a venue. |
| Seed | 42 (deterministic) |

**Why gradient-boosted trees.** The feature table is small (about 5,000 training rows), mixes numeric and categorical columns, and has many structurally missing values. Trees handle all of these natively, they are fast, and exact SHAP values are cheap to compute for them.

### 5.2 Team optimiser: integer linear program

For a candidate pool of players $P$ with predicted points $\hat{y}_i$, team $t_i$ and role $r_i$, choose binary variables $x_i$ to

$$\max (\sum_{i \in P} \hat{y}_i x_i) \quad \text{s.t.} \quad \sum_i x_i = 11,\quad \sum_{i: t_i = T} x_i \ge 1 \;\; \forall T,\quad 1 \le \sum_{i: r_i = R} x_i \le 8 \;\; \forall R.$$

The problem is solved with PuLP and the CBC solver. With about 30 candidates it solves in about 50 ms, far within the 10-second limit. Because the solution is an exact optimum of the objective, all the quality of the recommendation comes from the point predictions.

---

## 6. Evaluation protocol

- **Split by season, never shuffled:** train on 2022–2023, validate on 2024 (early stopping only), then **refit on 2022–2024** with the selected number of trees and **test on 2025**. This is the same setup as the official evaluation, which trains on ≤ 2025 and tests on 2026.
- **Player-level metrics:** MAE, RMSE, R², and the Spearman rank correlation between predicted and actual points *within each match*. Ranking within a match is what drives team selection.
- **Team-level metrics:** for each 2025 match, the ILP picks an XI from the players who appeared in that match, using each model's predictions. The **Dream Team** is the ILP solution using *actual* points (the best possible XI under the same constraints). We report:
  - the actual points scored by the predicted XI
  - those points as a percentage of the Dream Team's points
  - how many players the predicted XI shares with the Dream Team
  - the MAE between the predicted and actual XI totals

**Caveat.** The evaluation pool contains only players who actually took part (batted, bowled or fielded). At real prediction time the pool is the full squad of about 15 per side, so real-world performance will be somewhat lower.

**Baselines:**
- **Mean:** the same prediction for everyone, so the ILP's choice is effectively arbitrary
- **Career average:** the player's career average
- **LP heuristic:** the average of form, career, venue and opponent averages, as in the team's first prototype `LPBaselineModel.py`

---

## 7. Results

### 7.1 Player-level (2025 test season, 1,712 rows)

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Within-match Spearman ↑ |
|---|---|---|---|---|
| Mean baseline | 24.66 | 30.99 | 0.000 | — |
| Career average | 24.59 | 31.12 | −0.008 | **0.150** |
| LP heuristic (mean of 4 averages) | 25.26 | 32.36 | −0.090 | 0.129 |
| XGBoost, scraped roles | 24.31 | 30.56 | 0.027 | 0.112 |
| **XGBoost, stats-derived roles** | **24.27** | **30.52** | **0.030** | 0.118 |
| XGBoost, stats-derived roles, no weather | 24.31 | 30.57 | 0.027 | 0.129 |
| XGBoost, all features (+ SR/economy form) | 24.28 | 30.52 | 0.030 | 0.139 |

### 7.2 Team-level (74 matches in 2025; Dream Team averages 643 points)

| Model | Actual points of XI | % of Dream Team | Overlap with Dream XI | Team total MAE |
|---|---|---|---|---|
| Mean baseline (arbitrary XI) | 399.6 | 62.6 % | 5.6 / 11 | 73.9 |
| Career average | 423.7 | 66.1 % | 5.9 | 100.6 |
| LP heuristic | 420.7 | 65.7 % | 5.9 | 104.6 |
| XGBoost, scraped roles | 423.5 | 66.1 % | 5.9 | 88.9 |
| XGBoost, stats-derived roles | 423.5 | 66.2 % | 5.9 | 85.4 |
| XGBoost, no weather | 421.9 | 65.8 % | 5.9 | 86.3 |
| **XGBoost, all features** | **429.5** | **67.1 %** | **5.9** | 87.5 |

### 7.3 What the results say

1. **Every approach captures about two-thirds of the best possible score, and the models add only about 4 percentage points over an arbitrary XI.** The best model (all features) reaches 67.1 % against 62.6 %. Per-match results vary widely, from 37 % to 100 %. The stats-derived-roles model beats the arbitrary XI in only 51 % of matches.
2. **XGBoost improves slightly on MAE and on how well its predicted XI totals match reality (team total MAE 85 vs 101–105 for the averaging baselines), but not on ranking.** The simple career average has the best within-match rank correlation. The tree model is well calibrated overall but no better than an average at ordering players within a match.
3. **Stats-derived roles beat the scraped roles, but only slightly** (MAE 24.27 vs 24.31, team total MAE 85.4 vs 88.9). The larger benefit is operational: every player has a valid role, so the composition constraints always apply.
4. **Weather adds almost nothing.** Removing it changes MAE by 0.04. Its gain-based importance looks high only because trees split on noisy continuous variables. SHAP (below) shows its real contribution is small.
5. **Early stopping picks only 34–42 trees at learning rate 0.03.** The model barely moves away from the mean before validation error stops improving. That is a symptom of a low signal-to-noise target, not of a model that is too small.

---

## 8. Explainability

### 8.1 Global feature importance (SHAP, 2025 test set)

![Mean absolute SHAP value per feature](figures/shap_importance.png)

| Rank | Feature | Mean \|SHAP\| (points) |
|---|---|---|
| 1 | `runs_career_avg` | 1.40 |
| 2 | `fantasy_points_career_avg` | 1.18 |
| 3 | `venue` | 0.90 |
| 4 | `opponent` | 0.52 |
| 5 | `runs_form_5` | 0.50 |
| 6 | `role` | 0.49 |
| 7 | `matches_played` | 0.44 |
| 8 | `fantasy_points_form_5` | 0.40 |
| … | weather (`weather_code`, `temp_max`, `temp_min`, `precipitation`) | 0.16, 0.05, 0.05, 0.02 |

In plain terms: the model mostly asks **"how many runs and fantasy points does this player usually get?"** It then adjusts for the venue, the opponent, recent form and the player's role. All-rounders get a boost of about +2 to +3 points because they can score in two ways.

### 8.2 Per-player justification (Product UI output)

For every recommended player, the UI shows the predicted points and the three features that moved the prediction most, relative to the average player (base value 34.7 points). Example: **2025 final, PBKS vs RCB, Narendra Modi Stadium, 3 June 2025.**

| Player | Team | Role | Predicted | Actual | Top drivers (SHAP, points) |
|---|---|---|---|---|---|
| Rajat Patidar | RCB | batter | 45.1 | 31 | runs_career_avg (+4.1), fantasy_points_career_avg (+2.7), opponent (+0.8) |
| Kohli | RCB | batter | 44.8 | 50 | runs_career_avg (+4.1), fantasy_points_career_avg (+1.8), runs_form_5 (+1.2) |
| Phil Salt | RCB | wicketkeeper | 42.9 | 28 | runs_career_avg (+3.1), fantasy_points_career_avg (+1.9), opponent (+0.9) |
| Shreyas Iyer | PBKS | batter | 41.5 | 17 | runs_career_avg (+2.5), fantasy_points_career_avg (+1.6), fantasy_points_venue_avg (+0.8) |
| Priyansh Arya | PBKS | batter | 40.4 | 36 | runs_career_avg (+3.0), fantasy_points_career_avg (+1.7), runs_form_5 (+0.6) |
| Stoinis | PBKS | allrounder | 40.0 | 8 | role (+3.2), fantasy_points_career_avg (+1.5), venue (−0.9) |
| Prabhsimran | PBKS | wicketkeeper | 39.2 | 30 | runs_career_avg (+2.7), fantasy_points_career_avg (+1.4), runs_form_5 (+0.5) |
| Livingstone | RCB | allrounder | 39.0 | 37 | role (+2.4), fantasy_points_career_avg (+2.0), opponent (+1.0) |
| Josh Inglis | PBKS | wicketkeeper | 35.7 | 52 | fantasy_points_career_avg (+1.5), runs_career_avg (−1.2), fantasy_points_form_5 (+0.7) |
| Hazlewood | RCB | bowler | 35.5 | 25 | runs_career_avg (−1.3), fantasy_points_career_avg (+1.0), wickets_career_avg (−0.8) |
| Chahal | PBKS | bowler | 35.4 | 25 | runs_career_avg (−1.2), fantasy_points_career_avg (+1.0), wickets_form_5 (+0.4) |
| **Total** | | | **439.4** | **339** | Dream Team for this match: 629 |

This match shows the main weakness. The model spreads its predictions across a narrow band (35–45 points), so it can't anticipate the big individual performances that make up most of the Dream Team's points.

---

## 9. Interfaces

| Interface | Brief requirement | Current status |
|---|---|---|
| **Product UI** (Gradio, `src/frontend/main.py`) | Enter two teams and a date, get the recommended XI with a justification for each player, in under 10 s | Gradio page exists, but `inference.py` is still a placeholder. The pieces needed (model, ILP, SHAP) are all done. The model scores in milliseconds and the ILP solves in 0.05–0.08 s, so the 10-second limit is easily met. |
| **Model UI** | Choose train/test periods, retrain, save models to `src/model_artifacts/`, save processed data to `src/data/processed/`, and export a CSV with Match Date, Team 1, Team 2, Predicted XI, Dream Team XI, Predicted Points, Actual Points, MAE | **Not built yet.** The evaluation in Section 6 produces exactly these columns per match and is the logic this UI needs. |

---

## 10. Limitations and next steps

The following are ordered by expected impact on the 70 % "model quality" criterion:

1. **Model the target's distribution, not just its mean.** Points are zero-inflated and right-skewed. Promising options:
   - a Tweedie or Poisson objective
   - predicting batting, bowling and fielding points separately and adding them
   - optimising the XI for upside (for example a high quantile) rather than the expected value
2. **Expected involvement features.** The biggest source of variance is whether and how much a player bats or bowls. Useful pre-toss proxies:
   - typical batting position in recent matches
   - share of the team's overs bowled recently
   - how often the player was in the XI in recent matches
3. **Player identity resolution.** Cricsheet surnames produce duplicates (`Phil Salt` / `Philip Salt`, `Saha` / `W Saha`) and block external joins. A single canonical player ID would fix both.
4. **Pitch and venue features.** Use venue-level run rates and pace-vs-spin wicket shares, computed from past deliveries. The brief states pitch type is known before the match.
5. **Use weather forecasts at prediction time** so the features match between training and serving (Section 4.3).
6. **Evaluate with full squads (about 15 per side)** rather than only players who took part, to match the real use case.

---

## 11. Reproducibility and submission checklist

| Item | Status |
|---|---|
| Deterministic training (fixed seed, fixed season split) | Done |
| Training restricted to IPL 2022–2025 and no 2026 data | Done |
| External data fetched programmatically (`src/scripts/external_data_integration.ipynb`) | Done |
| Model training script (`src/scripts/train_xgboost_model.py`) | Needs updating: it lists features that aren't in the data yet (`humidity`, `wind_speed`, `avg_fours`, `avg_sixes`, `pitch_type`) |
| Evaluation script that reproduces Section 7 | To be added to the repo |
| `requirements.txt` | To be added. Dependencies are currently in `pyproject.toml` and `uv.lock`, and `xgboost`, `scikit-learn` and `shap` are missing from them. |
| `src/model_artifacts/`, `src/data/processed/` | To be created by the Model UI |
| Product UI and Model UI | See Section 9 |
| Video demo | To do |
