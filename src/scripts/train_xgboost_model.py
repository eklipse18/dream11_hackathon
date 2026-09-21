
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "player_match_features_with_external_data.csv"
OUTPUT_DIR = Path("outputs")

NUMERIC_FEATURES = [
    "runs_career_avg", "wickets_career_avg", "fantasy_points_career_avg",
    "fantasy_points_venue_avg", "fantasy_points_vs_opponent_avg",
    "temp_max", "temp_min", "precipitation"
]
CATEGORICAL_FEATURES = ["venue", "team", "opponent", "role_group", "weather_code"]
TARGET = "fantasy_points"

VAL_FRAC = 0.15     
TEST_FRAC = 0.20    # most recent matches


# Role

def normalize_role(raw_role) -> str:

    if pd.isna(raw_role):
        return "UNKNOWN"
    r = str(raw_role).lower()

    is_bowl_ar = "bowling" in r and ("all" in r or "rounder" in r)
    is_bat_ar = "batting" in r and ("all" in r or "rounder" in r)
    is_ar = ("all-round" in r or "allrounder" in r or "all round" in r
             or is_bowl_ar or is_bat_ar)
    is_wk = "keeper" in r
    is_bowler = "bowl" in r and not is_ar
    is_bat = any(k in r for k in
                 ["bat", "top-order", "top order", "middle-order",
                  "middle order", "opening"])

    if is_ar:
        return "AR"
    if is_wk:
        return "WK"
    if is_bowler:
        return "BOWL"
    if is_bat:
        return "BAT"
    return "UNKNOWN"


# Data

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["role_group"] = df["player_role"].apply(normalize_role)

    n_before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with no {TARGET} value.")

    return df


def prepare_features(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    return X, y


def time_based_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):

    order = df["date"].sort_values(kind="mergesort").index
    n = len(order)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)

    test_idx = order[n - n_test:]
    val_idx = order[n - n_test - n_val: n - n_test]
    train_idx = order[: n - n_test - n_val]

    print(f"Train: {len(train_idx)} rows, up to {df.loc[train_idx, 'date'].max().date()}")
    print(f"Val:   {len(val_idx)} rows, up to {df.loc[val_idx, 'date'].max().date()}")
    print(f"Test:  {len(test_idx)} rows, from {df.loc[test_idx, 'date'].min().date()} onward")

    return (
        X.loc[train_idx], X.loc[val_idx], X.loc[test_idx],
        y.loc[train_idx], y.loc[val_idx], y.loc[test_idx],
    )



# Model

def train_model(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=4,            
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,     
        reg_lambda=1.0,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=30,
        eval_metric="mae",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration} (of {model.n_estimators} max)")
    return model


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_test.mean()))

    print(f"\nTest set performance ({len(y_test)} matches):")
    print(f"  MAE:           {mae:.2f} fantasy points")
    print(f"  RMSE:          {rmse:.2f} fantasy points")
    print(f"  R2:            {r2:.3f}")
    print(f"  Baseline MAE:  {baseline_mae:.2f}  (predicting the mean for every row)")

    return {"mae": mae, "rmse": rmse, "r2": r2, "baseline_mae": baseline_mae}


def show_feature_importance(model, X_train) -> pd.Series:
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    print("\nFeature importance (gain-based):")
    print(importances.to_string())
    return importances


def explain_with_shap(model, X_test, n_examples: int = 3):

    try:
        import shap
    except ImportError:
        print("\n(shap not installed -- skipping explanations)")
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    print(f"\nSHAP explanation for {n_examples} sample predictions:")
    for i in range(min(n_examples, len(X_test))):
        row = X_test.iloc[i]
        pred = model.predict(X_test.iloc[[i]])[0]
        print(f"\n  Row {i} -- predicted: {pred:.1f} points")
        contribs = pd.Series(shap_values.values[i], index=X_test.columns).sort_values(
            key=abs, ascending=False
        )
        for feat, val in contribs.head(4).items():
            sign = "+" if val >= 0 else ""
            print(f"    {feat}: {sign}{val:.2f}  (value={row[feat]})")

    return shap_values


def save_model_and_metadata(model, X_train, metrics: dict):

    model_path = OUTPUT_DIR / "xgboost_fantasy_model.json"
    meta_path = OUTPUT_DIR / "xgboost_fantasy_model_metadata.json"

    model.save_model(model_path)

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": list(X_train.columns),
        "categories": {
            col: list(map(str, X_train[col].cat.categories))
            for col in CATEGORICAL_FEATURES
        },
        "target": TARGET,
        "best_iteration": int(model.best_iteration),
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nModel saved    -> {model_path}")
    print(f"Metadata saved -> {meta_path}")


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(df, X, y)

    model = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate(model, X_test, y_test)
    show_feature_importance(model, X_train)
    explain_with_shap(model, X_test)

    save_model_and_metadata(model, X_train, metrics)


if __name__ == "__main__":
    main()
