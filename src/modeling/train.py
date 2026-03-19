"""
Train an XGBoost regression model to predict fire response time (minutes).
Handles feature selection, train/test split, hyperparameter tuning via
cross-validation, and saving the trained model + feature list to disk.
"""
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Constants

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_response_time.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

TARGET_COL = "response_time_seconds"  # will be converted to minutes inside

EXCLUDE_COLS = [
    "INCIDENT_NUMBER",
    "TFS_ALARM_TIME",
    "TFS_ARRIVAL_TIME",
    "LAST_TFS_UNIT_CLEAR_TIME",
    "GEOMETRY",
    "INTERSECTION",
    "latitude",
    "longitude",
    "response_time_seconds",   # raw target
    "RESPONSE_TIME_MINUTES",   # duplicate target
    "HOUR",                    # replaced by hour_of_day from features.py
    "DAY_OF_WEEK",             # replaced by day_of_week
    "MONTH",                   # replaced by month
    "YEAR",                    # replaced by year
]

# helper functions

def _load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"  Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


def _build_target(df: pd.DataFrame) -> pd.Series:
    """Convert response_time_seconds → minutes and validate."""
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            "Run the pipeline first to build training_dataset.csv."
        )
    target = df[TARGET_COL] / 60.0
    target = target.clip(lower=0, upper=120) 
    return target


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns after dropping exclusions."""
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, bool]
    ]
    return feature_cols


def _make_xgb(n_estimators=500, learning_rate=0.05, max_depth=6,
              subsample=0.8, colsample_bytree=0.8, seed=42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",  
    )


# Grid search

def _tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Lightweight manual grid search over key XGBoost hyperparameters.
    Returns the best parameter dict.
    """
    print("\n  Running hyperparameter search (3-fold CV) …")

    param_grid = [
        {"n_estimators": 300,  "learning_rate": 0.05, "max_depth": 5},
        {"n_estimators": 500,  "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 500,  "learning_rate": 0.02, "max_depth": 6},
        {"n_estimators": 700,  "learning_rate": 0.03, "max_depth": 7},
    ]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_mae = np.inf
    best_params = param_grid[1]

    for params in param_grid:
        model = _make_xgb(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=kf,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mae = -scores.mean()
        print(f"    Params {params}  →  CV MAE = {mae:.3f} min")

        if mae < best_mae:
            best_mae = mae
            best_params = params

    print(f"\n  Best params: {best_params}  (CV MAE = {best_mae:.3f} min)")
    return best_params


# Main training function

def train_model(
    csv_path: str = "data/processed/training_dataset.csv",
    tune: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> XGBRegressor:


    os.makedirs(MODEL_DIR, exist_ok=True)

    # load
    print("\n=== Loading data ===")
    df = _load_dataset(csv_path)

    print("\n=== Building target & features ===")
    y = _build_target(df)
    feature_cols = _select_features(df)

    # Drop rows 
    mask = df[feature_cols].notna().all(axis=1) & y.notna()
    X = df.loc[mask, feature_cols].values.astype(np.float32)
    y = y[mask].values.astype(np.float32)

    print(f"  Training samples : {len(y):,}")
    print(f"  Feature count    : {len(feature_cols)}")
    print(f"  Target (minutes) : mean={y.mean():.2f}  std={y.std():.2f}  "
          f"median={np.median(y):.2f}")

    # train/split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n  Train set: {len(y_train):,}  |  Test set: {len(y_test):,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if tune:
        best_params = _tune_hyperparameters(X_train_s, y_train)
    else:
        best_params = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}

    print("\n=== Training final model ===")
    model = _make_xgb(**best_params, seed=random_state)

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )


    # evaluate & test
    print("\n=== Test-set evaluation ===")
    y_pred = model.predict(X_test_s)
    y_pred = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.3f} min  ({mae*60:.1f} sec)")
    print(f"  RMSE : {rmse:.3f} min  ({rmse*60:.1f} sec)")
    print(f"  R²   : {r2:.4f}")

    within_1_min  = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100
    within_2_min  = np.mean(np.abs(y_test - y_pred) <= 2.0) * 100
    print(f"  Predictions within ±1 min : {within_1_min:.1f}%")
    print(f"  Predictions within ±2 min : {within_2_min:.1f}%")

    print("\n=== Saving model artefacts ===")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"  Model   → {MODEL_PATH}")
    print(f"  Scaler  → {SCALER_PATH}")
    print(f"  Features→ {FEATURE_LIST_PATH}")

    return model

if __name__ == "__main__":
    train_model(tune=True)