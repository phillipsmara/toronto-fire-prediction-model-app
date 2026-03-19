"""
Predict fire response time (minutes) for new incidents.

"""
from __future__ import annotations

import os
import json
import argparse

import joblib
import numpy as np
import pandas as pd


# Paths (must match train.py)

MODEL_DIR        = "models"
MODEL_PATH       = os.path.join(MODEL_DIR, "xgb_response_time.joblib")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")


# Load artefacts

def load_model():
    """
    Load the trained XGBoost model, scaler, and feature list from disk.

    Returns
    -------
    model        : fitted XGBRegressor
    scaler       : fitted StandardScaler
    feature_cols : list of feature names expected by the model
    """
    for path in [MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artefact not found: {path}\n"
                "Run `python -m src.modeling.train` first."
            )

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    print(f"  Model loaded from  : {MODEL_PATH}")
    print(f"  Expects {len(feature_cols)} features")

    return model, scaler, feature_cols


# Core prediction function

def predict_response_time(
    df: pd.DataFrame,
    model=None,
    scaler=None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate response-time predictions for a DataFrame of incidents.

    The input DataFrame should contain the same feature columns that were
    produced by `src.data.features.build_training_dataset`.  Any missing
    feature columns are filled with 0 and a warning is printed.

    Parameters
    ----------
    df           : DataFrame of incidents (pre-processed / feature-engineered)
    model        : (optional) pre-loaded XGBRegressor; loaded from disk if None
    scaler       : (optional) pre-loaded StandardScaler; loaded from disk if None
    feature_cols : (optional) list of feature names; loaded from disk if None

    Returns
    -------
    DataFrame with columns:
        predicted_response_time_min  – point estimate (minutes)
        predicted_response_time_sec  – same value in seconds
    """

    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols = load_model()

    # Align columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} feature(s) not found in input — "
              f"filling with 0: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        for col in missing:
            df[col] = 0.0

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    # Scale and predict
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    preds = np.clip(preds, 0, None)          # no negative response times

    result = df.copy()
    result["predicted_response_time_min"] = np.round(preds, 3)
    result["predicted_response_time_sec"] = np.round(preds * 60, 1)

    return result


# Predict from raw CSV 

def predict_from_raw_csv(
    input_csv: str,
    output_csv: str | None = None,
    hydrants_csv: str = "data/raw/fire-hydrants-data-4326.csv",
    stations_csv: str = "data/raw/fire-station-locations-4326.csv",
) -> pd.DataFrame:
    """
    End-to-end prediction starting from a raw incidents CSV.
    Runs the same feature-engineering steps as the training pipeline.

    Parameters
    ----------
    input_csv    : Path to a raw incidents CSV
    output_csv   : If provided, write predictions to this path
    hydrants_csv : Path to the hydrants CSV (for spatial features)
    stations_csv : Path to the fire station CSV (for spatial features)

    Returns
    -------
    DataFrame with predictions appended
    """
    from src.data.preprocessing import (
        load_raw_data,
        clean_data_incidents,
        clean_data_hydrants,
        clean_data_stations,
    )
    from src.data.features import (
        extract_coordinates,
        create_time_features,
        add_nearest_station_distance,
        add_hydrant_density,
        encode_categorical_features,
    )

    print(f"\n=== Loading raw incident data: {input_csv} ===")
    # clean_data_incidents reads from glob pattern; for a single file we
    # temporarily load and process it directly
    raw_df = load_raw_data(input_csv)

    hydrant_raw  = load_raw_data(hydrants_csv)
    stations_raw = load_raw_data(stations_csv)

    hydrants_df  = clean_data_hydrants(hydrant_raw)
    stations_df  = clean_data_stations(stations_raw)

    # Standardise column names (same as clean_data_incidents)
    raw_df.columns = raw_df.columns.str.strip().str.upper()

    print("=== Applying feature engineering ===")
    df = extract_coordinates(raw_df)
    stations_df = extract_coordinates(stations_df)
    hydrants_df = extract_coordinates(hydrants_df)

    df = create_time_features(df)
    df = add_nearest_station_distance(df, stations_df)
    df = add_hydrant_density(df, hydrants_df)
    df = encode_categorical_features(df)

    print("=== Running predictions ===")
    result = predict_response_time(df)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"  Predictions written to: {output_csv}")

    return result

# Predict from already-processed CSV 

def predict_from_processed_csv(
    input_csv: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Predict from a pre-processed CSV (same schema as training_dataset.csv).

    Parameters
    ----------
    input_csv  : Path to pre-processed incidents CSV
    output_csv : If provided, write predictions to this path

    Returns
    -------
    DataFrame with predictions appended
    """
    print(f"\n=== Loading processed data: {input_csv} ===")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"  Rows: {len(df):,}")

    print("=== Running predictions ===")
    result = predict_response_time(df)

    # Summary stats
    preds = result["predicted_response_time_min"]
    print(f"\n  Prediction summary:")
    print(f"    Mean   : {preds.mean():.2f} min")
    print(f"    Median : {preds.median():.2f} min")
    print(f"    Std    : {preds.std():.2f} min")
    print(f"    Min    : {preds.min():.2f} min")
    print(f"    Max    : {preds.max():.2f} min")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"\n  Predictions written to: {output_csv}")

    return result


# CLI

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Predict fire response time for new incidents."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (raw or processed)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write predictions CSV (optional)"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="If set, treat input as a raw incidents file and run "
             "full feature engineering before predicting"
    )
    parser.add_argument(
        "--hydrants", default="data/raw/fire-hydrants-data-4326.csv",
        help="Path to hydrants CSV (only used with --raw)"
    )
    parser.add_argument(
        "--stations", default="data/raw/fire-station-locations-4326.csv",
        help="Path to stations CSV (only used with --raw)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.raw:
        predict_from_raw_csv(
            input_csv=args.input,
            output_csv=args.output,
            hydrants_csv=args.hydrants,
            stations_csv=args.stations,
        )
    else:
        predict_from_processed_csv(
            input_csv=args.input,
            output_csv=args.output,
        )