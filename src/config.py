from __future__ import annotations
import os

# Project root — works locally and on Streamlit Cloud
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAINING_CSV       = os.path.join(DATA_PROCESSED_DIR, "training_dataset.csv")

RAW_FILES = {
    "hydrants":  os.path.join(DATA_RAW_DIR, "fire-hydrants-data-4326.csv"),
    "stations":  os.path.join(DATA_RAW_DIR, "fire-station-locations-4326.csv"),
    "run_areas": os.path.join(DATA_RAW_DIR, "toronto-fire-services-run-areas-2952.csv"),
}

# Model artefact paths
MODEL_DIR          = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH         = os.path.join(MODEL_DIR, "xgb_response_time.joblib")
SCALER_PATH        = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_LIST_PATH  = os.path.join(MODEL_DIR, "feature_list.json")
EXPLAINABILITY_DIR = os.path.join(MODEL_DIR, "explainability")

# Model training settings
TARGET_COL                = "response_time_seconds"
TEST_SIZE                 = 0.2
RANDOM_STATE              = 42
MAX_RESPONSE_TIME_MINUTES = 120

EXCLUDE_COLS = [
    "INCIDENT_NUMBER", "TFS_ALARM_TIME", "TFS_ARRIVAL_TIME",
    "LAST_TFS_UNIT_CLEAR_TIME", "GEOMETRY", "INTERSECTION",
    "latitude", "longitude", "response_time_seconds",
    "RESPONSE_TIME_MINUTES", "HOUR", "DAY_OF_WEEK", "MONTH", "YEAR",
]

XGB_DEFAULT_PARAMS = {
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
}

# Feature engineering settings
HYDRANT_RADIUS_M = 500
EARTH_RADIUS_M   = 6371000

# App settings
APP_TITLE  = "Toronto Fire Response Time Predictor"
APP_ICON   = "🚒"
APP_LAYOUT = "centered"

INCIDENT_TYPES = [
    "Medical", "Fire - Residential", "Fire - Commercial", "Fire - Vehicle",
    "Alarm - False", "Alarm - Undetermined", "Rescue", "Hazardous Materials", "Other",
]

CALL_SOURCES = ["Alarm", "Phone", "Radio", "Transfer", "Walk-in"]

ALARM_LEVELS = ["1", "2", "3", "4"]

TORONTO_WARDS = [str(i) for i in range(1, 26)]

STATION_AREAS = [str(i) for i in range(1, 86)]