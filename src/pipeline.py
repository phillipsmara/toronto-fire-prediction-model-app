from src.data import load_raw_data_run_areas
from src.data import load_raw_data_fire_stations
from src.data import load_raw_data_fire_hydrants
from src.data import load_raw_data_incidents
from src.data import load_raw_data
from src.data import clean_data_run_areas
from src.data import clean_data_stations
from src.data import clean_data_hydrants
from src.data import clean_data_incidents
from src.data import build_training_dataset

import os

PROCESSED_PATH = "data/processed/training_dataset.csv"


def run_pipeline():

    # load data
    print("Loading Data:")
    load_raw_data_run_areas()
    load_raw_data_fire_hydrants()
    load_raw_data_fire_stations()
    load_raw_data_incidents()

    # load data for preprocessing (data cleaning)
    fire_hydrants  = load_raw_data("data/raw/fire-hydrants-data-4326.csv")
    fire_stations  = load_raw_data("data/raw/fire-station-locations-4326.csv")
    fire_run_areas = load_raw_data("data/raw/toronto-fire-services-run-areas-2952.csv")

    # preprocessing
    print("Preprocessing:")
    run_areas_df  = clean_data_run_areas(fire_run_areas)
    incidents_df  = clean_data_incidents()
    hydrants_df   = clean_data_hydrants(fire_hydrants)
    stations_df   = clean_data_stations(fire_stations)

    training_df = build_training_dataset(
        incidents_df,
        run_areas_df,
        hydrants_df,
        stations_df,
    )

    os.makedirs("data/processed", exist_ok=True)
    training_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Training dataset saved to {PROCESSED_PATH}")

    print("complete")


def run_training(tune: bool = True):
    """Train the XGBoost model on the processed dataset."""
    from src.modeling.train import train_model
    print("\n=== Model Training ===")
    train_model(csv_path=PROCESSED_PATH, tune=tune)


def run_explainability():
    """Generate feature importance and SHAP plots."""
    from src.modeling.explainability import explain_model
    print("\n=== Model Explainability ===")
    explain_model(csv_path=PROCESSED_PATH)