from src.data import load_raw_data_run_areas
from src.data import load_raw_data_fire_stations
from src.data import load_raw_data_fire_hydrants
from src.data import load_raw_data_incidents
from src.data import load_raw_data
from src.data import clean_data_run_areas
from src.data import clean_data_stations
from src.data import clean_data_hydrants
from src.data import clean_data_incidents
from src.data import generate_features

import os

def run_pipeline():
    
    # load data 
    """
    print("Loading Data:")
    load_raw_data_run_areas()
    load_raw_data_fire_hydrants()
    load_raw_data_fire_stations()
    load_raw_data_incidents()
    """

    # load data for preprocessing (data cleaning)
    fire_hydrants = load_raw_data("data/raw/fire-hydrants-data-4326.csv")
    fire_stations = load_raw_data("data/raw/fire-station-locations-4326.csv")
    fire_run_areas = load_raw_data("data/raw/toronto-fire-services-run-areas-2952.csv")

    #preprocessing
    print("Preprocessing:")
    run_areas_df = clean_data_run_areas(fire_run_areas)
    incidents_df = clean_data_incidents()
    hydrants_df = clean_data_hydrants(fire_hydrants)
    stations_df = clean_data_stations(fire_stations)