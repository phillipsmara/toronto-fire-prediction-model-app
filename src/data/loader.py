# Loads raw data via api_client.py
from .api_client import download_csv

def load_raw_data_run_areas():

    # Download Toronto Fire Services Run Areas 
    download_csv("toronto-fire-services-run-areas", "toronto-fire-services-run-areas-2952.csv")

def load_raw_data_fire_stations():
    # Download Fire Station Locations 
    download_csv("fire-station-locations", "fire-station-locations-4326.csv")

def load_raw_data_fire_hydrants():
    # Download Fire Hydrant Locations
    download_csv("fire-hydrants", "fire-hydrants-data-4326.csv")

def load_raw_data_incidents():
    # Download Fire Services Emergency Incident Basic Detail for Each Year
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2012-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2013-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2014-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2015-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2016-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2017-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2018-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2019-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2020-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2021-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2022-4326.csv")
    download_csv("fire-services-emergency-incident-basic-detail", "basic-incidents-details-2023-onward-4326.csv")