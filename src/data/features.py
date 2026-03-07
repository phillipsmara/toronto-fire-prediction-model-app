import pandas as pd
import numpy as np
import json
from sklearn.neighbors import BallTree


EARTH_RADIUS_M = 6371000


def extract_coordinates(df, geometry_col="GEOMETRY"):
    """
    Extract latitude and longitude from GeoJSON geometry column.
    """

    lats = []
    lons = []

    for geom in df[geometry_col]:

        try:
            geom = geom.replace('""', '"')
            g = json.loads(geom)

            lon, lat = g["coordinates"][0]

        except Exception:
            lat, lon = np.nan, np.nan

        lats.append(lat)
        lons.append(lon)

    df["latitude"] = lats
    df["longitude"] = lons

    return df




def create_time_features(df):
    """
    Create response time target and time-based features.
    """

    df = df.copy()

    df["TFS_ALARM_TIME"] = pd.to_datetime(df["TFS_ALARM_TIME"])
    df["TFS_ARRIVAL_TIME"] = pd.to_datetime(df["TFS_ARRIVAL_TIME"])

    df["response_time_seconds"] = (
        df["TFS_ARRIVAL_TIME"] - df["TFS_ALARM_TIME"]
    ).dt.total_seconds()

    df["hour_of_day"] = df["TFS_ALARM_TIME"].dt.hour
    df["day_of_week"] = df["TFS_ALARM_TIME"].dt.weekday
    df["month"] = df["TFS_ALARM_TIME"].dt.month
    df["year"] = df["TFS_ALARM_TIME"].dt.year

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df



def add_nearest_station_distance(incidents, stations):
    """
    Add distance from incident to nearest fire station.
    Uses BallTree for fast spatial lookup.
    """

    station_coords = stations[["latitude", "longitude"]].dropna().values
    incident_coords = incidents[["latitude", "longitude"]].values

    station_coords_rad = np.radians(station_coords)
    incident_coords_rad = np.radians(incident_coords)

    tree = BallTree(station_coords_rad, metric="haversine")

    distances, _ = tree.query(incident_coords_rad, k=1)

    incidents["dist_nearest_station_m"] = distances.flatten() * EARTH_RADIUS_M

    return incidents

def add_hydrant_density(incidents, hydrants, radius=500):
    """
    Count hydrants within a radius of each incident.
    Uses BallTree for efficient spatial queries.
    """

    hydrant_coords = hydrants[["latitude", "longitude"]].dropna().values
    incident_coords = incidents[["latitude", "longitude"]].values

    hydrant_coords_rad = np.radians(hydrant_coords)
    incident_coords_rad = np.radians(incident_coords)

    tree = BallTree(hydrant_coords_rad, metric="haversine")

    radius_rad = radius / EARTH_RADIUS_M

    counts = tree.query_radius(
        incident_coords_rad,
        r=radius_rad,
        count_only=True
    )

    incidents[f"hydrants_within_{radius}m"] = counts


    return incidents


def encode_categorical_features(df):
    """
    One-hot encode categorical variables.
    """

    categorical_cols = [
        "INITIAL_CAD_EVENT_TYPE",
        "INITIAL_CAD_EVENT_CALL_TYPE",
        "FINAL_INCIDENT_TYPE",
        "CALL_SOURCE",
        "EVENT_ALARM_LEVEL"
    ]

    existing_cols = [c for c in categorical_cols if c in df.columns]

    df = pd.get_dummies(
        df,
        columns=existing_cols,
        drop_first=True
    )

    return df



def build_training_dataset(
    incidents,
    run_areas,
    hydrants,
    stations
):
    """
    Main function to build the ML training dataset.
    """
    # Extract coordinates
    incidents = extract_coordinates(incidents)
    stations = extract_coordinates(stations)
    hydrants = extract_coordinates(hydrants)

    # Create time-based features and response time target
    incidents = create_time_features(incidents)

    # Add spatial features
    incidents = add_nearest_station_distance(
        incidents,
        stations
    )

    incidents = add_hydrant_density(
        incidents,
        hydrants
    )

    # Encode categorical features
    incidents = encode_categorical_features(
        incidents
    )

    # Remove geometry column
    incidents = incidents.drop(
        columns=["GEOMETRY"],
        errors="ignore"
    )
    
    # Remove rows without response time
    incidents = incidents.dropna(
        subset=["response_time_seconds"]
    )

    return incidents