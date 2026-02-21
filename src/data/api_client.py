# src/data/api_client.py
# API Client

import requests
import os
import pandas as pd
import geopandas as gpd

BASE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
RAW_DATA_DIR = "data/raw"


def _ensure_data_dir():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def get_package_metadata(dataset_id: str):
    url = f"{BASE_URL}/api/3/action/package_show"
    params = {"id": dataset_id}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_resource_info(dataset_id: str):
    package = get_package_metadata(dataset_id)

    # Prefer datastore_active resources (clean tabular)
    for resource in package["result"]["resources"]:
        if resource["datastore_active"]:
            return resource

    # Fallback: first available resource
    return package["result"]["resources"][0]


def download_csv(dataset_id: str, cache: bool = True) -> pd.DataFrame:
    _ensure_data_dir()
    file_path = os.path.join(RAW_DATA_DIR, f"{dataset_id}.csv")

    if cache and os.path.exists(file_path):
        print(f"Loading cached file: {file_path}")
        return pd.read_csv(file_path)

    resource = get_resource_info(dataset_id)
    url = resource["url"]

    print(f"Downloading {dataset_id} from API...")
    df = pd.read_csv(url)

    if cache:
        df.to_csv(file_path, index=False)

    return df


def download_geojson(dataset_id: str, cache: bool = True) -> gpd.GeoDataFrame:
    _ensure_data_dir()
    file_path = os.path.join(RAW_DATA_DIR, f"{dataset_id}.geojson")

    if cache and os.path.exists(file_path):
        print(f"Loading cached GeoJSON: {file_path}")
        return gpd.read_file(file_path)

    resource = get_resource_info(dataset_id)
    url = resource["url"]

    print(f"Downloading {dataset_id} GeoJSON from API...")
    gdf = gpd.read_file(url)

    if cache:
        gdf.to_file(file_path, driver="GeoJSON")

    return gdf