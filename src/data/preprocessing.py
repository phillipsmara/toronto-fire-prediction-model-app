# Preprocesses data
import pandas as pd

def load_raw_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data_run_areas(df: pd.DataFrame):
    columns_to_keep = [
        'RUN_AREA',
        'AREA_TYPE',
        'DATE_EFFECTIVE',
        'geometry'
    ]

    df = df[columns_to_keep].copy()
    df = df.drop_duplicates(subset=['RUN_AREA'])
    df = df.dropna(subset=['RUN_AREA', 'geometry'])

    df['DATE_EFFECTIVE'] = pd.to_datetime(df['DATE_EFFECTIVE'], errors='coerce')

    df['RUN_AREA'] = df['RUN_AREA'].astype(str)

    df = df.reset_index(drop=True)

    print("run_areas data cleaned successfully")
    return df


def clean_data_stations(df: pd.DataFrame):
    columns_to_keep = [
        'ADDRESS_NUMBER',
        'LINEAR_NAME_FULL',
        'ADDRESS',
        'MUNICIPALITY_NAME',
        'STATION',
        'WARD',
        'WARD_NAME',
        'TYPE_DESC',
        'YEAR_BUILD',
        'FIRE_PREV_OFFICE',
        'geometry',
    ]

    df = df[columns_to_keep].copy()
    df = df.dropna(subset=['ADDRESS_NUMBER', 'LINEAR_NAME_FULL', 'STATION', 'geometry'])

    df['YEAR_BUILD'] = pd.to_datetime(df['YEAR_BUILD'], errors='coerce')

    df = df.reset_index(drop=True)

    print(df)
    print("fire_station_locations data cleaned successfully")
    return df

def clean_data_hydrants(df: pd.DataFrame):

    return df

def clean_data_incidents(df: pd.DataFrame):

    return df

df = load_raw_data("data/raw/fire-station-locations-4326.csv")

clean_data_stations(df)