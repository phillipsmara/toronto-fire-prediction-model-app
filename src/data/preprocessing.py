# Preprocesses data
import pandas as pd

def load_raw_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data_run_areas(df: pd.DataFrame):
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )
    
    columns_to_keep = [
        'RUN_AREA',
        'AREA_TYPE',
        'DATE_EFFECTIVE',
        'geometry'
    ]

    df = df[columns_to_keep].copy()

    # Remove Duplicates and Null Entries
    df = df.drop_duplicates(subset=['RUN_AREA'])
    df = df.dropna(subset=['RUN_AREA', 'geometry'])

    # Format DATE_EFFECTIVE
    df['DATE_EFFECTIVE'] = pd.to_datetime(df['DATE_EFFECTIVE'], errors='coerce')

    # Format RUN_AREA
    df['RUN_AREA'] = df['RUN_AREA'].astype(str)

    df = df.reset_index(drop=True)

    print("run_areas data cleaned successfully")
    return df


def clean_data_stations(df: pd.DataFrame):
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )
    
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
        'geometry',
    ]

    df = df[columns_to_keep].copy()

    # Drop Null Entries
    df = df.dropna(subset=['ADDRESS_NUMBER', 'LINEAR_NAME_FULL', 'STATION', 'geometry'])

    # Format Text Columns
    text_cols = ['STATION', 'WARD_NAME', 'TYPE_DESC', 'ADDRESS', 'MUNICIPALITY_NAME']

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # Ensure Years Built Data is Accurate
    if 'YEAR_BUILD' in df.columns:
        df['YEAR_BUILD'] = pd.to_numeric(df['YEAR_BUILD'], errors='coerce')

        df = df[
            (df['YEAR_BUILD'].isna()) |
            ((df['YEAR_BUILD'] > 1800) & (df['YEAR_BUILD'] <= pd.Timestamp.today().year))
        ]

    df = df.reset_index(drop=True)

    print("fire_station_locations data cleaned successfully")
    return df

def clean_data_hydrants(df: pd.DataFrame):
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    columns_to_keep = [
        'FACILITYID',
        'LOCDESC',
        'OWNEDBY',
        'MAINTBY',
        'FIELDVERIFIED',
        'GEOMETRY'
    ]

    df = df[columns_to_keep].copy()

    # Format Text Columns 
    text_cols = ['LOCDESC', 'OWNEDBY', 'MAINTBY']

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # Format FIELDVERIFIED
    if 'FIELDVERIFIED' in df.columns:
        df['FIELDVERIFIED'] = df['FIELDVERIFIED'].astype(str).str.upper()

    # Remove Duplicates
    df = df.drop_duplicates(subset=['FACILITYID'])

    df = df.reset_index(drop=True)
    print(df)
    return df

def clean_data_incidents(df: pd.DataFrame):

    return df

df = load_raw_data("data/raw/fire-hydrants-data-4326.csv")

clean_data_hydrants(df)