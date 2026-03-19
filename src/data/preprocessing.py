# Preprocesses data
import pandas as pd
import glob

def load_raw_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
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
        'GEOMETRY'
    ]

    df = df[columns_to_keep].copy()

    # Remove Duplicates and Null Entries
    df = df.drop_duplicates(subset=['RUN_AREA'])
    df = df.dropna(subset=['RUN_AREA', 'GEOMETRY'])

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
        'GEOMETRY',
    ]

    df = df[columns_to_keep].copy()

    # Drop Null Entries
    df = df.dropna(subset=['ADDRESS_NUMBER', 'LINEAR_NAME_FULL', 'STATION', 'GEOMETRY'])

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

    print("fire_hydrants data cleaned successfully")

    return df

def clean_data_incidents():
    # Find all incident files
    path_pattern = "data/raw/basic-incidents-details-*.csv"
    files = glob.glob(path_pattern)
    df_list = []

    # Combine all incident files into one data frame
    for file in files:
        temp_df = pd.read_csv(file, low_memory=False)
        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)

    # Standardize column names
    combined_df.columns = combined_df.columns.str.strip().str.upper()

    # Select relevant columns
    columns_to_keep = [
        'INCIDENT_NUMBER',
        'INITIAL_CAD_EVENT_TYPE',
        'INITIAL_CAD_EVENT_CALL_TYPE',
        'FINAL_INCIDENT_TYPE',
        'EVENT_ALARM_LEVEL',
        'CALL_SOURCE',
        'INCIDENT_STATION_AREA',
        'INCIDENT_WARD',
        'WARD_AT_EVENT_DISPATCH',
        'INTERSECTION',
        'TFS_ALARM_TIME',
        'TFS_ARRIVAL_TIME',
        'LAST_TFS_UNIT_CLEAR_TIME',
        'PERSONS_RESCUED',
        'GEOMETRY'
    ]

    combined_df = combined_df[[col for col in columns_to_keep if col in combined_df.columns]].copy()

    # Remove duplicate incidents
    combined_df = combined_df.drop_duplicates(subset=['INCIDENT_NUMBER'])

    # Convert datetime columns
    combined_df['TFS_ALARM_TIME'] = pd.to_datetime(combined_df['TFS_ALARM_TIME'], errors='coerce')
    combined_df['TFS_ARRIVAL_TIME'] = pd.to_datetime(combined_df['TFS_ARRIVAL_TIME'], errors='coerce')
    combined_df['LAST_TFS_UNIT_CLEAR_TIME'] = pd.to_datetime(combined_df['LAST_TFS_UNIT_CLEAR_TIME'], errors='coerce')

    combined_df = combined_df.dropna(subset=['TFS_ALARM_TIME', 'TFS_ARRIVAL_TIME'])

    # Create response time feature
    combined_df['RESPONSE_TIME_MINUTES'] = (
        (combined_df['TFS_ARRIVAL_TIME'] - combined_df['TFS_ALARM_TIME'])
        .dt.total_seconds() / 60
    )

    combined_df = combined_df[
        (combined_df['RESPONSE_TIME_MINUTES'] >= 0) &
        (combined_df['RESPONSE_TIME_MINUTES'] <= 120)
    ]

    # Temporal features
    combined_df['HOUR'] = combined_df['TFS_ALARM_TIME'].dt.hour
    combined_df['DAY_OF_WEEK'] = combined_df['TFS_ALARM_TIME'].dt.dayofweek
    combined_df['MONTH'] = combined_df['TFS_ALARM_TIME'].dt.month
    combined_df['YEAR'] = combined_df['TFS_ALARM_TIME'].dt.year

    # Clean categorical columns
    categorical_cols = [
        'INITIAL_CAD_EVENT_TYPE',
        'INITIAL_CAD_EVENT_CALL_TYPE',
        'FINAL_INCIDENT_TYPE',
        'CALL_SOURCE',
        'INTERSECTION'
    ]

    for col in categorical_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).str.strip().str.upper()

    # Clean numeric column
    if 'PERSONS_RESCUED' in combined_df.columns:
        combined_df['PERSONS_RESCUED'] = pd.to_numeric(
            combined_df['PERSONS_RESCUED'], errors='coerce'
        ).fillna(0)

    combined_df = combined_df.dropna(subset=['GEOMETRY'])

    combined_df = combined_df.reset_index(drop=True)

    print("incidents data cleaned successfully")

    return combined_df
