import requests
import os

os.makedirs("data/raw", exist_ok=True)

# Function to download a dataset by package_id and optional target CSV
def download_csv(package_id, target_csv_name=None):
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"

    # Get package metadata
    package = requests.get(
        f"{base_url}/api/3/action/package_show", 
        params={"id": package_id}
    ).json()

    for resource in package["result"]["resources"]:
        # Only consider non-datastore resources
        if not resource["datastore_active"]:
            file_url = resource["url"]
            file_name = file_url.split("/")[-1]

            # If a target CSV is specified, skip others
            if target_csv_name and file_name != target_csv_name:
                continue

            # Download file
            r = requests.get(file_url)
            with open(f"data/raw/{file_name}", "wb") as f:
                f.write(r.content)
            print(f"Downloaded {file_name} successfully!")

# Download Toronto Fire Services Run Areas (only 2952 CSV)
download_csv("toronto-fire-services-run-areas", "toronto-fire-services-run-areas-2952.csv")

# Download Fire Station Locations (all CSVs)
download_csv("fire-station-locations", "fire-station-locations-4326.csv")