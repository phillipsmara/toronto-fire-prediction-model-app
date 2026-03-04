# Downloads raw data from City of Toronto Open Data using their API
import requests
import os

os.makedirs("data/raw", exist_ok=True)

def download_csv(package_id, target_csv_name=None):
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"

    # Get package metadata
    package = requests.get(
        f"{base_url}/api/3/action/package_show", 
        params={"id": package_id}
    ).json()

    for resource in package["result"]["resources"]:
        if not resource["datastore_active"]:
            file_url = resource["url"]
            file_name = file_url.split("/")[-1]

            if target_csv_name and file_name != target_csv_name:
                continue

            # Download file
            r = requests.get(file_url)
            with open(f"data/raw/{file_name}", "wb") as f:
                f.write(r.content)
            print(f"Downloaded {file_name} successfully!")

