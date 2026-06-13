import os
import subprocess
import time
import requests
import pandas as pd
import zipfile

# -------------------------
# Constants
# -------------------------
API_URL = "https://www.kaggle.com/api/v1/datasets/view/"
DATASET_ZIP_FOLDER = "dataset/kaggle_data_zip"
EXTRACT_FOLDER = "kaggle_data_extracted"
METADATA_CSV = "kaggle_metadata_with_data_path_shopping.csv"

# -------------------------
# 1. Search datasets by keyword using Kaggle CLI
# -------------------------
def search_datasets(query, max_results=50):
    result = subprocess.run(
        ["kaggle", "datasets", "list", "-s", query],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error searching datasets: {result.stderr}")
        return pd.DataFrame()

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        return pd.DataFrame()

    refs = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("  ") if p.strip()]
        if parts:
            refs.append(parts[0])
        if len(refs) >= max_results:
            break
    return pd.DataFrame({"ref": refs})

# -------------------------
# 2. Fetch metadata from Kaggle API
# -------------------------
def fetch_metadata(ref):
    try:
        owner, dataset = ref.split("/")
        url = API_URL + f"{owner}/{dataset}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"Cannot fetch metadata for {ref} (status {r.status_code})")
            return None
        return r.json()
    except Exception as e:
        print(f"Error fetching {ref}: {e}")
        return None

# -------------------------
# 3. Download dataset zip via Kaggle CLI
# -------------------------
def download_dataset(ref):
    os.makedirs(DATASET_ZIP_FOLDER, exist_ok=True)
    zip_path = os.path.join(DATASET_ZIP_FOLDER, ref.replace("/", "__") + ".zip")
    if os.path.exists(zip_path):
        print(f"Already downloaded: {zip_path}")
        return zip_path

    print(f"Downloading dataset {ref} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", ref, "-p", DATASET_ZIP_FOLDER, "--force"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error downloading {ref}: {result.stderr}")
        return None
    return zip_path

# -------------------------
# 4. Extract dataset zip
# -------------------------
def extract_dataset(zip_path):
    os.makedirs(EXTRACT_FOLDER, exist_ok=True)
    dataset_name = os.path.basename(zip_path).replace(".zip", "")
    folder_path = os.path.join(EXTRACT_FOLDER, dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    return folder_path

# -------------------------
# 5. Main workflow
# -------------------------
def main(query="shopping", max_results=20):
    print(f"📥 Searching datasets for '{query}'...")
    df_refs = search_datasets(query, max_results)
    if df_refs.empty:
        print("🔎 No datasets found.")
        return

    print(f"🔎 Found {len(df_refs)} datasets.\n")
    all_metadata = []

    for ref in df_refs["ref"]:
        print("==============================")
        print(f"DATASET: {ref}")
        print("==============================")

        # Fetch metadata
        meta = fetch_metadata(ref)
        if not meta:
            print("Could not fetch metadata.\n")
            continue

        # Download dataset zip
        zip_path = download_dataset(ref)
        if not zip_path:
            print("Could not download dataset.\n")
            continue

        # Extract dataset
        extracted_folder = extract_dataset(zip_path)

        # Build metadata dict
        metadata_dict = {
            "ref": ref,
            "title": meta.get("title"),
            "subtitle": meta.get("subtitle"),
            "description": meta.get("description"),
            "license": meta.get("licenseName"),
            "keywords": meta.get("keywords"),
            "totalFiles": meta.get("totalFiles"),
            "totalBytes": meta.get("totalBytes"),
            "fileTypes": [f.get("fileType") for f in meta.get("files", []) if f.get("fileType")],
            "data_path": extracted_folder,
            "keyword": query  # <-- also store the crawl keyword
        }

        all_metadata.append(metadata_dict)

        # Print summary
        for k, v in metadata_dict.items():
            if isinstance(v, str) and len(v) > 200:
                print(f"{k}: {v[:200]}...")
            else:
                print(f"{k}: {v}")
        print("\n")
        time.sleep(0.2)

    # Save metadata to CSV
    df_metadata = pd.DataFrame(all_metadata)
    df_metadata.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")
    print(f"All metadata saved to '{METADATA_CSV}'")

if __name__ == "__main__":
    main(query="shopping", max_results=10)
