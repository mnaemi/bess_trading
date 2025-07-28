# %%
import os
import re
import zipfile
import requests
from datetime import datetime
from typing import List
import pandas as pd

# ---------------------- Core Functions ----------------------

def find_remote_sensitivity_zips(target_date: str, base_url: str = "https://nemweb.com.au/Reports/ARCHIVE/Predispatch_Sensitivities/") -> List[str]:
    """
    Find ZIP file URLs on AEMO NEMWeb whose date range includes the target_date.
    :param target_date: 'YYYY-MM-DD'
    :param base_url: URL to the sensitivity archive
    :return: List of matching full URLs
    """
    html = requests.get(base_url).text
    zip_files = sorted(set(re.findall(r'PUBLIC_PREDISPATCH_SENSITIVITIES_\d{8}_\d{8}\.zip', html)))
    target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

    matching_urls = []
    for zip_file in zip_files:
        match = re.search(r'(\d{8})_(\d{8})', zip_file)
        if match:
            start_date = datetime.strptime(match.group(1), "%Y%m%d").date()
            end_date = datetime.strptime(match.group(2), "%Y%m%d").date()
            if start_date <= target_dt <= end_date:
                matching_urls.append(base_url + zip_file)

    return matching_urls


def download_zip_file(url: str, output_dir: str = "downloads") -> str:
    """
    Download a ZIP file from the given URL to the output directory.
    :param url: URL to the ZIP file
    :param output_dir: Folder to store the downloaded file
    :return: Local file path
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(url)
    file_path = os.path.join(output_dir, filename)

    print(f"Downloading: {filename}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Download failed with status code {response.status_code}")

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print(f"Saved to: {file_path}")
    return file_path


def unzip_file(zip_path: str, extract_to: str = "extracted") -> List[str]:
    """
    Unzip contents of a ZIP file to a directory.
    :param zip_path: Path to ZIP file
    :param extract_to: Folder to extract into
    :return: List of full paths to extracted files
    """
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = [os.path.join(extract_to, name) for name in zip_ref.namelist()]

    print(f"Extracted {len(extracted_files)} files to {extract_to}")
    return extracted_files


def find_target_date_zipfiles(folder_path: str, target_date: str, extract_to: str = "nested_extracted") -> List[str]:
    """
    Find and unzip ZIP files inside a folder that match a specific date in their filename.
    :param folder_path: Folder where nested ZIPs exist
    :param target_date: Target date in 'YYYY-MM-DD'
    :param extract_to: Where to extract matched ZIPs
    :return: List of full paths to extracted files
    """
    target_yyyymmdd = datetime.strptime(target_date, "%Y-%m-%d").strftime("%Y%m%d")
    os.makedirs(extract_to, exist_ok=True)

    extracted_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            match = re.search(r'_(\d{14})_', file)
            if match and match.group(1)[:8] == target_yyyymmdd:
                nested_zip_path = os.path.join(folder_path, file)
                with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    extracted_files += [os.path.join(extract_to, f) for f in zip_ref.namelist()]

    print(f"Nested ZIPs extracted for {target_date}: {len(extracted_files)} files")
    return extracted_files


def load_vic1_rows_from_csvs(folder_path: str) -> pd.DataFrame:
    """
    Reads all CSV files in a folder, skips the first row, and filters rows where REGIONID == 'VIC1'.

    :param folder_path: Path to folder containing CSV files
    :return: Concatenated DataFrame of VIC1 rows
    """
    all_dfs = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                # Skip first row, header on second row (row index 1)
                df = pd.read_csv(file_path, skiprows=[0])
                if 'REGIONID' in df.columns:
                    df_vic1 = df[df['REGIONID'] == 'VIC1']
                    if not df_vic1.empty:
                        all_dfs.append(df_vic1)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        print("No matching VIC1 rows found.")
        return pd.DataFrame()


# ---------------------- Example Usage ----------------------

if __name__ == "__main__":
    target_date = "2025-06-26"
#%%
    # Step 1: Find matching archive ZIPs from NEMWeb
    urls = find_remote_sensitivity_zips(target_date)

    if not urls:
        print(f"No archive files found covering {target_date}")
        exit()

    # Step 2: Download first match
    downloaded_zip = download_zip_file(urls[0])

    # Step 3: Unzip it
    top_level_files = unzip_file(downloaded_zip, extract_to="extracted")

    # Step 4: Find and extract nested ZIPs for the target day
    matched_csvs = find_target_date_zipfiles("pd_sensitivities_extract", target_date, extract_to="pd_sensitivities")
    df = load_vic1_rows_from_csvs('pd_sensitivities')
    df.to_csv(f"vic1_sensitivities_{target_date}.csv", index=False)


# %%
