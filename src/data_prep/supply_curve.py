#%%
import requests
import re
from datetime import datetime
from typing import List
import os
import zipfile
import pandas as pd

def get_bid_offer_file_links(target_date: str) -> List[str]:
    """
    For a given target date (YYYY-MM-DD), return URLs for BIDDAYOFFER_D and BIDPEROFFER_D zip files
    from AEMO's MMSDM archive.
    """
    base_template = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
    
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    year = dt.year
    month = dt.month
    yyyymmdd = dt.strftime("%Y%m%d")
    url = base_template.format(year=year, month=month)
    print(f"Accessing: {url}")

    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Failed to retrieve listing for {url}")

    pattern = re.compile(
        rf'PUBLIC_ARCHIVE#(BIDDAYOFFER_D|BIDPEROFFER_D)#FILE\d+#({yyyymmdd}0000)\.zip'
    )
    matches = pattern.findall(r.text)

    file_links = [
        f"{url}PUBLIC_ARCHIVE%23{table}%23FILE01%23{yyyymmdd}0000.zip"
        for table, _ in matches
    ]

    return file_links


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
#%%
target = "2025-06-25"
files_links = get_bid_offer_file_links(target)
for file in files_links[0:1]:
    downloaded_zips = download_zip_file(file)

# %%
unzip_file(downloaded_zips, extract_to= "supply_curve_files")
# %%
biddayoffer_files = [f for f in os.listdir("supply_curve_files") if "BIDDAYOFFER_D" in f]
bidperoffer_files = [f for f in os.listdir("supply_curve_files") if "BIDPEROFFER_D" in f]

bid_day_offer = pd.read_csv(f"supply_curve_files/{biddayoffer_files[0]}", skiprows=1)
#%%
bid_per_offer = pd.read_csv(f"supply_curve_files/{bidperoffer_files[0]}", skiprows=1)
# %%
bid_day_offer['SETTLEMENTDATE'] = pd.to_datetime(bid_day_offer['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
bid_day_offer = bid_day_offer.loc[(bid_day_offer['BIDTYPE']=='ENERGY')&(bid_day_offer['SETTLEMENTDATE'].isin(['2025-06-25','2025-06-26']))]

bid_per_offer['SETTLEMENTDATE'] = pd.to_datetime(bid_per_offer['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
bid_per_offer = bid_per_offer.loc[(bid_per_offer['BIDTYPE']=='ENERGY')&(bid_per_offer['SETTLEMENTDATE'].isin(['2025-06-25','2025-06-26']))]

#%%
duids = pd.read_csv("./inputs/duids.csv", encoding='windows-1252')
duids = duids[['DUID','Region']]
#%%
pricebands = ['PRICEBAND{}'.format(i) for i in range(1, 11)]
volbands = ['BANDAVAIL{}'.format(i) for i in range(1, 11)]
bid_per_offer_cols = ['SETTLEMENTDATE', 'INTERVAL_DATETIME', 'BIDTYPE', 'DIRECTION', 'DUID' ,'MAXAVAIL'] + volbands
bid_day_offer_cols = ['SETTLEMENTDATE', 'BIDTYPE', 'DIRECTION', 'DUID'] + pricebands

df = bid_day_offer[bid_day_offer_cols].merge(bid_per_offer[bid_per_offer_cols], on=['SETTLEMENTDATE', 'BIDTYPE', 'DUID', 'DIRECTION'])
df = df.merge(duids, on='DUID', how='left')
# %%
def cap_bandavails_to_maxavail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts BANDAVAIL1–10 so their row-wise sum does not exceed MAXAVAIL.
    Keeps early bands unchanged until MAXAVAIL is exceeded, then truncates and zeros out the rest.
    
    :param df: DataFrame with BANDAVAIL1–10 and MAXAVAIL columns
    :return: Adjusted DataFrame
    """
    band_cols = [f"BANDAVAIL{i}" for i in range(1, 11)]
    df_adjusted = df.copy()

    def adjust_row(row):
        total = 0
        for col in band_cols:
            val = row[col]
            if total + val <= row["MAXAVAIL"]:
                total += val
            else:
                # Truncate this band to fit and zero out the rest
                row[col] = max(0, row["MAXAVAIL"] - total)
                idx = band_cols.index(col)
                for rem_col in band_cols[idx + 1:]:
                    row[rem_col] = 0
                break
        return row

    df_adjusted = df_adjusted.apply(adjust_row, axis=1)
    return df_adjusted

def build_supply_curve(df, interval, region='VIC1'):
    """
    Create supply curve for a given settlement interval.
    
    Parameters:
        df (pd.DataFrame): Price band and availability data.
        interval (str or pd.Timestamp): Target SETTLEMENTDATE to filter on.

    Returns:
        pd.DataFrame: Supply curve with sorted (price, volume) and cumulative volume.
    """
    # Ensure datetime format for comparison
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'])
    interval = pd.to_datetime(interval)

    # Filter for the given interval
    df_interval = df[(df['INTERVAL_DATETIME'] == interval) & (df['Region']==region)]
    if df_interval.empty:
        raise ValueError(f"No data found for interval: {interval}")

    # Prepare lists to collect all (price, volume) pairs
    price_volume_pairs = []
    price_cols = [f'PRICEBAND{i}' for i in range(1, 11)]
    avail_cols = [f'BANDAVAIL{i}' for i in range(1, 11)]

    for _, row in df_interval.iterrows():
        prices = row[price_cols].to_numpy(dtype=float)
        volumes = row[avail_cols].to_numpy(dtype=float)

        for p, v in zip(prices, volumes):
            if pd.notna(p) and pd.notna(v) and v > 0:
                price_volume_pairs.append((p, v))

    # Convert to DataFrame
    curve_df = pd.DataFrame(price_volume_pairs, columns=['Price', 'Volume'])

    # Sort by price
    curve_df = curve_df.sort_values(by='Price').reset_index(drop=True)

    # Add cumulative volume
    curve_df['CumulativeVolume'] = curve_df['Volume'].cumsum()

    return curve_df


# %%
df = cap_bandavails_to_maxavail(df)
# %%
interval = pd.Timestamp("2025-06-25 17:30:00")
supply_curve = build_supply_curve(df, interval, region='VIC1')

# %%
import matplotlib.pyplot as plt

def plot_supply_curve(supply_curve, title="Supply Curve"):
    """
    Plot cumulative volume vs price from a supply curve DataFrame.
    """
    plt.figure(figsize=(8, 5))
    plt.step(supply_curve['CumulativeVolume'], supply_curve['Price'], where='post')
    plt.xlabel('Price ($/MWh)')
    plt.ylabel('Cumulative Volume (MW)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_supply_curve(supply_curve, title=f"Supply Curve for {interval} VIC1")
# %%
