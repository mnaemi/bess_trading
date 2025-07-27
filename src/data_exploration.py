#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def price_forecast_analysis(file, trading_date ='2025-06-26' , visualize=True, region="VIC1"):
    df_forecast = pd.read_excel(file, sheet_name="Price Forecast")

    # Filter for VIC1 region
    df_vic = df_forecast[df_forecast["RegionID"] == region].copy()

    # Pivot the data to wide format: one column per percentile
    df_pivot = df_vic.pivot(index="HalfHourInterval", columns="Percentile", values="Value").reset_index()
    df_pivot["HalfHourInterval"] = pd.to_datetime(df_pivot["HalfHourInterval"])

    # Filter for time range: 2025-06-26 04:30 to 2025-06-27 04:00
    start_time = pd.Timestamp(f"{trading_date} 04:30:00")
    end_time = start_time + pd.Timedelta(hours=23.5)  # 24 hours later
    df_filtered = df_pivot[(df_pivot["HalfHourInterval"] >= start_time) & (df_pivot["HalfHourInterval"] <= end_time)]

    # Reshape to long format for plotting
    df_melted = df_filtered.melt(
        id_vars="HalfHourInterval",
        value_vars=["P10", "P25", "P50", "P75", "P90"],
        var_name="Quantile",
        value_name="Price"
    )

    if visualize:
        # Plot 1: Time Series
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_melted, x="HalfHourInterval", y="Price", hue="Quantile", palette="muted")
        plt.title(f"Price Forecast Quantiles - {region} - {trading_date}")
        plt.xlabel("Time")
        plt.ylabel("Price ($/MWh)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot 2: Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_melted,
            x="Price",
            hue="Quantile",
            multiple="stack",
            stat="probability",
            bins=50,
            palette="muted"
        )
        plt.title(f"Probability Distribution of Price Forecast Quantiles - {region} - {trading_date}")
        plt.xlabel("Price ($/MWh)")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.show()
    return df_melted.groupby('Quantile')['Price'].describe()


# %%
def supply_demand_analysis(file, trading_date='2025-06-26', visualize=True, region="VIC1"):
    # Load the Supply and Demand Forecast sheet
    df_sd = pd.read_excel(file, sheet_name="Supply and Demand Forecast")

    # Convert the HalfHourInterval to datetime
    df_sd["HalfHourInterval"] = pd.to_datetime(df_sd["HalfHourInterval"])

    start_time = pd.Timestamp(f"{trading_date} 04:30:00")
    end_time = start_time + pd.Timedelta(hours=23.5)  # 24 hours later
   
    df_sd_filtered = df_sd[
        (df_sd["HalfHourInterval"] >= start_time) &
        (df_sd["HalfHourInterval"] <= end_time)
    ].copy()

    # Rename for convenience and compute margin
    df_sd_filtered["Demand"] = df_sd_filtered["Demand MW"]
    df_sd_filtered["Supply"] = df_sd_filtered["Supply MW"]
    df_sd_filtered["Margin"] = df_sd_filtered["Supply"] - df_sd_filtered["Demand"]

    df_sd_filtered["Margin"] = df_sd_filtered["Supply"] - df_sd_filtered["Demand"]

    if visualize:
    # Plot: Supply, Demand, and Supply-Demand Margin
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_sd_filtered, x="HalfHourInterval", y="Demand", label="Demand", color="red")
        sns.lineplot(data=df_sd_filtered, x="HalfHourInterval", y="Supply", label="Supply", color="green")
        sns.lineplot(data=df_sd_filtered, x="HalfHourInterval", y="Margin", label="Supply - Demand Margin", color="blue")
        plt.title("Supply, Demand, and Margin Forecast - VIC1 - 26 June 2025")
        plt.xlabel("Time")
        plt.ylabel("MW")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

#%%
# Load Excel file and read the price forecast sheet
file_path = "./inputs/Quant Trader Assessment Data.xlsx"
xls = pd.ExcelFile(file_path)
price_forecast_analysis(file=xls)
supply_demand_analysis(file=xls)
# %%
