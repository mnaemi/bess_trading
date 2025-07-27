#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

schedule_pricetaker = pd.read_csv('./outputs/schedule_price_taker.csv')
schedule_pricemaker_mean = pd.read_csv('./outputs/schedule_price_maker_beta_mean.csv')
schedule_pricemaker_max = pd.read_csv('./outputs/schedule_price_maker_beta_max.csv')
schedule_pricemaker_mean['DATETIME'] = pd.to_datetime(schedule_pricemaker_mean['DATETIME'])
schedule_pricemaker_max['DATETIME'] = pd.to_datetime(schedule_pricemaker_max['DATETIME'])

fc_long = pd.read_csv('./inputs/price_forecast.csv', parse_dates=["HalfHourInterval"])

forecast_df = (
    fc_long.pivot(index="HalfHourInterval",
                  columns="Percentile",
                  values="Value")
             .reset_index()
             .rename(columns={"HalfHourInterval": "DATETIME"})
)

# keep only the percentiles the optimiser actually uses
needed_cols = ["DATETIME", "P5", "P10", "P25", "P50", "P75", "P90", "P95"]
for col in needed_cols[1:]:
    # if some percentiles are missing just fill with NaN – the code handles it
    if col not in forecast_df.columns:
        forecast_df[col] = pd.NA
forecast_df = forecast_df[needed_cols]
mask = (forecast_df['DATETIME'] >= "2025-06-26 04:05") & (forecast_df['DATETIME'] <= "2025-06-27 04:00")
forecast_df = forecast_df.loc[mask]

# %%
plt.figure(figsize=(10, 6))
sns.lineplot(data=schedule_pricetaker, x=schedule_pricetaker.DATETIME, y='net_MW', label='Price Taker')
sns.lineplot(data=schedule_pricemaker_mean, x=schedule_pricemaker_mean.DATETIME, y='net_MW', label='Price Maker')
plt.xlabel('Index')
plt.ylabel('Net MW')
plt.title('Net MW Comparison: Price Taker vs Price Maker')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
# %%
total_revenue = {
    'Price Taker': schedule_pricetaker['revenue'].sum(),
    'Price Maker': schedule_pricemaker_mean['revenue'].sum()
}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(total_revenue.keys()), y=list(total_revenue.values()), palette='viridis')
for i, v in enumerate(total_revenue.values()):
    plt.text(i, v, f'{v:,.2f}', ha='center', va='bottom')
plt.ylabel('Total Revenue')
plt.ylim(4e6,6e6)
plt.title('Total Revenue: Price Taker vs Price Maker')
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.lineplot(data=schedule_pricetaker, x=schedule_pricetaker.DATETIME, y='price_expected', label='Price Taker Exp Price')
sns.lineplot(data=schedule_pricemaker_mean, x=schedule_pricemaker_mean.DATETIME, y='price_expected', label='Price Maker Exp Price')
plt.xlabel('Index')
plt.ylabel('Expected Price')
plt.title('Expected Price Comparison: Price Taker vs Price Maker')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Calculate total revenue divided by (sum(discharge_MW)/2) for price maker
total_revenue_pm = schedule_pricemaker_mean['revenue'].sum()
total_discharge_pm = schedule_pricemaker_mean['discharge_MW'].sum()
result = total_revenue_pm / (total_discharge_pm / 2)
total_revenue_pm_bmax = schedule_pricemaker_max['revenue'].sum()
total_discharge_pm_bmax = schedule_pricemaker_max['discharge_MW'].sum()
result_bmax = total_revenue_pm_bmax / (total_discharge_pm_bmax / 2)
print(f"Spread Captured for Price Maker (beta_mean): {result:,.2f}")
print(f"Spread Captured for Price Maker (beta_max): {result_bmax:,.2f}")

# %%
total_revenue_pm = schedule_pricemaker_mean['revenue'].sum()
total_discharge_pm = schedule_pricemaker_mean['discharge_MW'].sum()
result = total_revenue_pm / (total_discharge_pm /2/ 200)
total_revenue_pm_bmax = schedule_pricemaker_max['revenue'].sum()
total_discharge_pm_bmax = schedule_pricemaker_max['discharge_MW'].sum()
result_bmax = total_revenue_pm_bmax / (total_discharge_pm_bmax / 2/ 200)
print(f"$/cycle for Price Maker (beta_mean): {result:,.2f}")
print(f"$/cycle for Price Maker (beta_max): {result_bmax:,.2f}")

# %%
def merge_and_compare_revenue(schedule_df, forecast_df, percentile_cols):
    merged = pd.merge(schedule_df, forecast_df, on="DATETIME", how="left")
    revenues = {}
    for perc in percentile_cols:
        if perc not in merged.columns:
            continue
        # Calculate revenue: net_MW * price at percentile, sum over all intervals
        merged['revenue_' + perc] = merged['net_MW'] * merged[perc] / 2  # divide by 2 for half-hour intervals
        revenues[perc] = merged['revenue_' + perc].sum()
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(revenues.keys()), y=list(revenues.values()), palette='mako')
    for i, v in enumerate(revenues.values()):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue by Forecast Percentile (Price Maker)')
    plt.tight_layout()
    plt.show()

percentile_cols = ["P5", "P10", "P25", "P50", "P75", "P90", "P95"]
merge_and_compare_revenue(schedule_pricemaker_mean, forecast_df, percentile_cols)
# %%
