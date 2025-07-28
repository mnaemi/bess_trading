
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import numpy as np

def prepare_data_for_elasticity(df_senstivities, df_predispatch):
    sel_cols = ['REGIONID', 'DATETIME', 'RRPEEP8', 'RRPEEP9','LASTCHANGED']
    df_senstivities =  df_senstivities[sel_cols]
    sel_cols = ['REGIONID', 'PERIODID', 'RRP', 'TOTALDEMAND','PREDISPATCHSEQNO']
    df_predispatch =  df_predispatch[sel_cols]
    df_predispatch.rename(columns={'PERIODID': 'DATETIME','PREDISPATCHSEQNO':'LASTCHANGED'}, inplace=True)
    df_senstivities['DATETIME'] = pd.to_datetime(df_senstivities['DATETIME'])
    df_predispatch['DATETIME'] = pd.to_datetime(df_predispatch['DATETIME'], format='%Y/%m/%d %H:%M:%S')
    df_senstivities['LASTCHANGED'] = pd.to_datetime(df_senstivities['LASTCHANGED'], format='%Y/%m/%d %H:%M:%S')
    df_predispatch['LASTCHANGED'] = pd.to_datetime(df_predispatch['LASTCHANGED'], format='%Y/%m/%d %H:%M:%S')

    df_senstivities['LASTCHANGED'] = df_senstivities['LASTCHANGED'].dt.round('30min')
    df = df_predispatch.merge(df_senstivities, on=['REGIONID', 'DATETIME', 'LASTCHANGED'], how='left')

    return df




def create_price_elasticity_curve(df, demand_offset_map):
    
    for col in demand_offset_map.keys():
        df[f'delta_{col}'] = df[col] - df['RRP']
    price_elasticity = df.melt(
        id_vars=['REGIONID', 'DATETIME', 'RRP', 'TOTALDEMAND','LASTCHANGED'],
        value_vars=list(demand_offset_map.keys()),
        var_name='RRPEEP',
        value_name='PriceSensitivity'
    )

    # Map demand offset and calculate price delta
    price_elasticity['DemandOffset'] = price_elasticity['RRPEEP'].map(demand_offset_map)
    price_elasticity['PriceDelta'] = price_elasticity['PriceSensitivity'] - price_elasticity['RRP']
    price_elasticity.dropna(axis=0, inplace=True)
    price_elasticity.to_csv('./inputs/price_elasticity.csv')
    return price_elasticity

def beta_from_elasticity(
    df: pd.DataFrame,
    clip: Tuple[float, float] = (-300.0, 100.0),
) -> pd.Series:
    """Return Series indexed by DATETIME with β_t (negative)."""

    def _beta_one(group: pd.DataFrame) -> float:
        dq = group["DemandOffset"].astype(float).values
        dp = group["PriceDelta"].astype(float).values

        slopes = dp / dq
        beta_raw = np.max(slopes)

        beta = -beta_raw
        if clip is not None:
            beta = float(np.clip(beta, clip[0], clip[1]))
        return beta

    return (
        df.groupby("DATETIME", group_keys=False)
          .apply(_beta_one)
          .rename("beta")
          .sort_index()
    )

def heatmap_price_elasticity(price_elasticity, interval="2025-06-26 17:30:00"):

    subset = price_elasticity[price_elasticity['DATETIME'] == interval]
    subset['RoundedDemand'] = 50 * round(subset['TOTALDEMAND'] / 50)

    heatmap_data = subset.pivot_table(
        index='DemandOffset',
        columns='RoundedDemand',
        values='PriceDelta',
        aggfunc='mean'
    )

    # Step 7: Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='coolwarm', center=0, annot=False, fmt=".1f")
    plt.title('Price Elasticity Heatmap')
    plt.xlabel('Total Demand (MW)')
    plt.ylabel('Demand Offset (MW)')
    plt.tight_layout()
    plt.show()

def plot_price_elasticity(price_elasticity, trading_date="2025-06-26"):
    start_time = pd.Timestamp(f"{trading_date} 04:30:00")
    end_time = start_time + pd.Timedelta(hours=23.5)  # 24 hours later

    price_elasticity = price_elasticity[
        (price_elasticity["DATETIME"] >= start_time) &
        (price_elasticity["DATETIME"] <= end_time)
    ].copy()
    plt.figure(figsize=(14, 10))
    sns.boxplot(data=price_elasticity, x="DATETIME", y="PriceDelta", fliersize=0, color="lightgray", linewidth=1)
    sns.stripplot(data=price_elasticity, x="DATETIME", y="PriceDelta", color="black", size=2, alpha=0.4, jitter=0.2)
    plt.xticks(rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title("PriceDelta Across the Day", fontsize=18)
    plt.xlabel("Time of Day", fontsize=16)
    plt.ylabel("PriceDelta ($/MWh)", fontsize=16)
    plt.tight_layout()
    plt.show()

demand_offset_map = {
    'RRPEEP8':  +100,
    'RRPEEP9':  -100,
    # 'RRPEEP10': +200,
    # 'RRPEEP11': -200,
    # 'RRPEEP12': +500,
    # 'RRPEEP13': -500,
    # 'RRPEEP14': +1000
}
df_senstivities = pd.read_csv("./inputs/vic1_sensitivities_2025-06-26.csv")
df_predispatch = pd.read_csv("./inputs/vic1_predispatch_2025-06-26.csv")
df_predispatch = df_predispatch.loc[df_predispatch['5'] == 5]
df= prepare_data_for_elasticity(df_senstivities, df_predispatch)
price_elasticity = create_price_elasticity_curve(df, demand_offset_map)
plot_price_elasticity(price_elasticity,"2025-06-26")

# %%


ELAS_FILE  = "./inputs/price_elasticity.csv"

elasticity = pd.read_csv(ELAS_FILE)
beta_series = beta_from_elasticity(elasticity)

# before you call reindex
beta_series.index = pd.to_datetime(beta_series.index)   # key line
beta_series.to_csv("./inputs/beta_series_max.csv")
plt.plot(beta_series)
plt.ylabel("beta ($/MW)")
plt.xticks(rotation = 90)
# %%
