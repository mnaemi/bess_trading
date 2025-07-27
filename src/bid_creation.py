
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------- tunables ----------------------------
EPS        = 0.05      # $ undercut vs competitor 5%
SMALL      = 0.05     #  threshold
MAX_LOOP   = 5
PRICE_FLOOR, PRICE_CAP = -1000.0, 17500.0
# -----------------------------------------------------------------

def _force_monotone(arr):
    out = arr.copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i-1] + 1.0)
    return np.clip(out, PRICE_FLOOR, PRICE_CAP)

def _initial_ladder(comp_row, direction):
    bands = comp_row.filter(regex=r"PRICEBAND\d+").astype(float).to_numpy()
    bands[0] = PRICE_FLOOR
    bands[-1] = PRICE_CAP
    if direction == "GEN":
        bands[1:-1] *= (1-EPS)
    else:
        bands[1:-1] *= (1+EPS)
    return _force_monotone(bands)

def _frequency_above(forecast_slice,exp_price_col, threshold):
    """fraction of p50 points above threshold"""
    return (forecast_slice[exp_price_col] > threshold).mean()

def optimise_direction(direction, schedule, forecast, exp_price_col = 'expected_price', comp_row='GEN'):
    # slice schedule & forecast to relevant intervals
    if direction == "GEN":
        mask = schedule["discharge_MW"] > 0
        qty_col = "discharge_MW"
    else:
        mask = schedule["charge_MW"] > 0
        qty_col = "charge_MW"

    sched = schedule[mask].copy()
    sched = sched.merge(forecast, on="DATETIME", how="left")

    # start ladder from competitor
    bands = _initial_ladder(comp_row, direction)

    for _ in range(MAX_LOOP):
        change = False

        # --------------- step through each interval ----------------
        for _, row in sched.iterrows():
            exp_price = row[exp_price_col]
            mw        = row[qty_col]

            # find nearest higher band
            dist = np.abs(bands - exp_price)
            idx = np.nanargmin(dist)
            target = exp_price

            # decide if tweak needed
            if abs(target - bands[idx])/abs(bands[idx]) <= SMALL:
                if target < bands[idx]:
                    bands[idx] = target
                    change = True
            else:
                # far change → choose high‑ or low‑band to overwrite
                freq_above = _frequency_above(sched, exp_price_col,target)
                if freq_above < 0.2:         # high prices rare → kill high band
                    high_idx = np.argmax(bands[:-1])
                    bands[high_idx] = target
                else:                        # low prices rare → kill low band
                    bands[idx] = target
                change = True

        bands.sort()
        if not change:
            break

    return bands

# -------------------- MAIN ENTRYPOINT ----------------------------
def build_price_bands(schedule_df, forecast_df, competitor_df, price_col='expected_price'):
    comp_gen = competitor_df[competitor_df.DIRECTION == "GEN"].iloc[0]
    comp_load = competitor_df[competitor_df.DIRECTION == "LOAD"].iloc[0]

    gen_bands = optimise_direction("GEN", schedule_df, forecast_df, price_col,comp_gen)
    load_bands = optimise_direction("LOAD", schedule_df, forecast_df,price_col, comp_load)

    return gen_bands, load_bands

def prepare_price_forecast(df):

    forecast_df = (
        df.pivot(index="HalfHourInterval",
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
    return forecast_df

def visualize_price_bands(price_bands_df):
    # Reset index to get DIRECTION as a column
    df_plot = price_bands_df.reset_index().rename(columns={'index': 'DIRECTION'})


    # Assuming df is already defined
    df_gen = df_plot[df_plot["DIRECTION"] == "GEN"].set_index("Price")
    df_load = df_plot[df_plot["DIRECTION"] == "LOAD"].set_index("Price")

    # Select only price band columns
    band_cols = [col for col in df_plot.columns if "PRICEBAND" in col]

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    sns.heatmap(df_gen[band_cols], annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("GEN Price Bands")
    axes[0].set_xlabel("Price Band")
    axes[0].set_ylabel("Scenario (Price)")

    sns.heatmap(df_load[band_cols], annot=True, fmt=".0f", cmap="Blues", ax=axes[1])
    axes[1].set_title("LOAD Price Bands")
    axes[1].set_xlabel("Price Band")
    axes[1].set_ylabel("")

    plt.suptitle("Price Band Values Across Scenarios", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

schedule_df = (
    pd.read_csv('./outputs/schedule_price_response_linear_beta_avg.csv', parse_dates=["DATETIME"])
      .rename(columns={"price_expected": "expected_price"})   # if your column name differs
      [["DATETIME", "expected_price", "discharge_MW", "charge_MW"]]
)


forecast_df = pd.read_csv('./inputs/price_forecast.csv', parse_dates=["HalfHourInterval"])
forecast_df = prepare_price_forecast(forecast_df)
competitor_df = pd.read_csv('./inputs/Competitor_Price_Bands.csv')

# keep only the essential columns
pb_cols = [f"PRICEBAND{i}" for i in range(1, 11)]
competitor_df = competitor_df[["DIRECTION"] + pb_cols]

price_cols = ["P5", "P10", "P25", "P50", "P75", "P90", "P95"] + ["expected_price"]
price_bands_list = []
for p in price_cols:
    gen_b, load_b = build_price_bands(schedule_df, forecast_df, competitor_df, p)
    gen_b[-1] = PRICE_CAP
    price_bands = pd.DataFrame({
        "GEN": gen_b,
        "LOAD": load_b,
    }, index=[f"PRICEBAND{i}" for i in range(1, 11)])
    price_bands = price_bands.T
    price_bands['Price'] = p
    price_bands_list.append(price_bands)

price_bands_df = pd.concat(price_bands_list)
price_bands_df.to_csv("./outputs/price_bands_beta_max.csv")

visualize_price_bands(price_bands_df)


# %%

# ------------------------------------------------------------------
# USER‑TUNEABLE WEIGHTS
# ------------------------------------------------------------------
LOWER_WT_FULL    = (0.30, 0.70, 0.0)  # full‑discharge: 60‑30‑10 (target‑1 | target | next higher)
HIGHER_WT_FULL   = (0.0, 0.60, 0.40)  # full‑charge   : target‑1 | target | target+1
NAMEPLATE_MW     = schedule_df[["discharge_MW","charge_MW"]].to_numpy().max()
DEFAULT_BIDS = {'GEN': {"P9":0.5, "P10":0.5},
                'LOAD': {"P1":1.0}}

def band_index(price, bands, direction = 'GEN'):
    """largest index i such that bands[i] <= price"""
    if direction == 'GEN':
        return max(1, np.searchsorted(bands, price, side="right") - 1)
    else:  # LOAD
        return max(1, np.searchsorted(bands, price, side="right"))


def alloc_interval(row, bands, direction):
    """
    Returns length‑10 array of MW that **sums to NAMEPLATE_MW**.
    direction = 'GEN' | 'LOAD'
    """
    vol = np.zeros(10)
    target_mw = row.discharge_MW if direction == "GEN" else row.charge_MW
    residual  = NAMEPLATE_MW - target_mw

    # ---------------- active volume ----------------
    if target_mw > 0:
        idx = band_index(row.expected_price, bands, direction)

        if direction == "GEN":
            b_low, b_t, b_hi = idx-1, idx, idx+1
            for j, wt in zip((b_low, b_t, b_hi), LOWER_WT_FULL):
                vol[max(1, min(j, 9))] += target_mw * wt

        else:  # LOAD
            b_low, b_t, b_hi = idx-1, idx, idx+1
            for j, wt in zip((b_low, b_t, b_hi), HIGHER_WT_FULL):
                vol[max(1, min(j, 9))] += target_mw * wt

    # no active volume → fill with default bids
    if direction == "GEN":
        for band, wt in DEFAULT_BIDS['GEN'].items():
            vol[int(band[1:]) - 1] += residual * wt
    else:
        for band, wt in DEFAULT_BIDS['LOAD'].items():
            vol[int(band[1:]) - 1] += residual * wt

    # final tidy‑up
    vol = np.round(vol, 2)
    assert abs(vol.sum() - NAMEPLATE_MW) < 1e-6, "Volume not balanced"
    return vol

# ------------------------------------------------------------------
# BUILD BID TABLE PER INTERVAL
# ------------------------------------------------------------------
band_cols = [f"BANDAVAIL{i}" for i in range(1, 11)]

records = []
for _, row in schedule_df.iterrows():
    rec_gen  = {"DATETIME": row.DATETIME, "DIRECTION": "GEN"}
    rec_load = {"DATETIME": row.DATETIME, "DIRECTION": "LOAD"}

    rec_gen.update( dict(zip(band_cols, alloc_interval(row, gen_b , "GEN"))) )
    rec_load.update(dict(zip(band_cols, alloc_interval(row, load_b, "LOAD"))) )

    records.extend([rec_gen, rec_load])

bid_volume_df = pd.DataFrame(records)


def plot_stacked_area(bid_volume_df, direction):
    df = bid_volume_df[bid_volume_df["DIRECTION"] == direction].copy()
    df = df.sort_values("DATETIME")
    band_cols = [f"BANDAVAIL{i}" for i in range(1, 11)]
    plt.figure(figsize=(14, 6))
    plt.stackplot(df["DATETIME"], [df[col] for col in band_cols], labels=band_cols)
    plt.legend(loc="upper right", ncol=2)
    plt.title(f"Stacked Area Chart for {direction} Bands")
    plt.xlabel("DATETIME")
    plt.ylabel("MW")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_stacked_area(bid_volume_df, "GEN")
plot_stacked_area(bid_volume_df, "LOAD")
# ------------------------------------------------------------------
# QUICK LOOK
# ------------------------------------------------------------------
print(bid_volume_df.head(6))

# %%
