#%%
import pandas as pd, numpy as np
from pyomo.environ import *

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
MODE = "price_maker"  # "price_taker" or "price_maker"

CAP_MW   = 100
CAP_MWh  = 200
ETA_RT   = 0.85
ETA_C    = ETA_D = np.sqrt(ETA_RT)
SOC0     = 0.80 * CAP_MWh
DELTA_T  = 0.5  # hours

PRICE_FILE = "./inputs/price_forecast.csv"
BETA_FILE  = "./inputs/beta_series_max.csv"

QUANT_WEIGHTS = {
    "P5": 0.05, "P10": 0.05, "P25": 0.15,
    "P50": 0.25, "P75": 0.25, "P90": 0.15, "P95": 0.10
}

# ───────────────────────────────────────────────────────────────
# LOAD PRICE AND ELASTICITY DATA
# ───────────────────────────────────────────────────────────────
df = pd.read_csv(PRICE_FILE)
df = df.pivot(index="HalfHourInterval", columns="Percentile", values="Value").reset_index()
df.rename(columns={"HalfHourInterval": "Interval_datetime"}, inplace=True)
df["Interval_datetime"] = pd.to_datetime(df["Interval_datetime"], format="%d/%m/%Y %H:%M")

df = df[(df["Interval_datetime"] >= "2025-06-26 04:05") & (df["Interval_datetime"] <= "2025-06-27 04:00")]
df.sort_values("Interval_datetime", inplace=True)

Pbar = sum(df[q] * w for q, w in QUANT_WEIGHTS.items()).to_numpy()
T = len(Pbar)
timestamps = pd.date_range("2025-06-26 04:30", periods=T, freq="30min")

# Load beta (price elasticity)
if MODE == "price_maker":
    beta_series = pd.read_csv(BETA_FILE, parse_dates=["DATETIME"]).set_index("DATETIME").squeeze()
    beta_vector = beta_series.reindex(timestamps, method="ffill").to_numpy()
else:
    beta_vector = np.zeros(T)

# ───────────────────────────────────────────────────────────────
# BUILD PYOMO MODEL
# ───────────────────────────────────────────────────────────────
m = ConcreteModel()
m.T = RangeSet(0, T - 1)

m.charge    = Var(m.T, bounds=(0, CAP_MW))
m.discharge = Var(m.T, bounds=(0, CAP_MW))
m.soc       = Var(m.T, bounds=(0, CAP_MWh))
m.mode      = Var(m.T, within=Binary)
m.z         = Var(m.T, bounds=(-CAP_MW, CAP_MW))
m.w         = Var(m.T)

m.Pbar = Param(m.T, initialize=lambda m, t: float(Pbar[t]))
m.beta = Param(m.T, initialize=lambda m, t: float(beta_vector[t]))
m.dt   = Param(initialize=DELTA_T)

# Constraints
m.z_def = Constraint(m.T, rule=lambda m, t: m.z[t] == m.discharge[t] - m.charge[t])
m.charge_limit = Constraint(m.T, rule=lambda m, t: m.charge[t]    <= CAP_MW * m.mode[t])
m.discharge_limit = Constraint(m.T, rule=lambda m, t: m.discharge[t] <= CAP_MW * (1 - m.mode[t]))

def soc_rule(m, t):
    if t == 0:
        return m.soc[t] == SOC0 + m.dt * (ETA_C * m.charge[t] - m.discharge[t] / ETA_D)
    return m.soc[t] == m.soc[t - 1] + m.dt * (ETA_C * m.charge[t] - m.discharge[t] / ETA_D)
m.soc_con = Constraint(m.T, rule=soc_rule)
m.final_soc = Constraint(expr=m.soc[T - 1] == 0.2 * CAP_MWh)

# Piecewise for z² if price_maker
if MODE == "price_maker":
    B = np.linspace(-CAP_MW, CAP_MW, 6)
    m.pw = Piecewise(
        m.T, m.w, m.z,
        pw_pts=B.tolist(),
        f_rule=[float(b**2) for b in B],
        pw_constr_type='EQ',
        pw_repn='BIGM_BIN'
    )
else:
    for t in m.T:
        m.w[t].fix(0)

# Objective: revenue minus price impact
EPS = 1e3
m.obj = Objective(
    expr = sum(m.dt * (m.Pbar[t] * m.z[t] + m.beta[t] * m.w[t]) for t in m.T)
         - EPS * sum(m.charge[t] + m.discharge[t] for t in m.T),
    sense = maximize
)

# ───────────────────────────────────────────────────────────────
# SOLVE
# ───────────────────────────────────────────────────────────────
solver = None
for sname in ["highs", "cbc", "glpk", "gurobi", "cplex"]:
    if SolverFactory(sname).available():
        solver = SolverFactory(sname)
        break
if solver is None:
    raise RuntimeError("No MILP solver available.")

solver.solve(m, tee=True)

# ───────────────────────────────────────────────────────────────
# OUTPUT
# ───────────────────────────────────────────────────────────────
schedule = pd.DataFrame({
    "DATETIME": timestamps,
    "price_base": Pbar,
    "beta": beta_vector,
    "charge_MW": [value(m.charge[t]) for t in m.T],
    "discharge_MW": [value(m.discharge[t]) for t in m.T],
    "SOC_MWh": [value(m.soc[t]) for t in m.T]
})
schedule["net_MW"] = schedule["discharge_MW"] - schedule["charge_MW"]
schedule["price_expected"] = schedule["price_base"] + schedule["beta"] * schedule["net_MW"]
schedule["revenue"] = schedule["net_MW"] * schedule["price_expected"] * DELTA_T

output_file = f"./outputs/schedule_{MODE}_beta_max.csv"
schedule.to_csv(output_file, index=False)

print(f"\n Optimal profit ({MODE}): ${value(m.obj):,.0f}")
print(f" Schedule saved to: {output_file}")

# %%
