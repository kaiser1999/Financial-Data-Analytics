import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

P = np.array([[2, 1], [1, 6]])
q = np.array([7, 3])
G = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
h = np.array([1, ]*4)

def QPex(x_1, x_2):
    # x: A 3-D tensor of shape 2 x len(x_1) x len(x_2)
    x = np.stack([x_1, x_2], axis=0)
    return np.sum(1/2 * x.T @ P * x.T, axis=-1) + x.T @ q

x1 = np.linspace(-5, 1, 1000)
x2 = np.linspace(-3, 3, 1000)
X1, X2 = np.meshgrid(x1, x2)
Z = QPex(X1, X2)

plt.figure(figsize=(10, 10))
# Need transpose for Z
fig = plt.contour(X1, X2, Z.T, 8, colors="green", linewidths=2, 
                  linestyles="dashed")
plt.xlim(-5.1, 1.1)
plt.clabel(fig, fontsize=15, inline=1)

# solve.QP: minimize (1/2 x^T P x + q^T x) with the constraints (G x <= h)
constr_ans = solve_qp(P, q, G, h, solver="ecos", sym_proj=True)
real_ans = solve_qp(P, q, solver="ecos", sym_proj=True)

plt.plot(*constr_ans, "r*", markersize=15)
plt.plot(*real_ans, "r.", markersize=15)
# Need transpose for Z
constr_fig = plt.contour(X1, X2, Z.T, levels=[QPex(*constr_ans)], 
                         colors="purple")
plt.clabel(constr_fig, fontsize=15, inline=1)
plt.fill([-1,1,1,-1], [-1,-1,1,1], "gray", alpha=0.9)
plt.xlabel(r"$x^{(1)}$", fontsize=20); plt.ylabel(r"$x^{(2)}$", fontsize=20)

plt.tight_layout()
plt.savefig("../Graphs/Python Quadratic Program Example 1.png", dpi=200)

#%%
'''
# Get hourly adjusted open prices from yahoo finance
# Requested range of hourly prices must be within the last 730 days!
import pandas as pd
import datetime as dt
import yfinance as yf

YEAR = 2023
df = pd.read_csv("../Datasets/S&P500 list 2023.csv")
date_start = dt.datetime(YEAR, 1, 1, 1)   # 1-Jan, 2021 01:00:00
date_end = dt.datetime(YEAR+1, 2, 1, 1)   # 2-Feb, 2022 01:00:00

comp = df["Symbol"].str.replace(".", "-").to_list()
# Requested range of hourly prices must be within the last 730 days!
df_comp = yf.download(comp, period="23mo", interval="60m", auto_adjust=False,
                      start=date_start, end=date_end)

df_price = df_comp.Open
df_price.index = pd.to_datetime(df_price.index).tz_localize(None)
nan_row_mask = df_price.isnull().sum(axis=1) > 100     # remove one row
print(df_comp.index[nan_row_mask])
df_price = df_price.loc[~nan_row_mask,:]

nan_col_mask = df_price.isnull().sum(axis=0) > 0
print(df_price.columns[nan_col_mask])
df_price = df_price.loc[:,~nan_col_mask]
df_price.to_csv(f"../Datasets/S&P500_Open_hourly_{YEAR}.csv")

df_index = yf.download("^SPX", period="23mo", interval="60m",
                       start=date_start, end=date_end)
df_index = df_index.Open
df_index.index = pd.to_datetime(df_index.index.values)
df_index.to_csv(f"../Datasets/SPX_Open_hourly_{YEAR}.csv")
'''
# S&P500_Open_hourly_2021: Unadjusted opening prices of 375 securities
# S&P500_Open_hourly_2023: Unadjusted opening prices of 491 securities

#%%
import pandas as pd
import numpy as np
import datetime as dt

year = 2023
df_price = pd.read_csv(f"../Datasets/S&P500_Open_hourly_{year}.csv", 
                       index_col=0)
df_price.index = pd.to_datetime(df_price.index.values)
train_end = dt.datetime(year=year+1, month=1, day=1, hour=1)
test_end = train_end + dt.timedelta(days=7)
print(train_end, test_end)  # 1-Jan, 2022 01:00:00; 8-Jan, 2022 01:00:00
mask_train = (df_price.index < train_end)
mask_test = (train_end < df_price.index) * (df_price.index <= test_end)
df_train, df_test = df_price.loc[mask_train], df_price.loc[mask_test]
n_stocks = len(df_price.columns)

#%%
from qpsolvers import solve_qp

PRINCIPAL = 100_000         # Initial Investment of 100,000 USD
mu_annual = 0.15            # 15% minimum desired portfolio annual return 
mu_P = mu_annual / 7 / 256  # 7 hours per day and 256 trading days per year
print(mu_P)                 # minimum desired portfolio hourly return

r = np.log(1 + df_train.pct_change()).iloc[1:,:]
mu = r.ewm(alpha=0.94, adjust=False).mean().iloc[-1,:]
P = r.cov(ddof=1)           # Sample covariance matrix
q = np.zeros(n_stocks)
G = np.vstack([-mu, -np.eye(n_stocks)])
h = np.append([-mu_P], np.zeros(n_stocks))
A = np.ones(n_stocks)
b = np.array([1]).astype(np.float32)

w = solve_qp(P, q, G, h, A, b, solver="ecos")

w[w < 1e-2] = 0             # Treat weights less than 0.01 as 0
w = w/np.sum(w)             # Recale the weights
n_portfolio = PRINCIPAL * w / df_test.iloc[0,:].values
Portfolio_value = df_test.values @ n_portfolio
Markowitz_portfolio = df_test.columns.values[w > 0]
print(dict(zip(Markowitz_portfolio, np.round(w[w > 0], 5))))

#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

df_index = pd.read_csv(f"../Datasets/SPX_Open_hourly_{year}.csv", index_col=0)
df_test_index = df_index.loc[mask_test]
n_index = PRINCIPAL / df_test_index.iloc[0].values
Index_value = df_test_index.values.reshape(-1) * n_index

fig, ax = plt.subplots(figsize=(15, 10))
line1, = ax.plot(df_test.index, Portfolio_value, label="MPT")
line2, = ax.plot(df_test.index, Index_value, label="S&P 500")
ax.legend(loc="upper right", handles=[line1, line2], prop={"size": 20})
plt.xlabel("Time", fontsize=18)
plt.ylabel("Price", fontsize=18)
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

print(Portfolio_value[-1] - PRINCIPAL, Index_value[-1] - PRINCIPAL)

fig.tight_layout()
fig.savefig(f"../Picture/Optimal_Portfolio_Value_{year}.png", dpi=200)

#%%
print(df_test_index.values)