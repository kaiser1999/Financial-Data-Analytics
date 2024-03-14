import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox

def find_mu(returns, tail=5):
    mu = returns.rolling(tail + 1 + tail).mean()
    return mu.shift(-tail)

def find_sigma(returns, tail=15):
    sig = returns.rolling(tail + 1 + tail).std(ddof=1)
    return sig.shift(-tail)

#INDICES = ["^DJI", "^FTSE", "^HSI"]
#NAMES = ["Dow Jones", "FTSE 100", "HSI"]
#year_start = [1992, 1984, 1987]

INDICES = ["^SPX", "^FTSE", "^HSI"]
NAMES = ["S&P 500", "FTSE 100", "HSI"]
year_start = [1985, 1984, 1987]

width = 0.6
xtick_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
ytick_labels = ["0%", "20%", "40%", "60%", "80%", "100%"]

def plot_monthly_index(daily_return, start, end, index_name, m=65):
    date_start = dt.datetime(start, 1, 1)
    date_end = dt.datetime(end, 12, 31)
    date = daily_return.index
    mask = (date >= date_start) * (date <= date_end)
    title = f"Goodness index for {index_name} from {start} to {end}"
    
    mu = find_mu(daily_return, tail=m)
    sigma = find_sigma(daily_return, tail=m)
    mu, sigma = mu[mask], sigma[mask]

    stand_return = (daily_return - mu)/sigma
    date = stand_return.index
    lb_pvalue = np.empty(end - start + 1)
    for i, year in enumerate(range(start, end+1)):
        mask = (date >= dt.datetime(year, 1, 1)) * (date <= dt.datetime(year, 12, 31))
        lb_pvalue[i] = acorr_ljungbox(stand_return[mask], 
                                      lags=[10]).lb_pvalue.values[0]
    
    '''
    goodness_index = mu/sigma - 0.5*sigma
    monthly_goodness = {mon: None for mon in range(1, 13)}
    date = goodness_index.index
    for mon in range(1, 13):
        lamb = goodness_index[date.month == mon]
        total = np.sum(lamb < 0) + np.sum(lamb > 0)
        monthly_goodness[mon] = {"neg": np.sum(lamb < 0)/total, 
                                 "pos":np.sum(lamb > 0)/total}
    
    montly_pos = [monthly_goodness[mon]["pos"] for mon in range(1, 13)]
    monthly_neg = [monthly_goodness[mon]["neg"] for mon in range(1, 13)]

    xtick = np.arange(12)
    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(xtick+width, montly_pos, width=width, color="green", 
                    label="positive")
    rects2 = ax.bar(xtick+width, monthly_neg, width=width, color="red", 
                    label="negative", bottom=montly_pos)
    
    ax.set_xticks(xtick+width)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(xtick_labels, fontsize=20)
    ax.set_yticklabels(ytick_labels, fontsize=20)
    ax.set_xlabel("Month", fontsize=20)
    ax.yaxis.grid(color="b")
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), 
              prop={"size": 20}, ncol=2)
    fig.subplots_adjust(bottom=0.2)
    
    for (rect1, rect2) in zip(rects1, rects2):
        xtick = rect1.get_x()
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax.text(xtick + width/2, 0.5*height1, f"{height1*100:.2f}%",
                rotation=30, ha="center", va="center", fontsize=15)
        ax.text(xtick + width/2, 0.5*height2+height1, f"{height2*100:.2f}%",
                rotation=30, ha="center", va="center", fontsize=15)
    
    fig.tight_layout()
    fig.savefig(f"../Picture/Calendar effect_{index_name}.png", dpi=200)
    
    '''
    return lb_pvalue
    
#%%
start, end = 2000, 2015
m_lst = [65, 70, 65]
m_lst = [70, 65, 70]

df_lb = pd.DataFrame(index=range(start, end+1), columns=INDICES)
for index, index_name, m in zip(INDICES, NAMES, m_lst):
    historical_data = yf.Ticker(index).history(period="max").tz_localize(None)
    daily_return = pd.Series(np.diff(np.log(historical_data.Close)), 
                             index=historical_data.index[1:])
    
    df_lb[index] = plot_monthly_index(daily_return, start, end, index_name, m=m)

print(df_lb)

#%%
m_lst = [55, 60, 65, 70, 75]
idx = 2
index = INDICES[idx]
index_name = NAMES[idx]

historical_data = yf.Ticker(index).history(period="max").tz_localize(None)
daily_return = pd.Series(np.diff(np.log(historical_data.Close)), 
                         index=historical_data.index[1:])

df_lb = pd.DataFrame(index=range(start, end+1), columns=[f"{index}_{m}" for m in m_lst])
for m in m_lst:
    ind = f"{index}_{m}"
    df_lb[ind] = plot_monthly_index(daily_return, start, end, index_name, m=m)

print(df_lb)