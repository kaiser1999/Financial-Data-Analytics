import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def find_mu(returns, window=5):
    mu = returns.rolling(window + 1 + window).mean()
    return mu.shift(-window)

def find_sigma(returns, window=15):
    sig = returns.rolling(window + 1 + window).std()
    return sig.shift(-window)

INDICES = ['^DJI', '^FTSE', '^HSI']
NAMES = ['Dow Jones Index', 'FTSE 100 Index', 'Hang Seng Index']
year_start = [1992, 1984, 1987]
date_start = dt.datetime(max(year_start)-1, 1, 1)
date_end = dt.datetime(2021+1, 12, 31)
#date_end = dt.datetime(2012, 7, 31)

pick = 2

index = INDICES[pick]
index_name = NAMES[pick]
historical_data = yf.Ticker(index).history(period="max").tz_localize(None)
daily_return = historical_data.Close.pct_change()

mu = find_mu(daily_return, window=1)
sigma = find_sigma(daily_return, window=15)

date = historical_data.index
mask = (date >= date_start) * (date <= date_end)
mu = mu[mask]
sigma = sigma[mask]

#%%

goodness_index = mu/sigma - 0.5 * sigma
Shiryaev_Zhou_index = mu/sigma**2 - 0.5

monthly_goodness = {mon: None for mon in range(1, 13)}
monthly_Shiryaev_Zhou = {mon: None for mon in range(1, 13)}
date = goodness_index.index
for mon in range(1, 13):
    lamb = goodness_index.loc[date.month == mon]
    total = np.sum(lamb < 0) + np.sum(lamb > 0)
    monthly_goodness[mon] = {'neg': np.sum(lamb < 0)/total, 'pos':np.sum(lamb > 0)/total}
    
    lamb = Shiryaev_Zhou_index[date.month == mon]
    total = np.sum(lamb < 0) + np.sum(lamb > 0)
    monthly_Shiryaev_Zhou[mon] = {'neg': np.sum(lamb < 0)/total, 'pos':np.sum(lamb > 0)/total}

montly_pos = [monthly_goodness[mon]['pos'] for mon in range(1, 13)]
monthly_neg = [monthly_goodness[mon]['neg'] for mon in range(1, 13)]

width = 0.6
xtick = np.arange(12)
xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
ytick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']

fig, ax = plt.subplots(figsize=(15, 10))
rects1 = ax.bar(xtick+width, montly_pos, width=width, color='green', label='positive')
rects2 = ax.bar(xtick+width, monthly_neg, width=width, color='red', label='negative', bottom=montly_pos)

ax.set_xticks(xtick+width)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticklabels(xtick_labels, fontsize=20)
ax.set_yticklabels(ytick_labels, fontsize=20)
ax.set_xlabel("Month", fontsize=20)
ax.yaxis.grid(color='b')
ax.set_axisbelow(True)
#ax.set_title(f'Goodness index for {index_name} from {date_start.date()} to {date_end.date()}', fontsize=25)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), prop={'size': 20}, ncol=2)
fig.subplots_adjust(bottom=0.2)

for (rect1, rect2) in zip(rects1, rects2):
    xtick = rect1.get_x()
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    ax.text(xtick + width/2, 0.5*height1, f'{height1*100:.2f}%',
            rotation=30, ha='center', va='center', fontsize=15)
    ax.text(xtick + width/2, 0.5*height2+height1, f'{height2*100:.2f}%',
            rotation=30, ha='center', va='center', fontsize=15)

fig.savefig(f"Calendar effect_{index_name}.png", dpi=200)
