import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

#%%
df = pd.read_csv("../Datasets/Bitstamp_BTCUSD_2018_minute.csv", header=1)
df = df.iloc[::-1]          # Reverse the order of dates
df.index = df.date

# Select the last quarter as the training dataset
date_index = pd.to_datetime(df.index)
mask_train = pd.Series(date_index).between("2018-10-01", "2018-12-31",
                                           inclusive="left")
train_close = df.close.loc[mask_train.values]

# Select the first day as the test dataset
mask_test = pd.Series(date_index).between("2018-12-31", "2019-01-01",
                                          inclusive="left")
test_close = df.close[mask_test.values]

print(train_close.index[0], train_close.index[-1], test_close.index[0])

alpha = 0.05
c_i = stats.norm.ppf(1-alpha/2) / np.sqrt(len(train_close))
fig, ax = plt.subplots(figsize=(10,10))
plot_acf(train_close.values, lags=30, ax=ax, alpha=None, 
         color="blue", title="ACF Plot of Bitcoin Price")
ax.set_ylim((-0.15, 1.1))
ax.axhline(c_i, linestyle="--", c="purple")
ax.axhline(-c_i, linestyle="--", c="purple")

fig.tight_layout()
fig.savefig("../Picture/ACF plots of Bitcoin prices.png", dpi=200)

lag_price = train_close.diff().values[1:]
fig, axs = plt.subplots(ncols=2, figsize=(20,10))
plot_acf(lag_price, lags=30, ax=axs[0], alpha=None, 
         color="blue", title="ACF Plot of 1-lagged Bitcoin Price")
axs[0].set_ylim((-0.15, 1.1))
axs[0].axhline(c_i, linestyle="--", c="purple")
axs[0].axhline(-c_i, linestyle="--", c="purple")

plot_pacf(lag_price, lags=30, ax=axs[1], alpha=None, zero=False,
          color="blue", title="PACF Plot of 1-lagged Bitcoin Price")
axs[1].set_ylim((-0.055, 0.055))
axs[1].axhline(c_i, linestyle="--", c="purple")
axs[1].axhline(-c_i, linestyle="--", c="purple")

#%%
# 1 lag difference; ACF and PACF: 2 significant lags and looks similar
print(ARIMA(train_close, order=(2, 1, 0)).fit().aic)
print(ARIMA(train_close, order=(0, 1, 2)).fit().aic)

fig.tight_layout()
fig.savefig("../Picture/ACF plots of lagged Bitcoin prices.png", dpi=200)

#%%
import pmdarima as pm

best_model = pm.auto_arima(train_close, max_p=5, max_q=5, max_d=2, 
                           information_criterion="aic", seasonal=False,
                           stepwise=False, with_intercept=False)
print(best_model.summary())

#%%
p, d, q = best_model.order
arima_name = f"ARIMA({p}, {d}, {q})"

resid = best_model.resid()[1:]          # remove the first residual
# Ljung-Box test on (non-standardized) residuals
print(acorr_ljungbox(resid, lags=[10], model_df=p+q))

fig = plt.figure(figsize=(13,8))
plt.plot(resid/np.sqrt(best_model.params().sigma2), "b-")
skip_minutes = len(train_close)//6
plt.xticks(np.arange(0, len(train_close), skip_minutes), 
           train_close.index[::skip_minutes])
plt.title(f"Standardized residuals of {arima_name}", fontsize=20)
plt.ylabel("")

print(np.where(abs(resid)/np.sqrt(best_model.params().sigma2) > 12)[0])
# Python indexing starts from 0 and we removed one row from resid
split_idx = 115698
plt.axvline(split_idx, c="orange", linewidth=2, linestyle="--")
# +1 for removing first row in resid and -1 for d=1
print(train_close.index[split_idx+1-1])

plt.tight_layout()
fig.savefig("../Picture/Bitcoin Standardized Residuals.png", dpi=200)

#%%
split_train = train_close[(split_idx+1-1):]
best_split = pm.auto_arima(split_train, max_p=10, max_q=10, max_d=2, 
                           information_criterion="aic", seasonal=False,
                           stepwise=False, with_intercept=False)
print(best_split.summary())
s_p, s_d, s_q = best_split.order
split_arima = f"ARIMA({s_p}, {s_d}, {s_q})"

split_resid = best_split.resid()[1:]     # remove the first residual
fig = plt.figure(figsize=(13,8))
plt.plot(split_resid/np.sqrt(best_split.params().sigma2), "b-")
skip_minutes = len(split_train)//6
plt.xticks(np.arange(0, len(split_train), skip_minutes), 
           split_train.index[::skip_minutes])
plt.title(f"Standardized residuals of {arima_name}", fontsize=20)
plt.ylabel("")
# Ljung-Box test on (non-standardized) residuals
print(acorr_ljungbox(split_resid, lags=[10], model_df=s_p+s_q))

plt.tight_layout()
fig.savefig("../Picture/Bitcoin Standardized Residuals Split.png", dpi=200)

#%%
hist_data = train_close[-30:].tolist()
arima_pred = np.empty(test_close.shape)
for i, x_t in enumerate(test_close):
    model = ARIMA(hist_data, order=(s_p, s_d, s_q))
    model_fit = model.smooth(best_split.params())
    arima_pred[i] = model_fit.forecast()[0]
    hist_data.append(x_t)

date_val = pd.to_datetime(test_close.index)
xticks = date_val.strftime('%H:%M')

fig = plt.figure(figsize=(13,8))
plt.plot(test_close, color='red', label='Actual')
plt.plot(arima_pred, color='blue', linestyle='dashed', label=split_arima)
skip_minutes = len(test_close)//6
plt.xticks(np.arange(0, len(test_close), skip_minutes), 
           xticks[::skip_minutes])
plt.title('Bitcoin Price Prediction on ' + 
          f'{date_val.date[0]}', fontsize=20)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.legend()

price = np.append(train_close[-1], test_close.values)
print(np.mean(np.diff(price)**2))
print(np.mean((test_close.values - arima_pred)**2))

plt.tight_layout()
fig.savefig("../Picture/Bitcoin ARIMA.png", dpi=500)