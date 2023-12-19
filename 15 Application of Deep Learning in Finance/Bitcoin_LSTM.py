import pandas as pd
import numpy as np

#%%
df = pd.read_csv("../Datasets/Bitstamp_BTCUSD_2018_minute.csv", header=1)
df.index = df.date
df.drop(["unix", "date", "symbol", "Volume USD"], axis=1, inplace=True)
df = df.iloc[::-1]          # Reverse the order of dates

BTC_vol = df["Volume BTC"].values
df_diff = df.diff()
df_diff["Volume BTC"] = BTC_vol

# Select the last quarter as the training dataset
date_index = pd.to_datetime(df_diff.index)
mask_train = pd.Series(date_index).between("2018-10-01", "2018-12-31",
                                           inclusive="left")
df_train = df_diff.loc[mask_train.values]
y_close_train = df.close.loc[mask_train.values]

# Select the first day as the test dataset
mask_test = pd.Series(date_index).between("2018-12-31", "2019-01-01",
                                          inclusive="left")
df_test = df_diff.loc[mask_test.values]
y_close_test = df.close[mask_test.values]

print(df_train.index[0], df_train.index[-1], df_test.index[0])

def generate_dataset(df, seq_len):
    X_list, y_list = [], []
    for i in range(len(df.index) - seq_len):
        X_list.append(np.array(df.iloc[i:i+seq_len,:]))
        y_list.append(df.close.values[i+seq_len])
        
    return np.array(X_list), np.array(y_list)

#%%
LAG = 10
# Add LAG number of observations in training dataset to test dataset
df_test = pd.concat((df_train.iloc[-LAG:,:], df_test), axis=0)

X_train, y_train = generate_dataset(df_train, seq_len=LAG)
X_test, y_test = generate_dataset(df_test, seq_len=LAG)

print(np.mean(y_train))
print(np.mean((y_train - np.mean(y_train))**2))
print(np.mean((y_test - np.mean(y_train))**2))

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

tf.keras.utils.set_random_seed(4002)

model = Sequential()
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(32, dropout=0.4))

model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

LSTM_pred = np.squeeze(model.predict(X_test))
print(np.mean((LSTM_pred - y_test)**2))

#%%
import matplotlib.pyplot as plt

date_val = pd.to_datetime(y_close_test.index)
xticks = date_val.strftime('%H:%M')

LSTM_close = LSTM_pred + y_close_test
fig = plt.figure(figsize=(13,8))
plt.plot(y_close_test, color='blue', linewidth=2, label='Actual')
plt.plot(LSTM_close, color='pink', linestyle='dashed', 
         linewidth=2, label="LSTM")
skip_days = len(y_close_test)//10
plt.xticks(np.arange(0, len(y_close_test), skip_days), 
           xticks[::skip_days])
plt.title(f'''Bitcoin Price Prediction on 
          {date_val.date[0]}''', fontsize=20)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.legend()

plt.tight_layout()
plt.savefig("../Picture/Bitcoin LSTM.png", dpi=200)