import pandas as pd
import numpy as np

#%%
Year = 2018
LAG = 20

df = pd.read_csv(f"Bitstamp_BTCUSD_{Year}_minute.csv", header=1)
df.index = df.date
df.drop(["unix", "date", "symbol", "Volume USD"], axis=1, inplace=True)
df = df.iloc[::-1]          # Reverse the order of dates

BTC_vol = df["Volume BTC"].values
df_diff = df.diff()
df_diff["Volume BTC"] = BTC_vol

# Select the last quarter as the training dataset
date_index = pd.to_datetime(df_diff.index)
mask_train = pd.Series(date_index).between(f"{Year}-09-30", 
                                           f"{Year}-12-30")
df_train = df_diff.loc[mask_train.values]
y_close_train = df.close.loc[mask_train.values]

# Select the first day as the test dataset
mask_test = pd.Series(date_index).between(f"{Year}-12-30", 
                                          f"{Year}-12-31")
df_test = df_diff.loc[mask_test.values]
# Add LAG number of observations in training dataset to test dataset
df_test = pd.concat((df_train.iloc[-LAG:,:], df_test), axis=0)
y_close_test = df.close[mask_test.values]

#%%
def generate_dataset(df, seq_len):
    X_list, y_list = [], []
    for i in range(len(df.index) - seq_len):
        X_list.append(np.array(df.iloc[i:i+seq_len,:]))
        y_list.append(df.close.values[i+seq_len])
        
    return np.array(X_list), np.array(y_list)

X_train, y_train = generate_dataset(df_train, seq_len=LAG)
X_test, y_test = generate_dataset(df_test, seq_len=LAG)

print(np.mean(y_train))
print(np.mean((y_train - np.mean(y_train))**2))
print(np.mean((y_test - np.mean(y_train))**2))

#%%
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(4010)

model = Sequential()
model.add(LSTM(32, return_sequences=True, dropout=0))
model.add(LSTM(16, dropout=0))

model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

LSTM_pred = np.squeeze(model.predict(X_test))
print(np.mean((LSTM_pred - y_test)**2))

#%%
import matplotlib.pyplot as plt

nminute = 500
date_val = df_test.index[LAG:][:nminute]
xticks = pd.to_datetime(date_val).strftime('%H:%M')
plt.rcParams['font.size'] = "20"

LSTM_close = LSTM_pred + y_close_test
fig = plt.figure(figsize=(15,10))
plt.plot(LSTM_close[:nminute], color="blue", linestyle='dashed', 
         label="LSTM", linewidth=4)
plt.plot(y_close_test[:nminute], color='orange', label='Actual')
plt.xticks(date_val[:nminute:int(nminute/10)], 
           xticks[:nminute:int(nminute/10)])
plt.title(f'''Bitcoin Price Prediction from {date_val[0]} 
          to {date_val[-1]}''', fontsize=25)
plt.ylim(3690, 3770)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig("../Picture/Bitcoin LSTM.png", dpi=200)