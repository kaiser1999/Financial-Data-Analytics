import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_dataset(price, seq_len):
    X_list, y_list = [], []
    for i in range(len(price) - seq_len):
        X = np.array(price[i:i+seq_len])
        y = np.array([price[i+seq_len]])
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list), np.array(y_list)

# self: a syntex referring to the class object itself, i.e. MLP_stock
class MLP_stock:
    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        model.compile(optimizer=optimizer, loss="mse")
        return model
    
    def train(self, X_train, y_train, bs=32, ntry=5, n_epochs=50):
        self.best_model = self.build_model()
        self.best_model.fit(X_train, y_train, batch_size=bs, 
                            epochs=n_epochs, shuffle=True)
        eval_data = (X_train[-50:], y_train[-50:])
        best_loss = self.best_model.evaluate(*eval_data)
        for i in range(1, ntry):
            model = self.build_model()
            model.fit(X_train, y_train, batch_size=bs, 
                      epochs=n_epochs, shuffle=True)
            if model.evaluate(*eval_data) < best_loss:
                self.best_model = model
                best_loss = model.evaluate(*eval_data)
    
    def predict(self, X_test):
        return self.best_model.predict(X_test)

#%%
tf.random.set_seed(4002)
STOCKS = ["AAL", "GS", "FB", "MS"]
train_len = 900  # 900 trading days approximately 3.5 of calender years
seq_len = 10
for stock in STOCKS:
    df = pd.read_csv(f"{stock}.csv")
    stock_train = df["Adj Close"].iloc[:train_len].values
    stock_test = df["Adj Close"].iloc[train_len:].values
    
    X_train, y_train = generate_dataset(stock_train, seq_len)
    X_test, y_test = generate_dataset(stock_test, seq_len)
    
    MLP = MLP_stock()
    MLP.train(X_train, y_train)
    y_pred = np.squeeze(MLP.predict(X_test))
    
    plt.figure(figsize=(15, 10))
    # setting font size to be 20 for all the text in the plot
    plt.rcParams['font.size'] = "20"
    test_date = df.Date[-len(y_test):]
    plt.plot(test_date, y_test, label="true")
    plt.plot(test_date, y_pred, label="predict")
    plt.xticks(test_date[::50])
    
    plt.title(f"{stock} prediction from {df.Date[train_len]}", 
              fontsize=35)
    plt.ylabel("price", fontsize=25)
    plt.xlabel("trading days", fontsize=25)
    plt.legend(loc='lower right', fontsize=25)
    plt.tight_layout()
    plt.savefig("Prediction of " + stock+".png", dpi=200)