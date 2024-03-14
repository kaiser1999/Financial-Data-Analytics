import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=2, max_iter=1000, 
                     solver="lbfgs", activation="logistic", 
                     random_state=1999)
model.fit(X, y)

W1, W2 = model.coefs_
b1, b2 = model.intercepts_
W1, W2 = W1.T, W2.T
print(W1)
print(W2)
print(b1)
print(b2)

#%%
logistic = lambda x: 1/(1 + np.exp(-x))

X_ = X[[0,50,100]]
a1 = logistic(W1 @ X_.T + b1.reshape(-1, 1))
a2 = W2 @ a1 + b2.reshape(-1, 1)
pr = np.argmax(a2, axis=0) # logistic / np.round(a2[0]) linear

print(a1)
print(a2)

#%%
import pandas as pd

y_pred = model.predict(X)
y_pred = np.round(y_pred)

print(pd.crosstab(y_pred, y))   # confusion matrix

#%%
import pandas as pd

df = pd.read_csv("../Datasets/fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]
print(X.columns.values)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=3, max_iter=1000, 
                     solver="lbfgs", activation="logistic", 
                     random_state=4002)

model.fit(X, y)
y_pred = model.predict(X)
y_pred = y_pred > 0.5
print(pd.crosstab(y_pred, y))   # confusion matrix

#%%
from ANNet import ANNet
import pandas as pd
import numpy as np
np.random.seed(4002)

df = pd.read_csv("../Datasets/fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]

model = ANNet(X, y, size=3, linout=True, max_iter=1000, trial=10)
y_pred = model.predict(X)
y_pred = y_pred > 0.5

print(pd.crosstab(y_pred, y))		 		# confusion matrix
W1, W2 = model.coefs_
b1, b2 = model.intercepts_
W1, W2 = W1.T, W2.T
print(W1)
print(W2)
print(b1)
print(b2)

#%%
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
model = ANNet(X, y, size=2, linout=False, max_iter=1000, trial=10)
y_pred = model.predict(X)

print(pd.crosstab(y_pred, y))		 		# confusion matrix
W1, W2 = model.coefs_
b1, b2 = model.intercepts_
W1, W2 = W1.T, W2.T
print(W1)
print(W2)
print(b1)
print(b2)

#%%
y_proba = model.predict_proba(X)
y_pred = np.argmax(y_proba, axis=1)
print(y_pred)

#%%
df = pd.read_csv("../Datasets/fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]

model = ANNet(X, y, size=3, linout=False, max_iter=1000, trial=10)
y_pred = model.predict(X)

print(pd.crosstab(y_pred, y))		 		# confusion matrix
W1, W2 = model.coefs_
b1, b2 = model.intercepts_
W1, W2 = W1.T, W2.T
print(W1)
print(W2)
print(b1)
print(b2)

#%%
import numpy as np
logistic = lambda x: 1/(1 + np.exp(-x))

X = np.array([[0.4,0.7], [0.8, 0.9], [1.3, 1.8], [-1.3, -0.9]])
y = np.array([0, 0, 1, 0])              # target value

# hidden layer bias and weights
W1 = np.array([[0.1,-0.2,0.1], [0.4,0.2,0.9]])
# output layer bias and weights
W2 = np.array([[0.2,-0.5,0.1]])

# transpose to fit the input format of ANN
X1 = np.c_[np.ones(len(y)), X]
h = logistic(W1 @ X1.T)                 # logistic hidden h'
h = np.c_[np.ones(len(y)), h.T].T
o = W2 @ h								# linear output o'
err = y - o
print(err)                              # output error
print(np.mean(err**2))					# mean SSE

#%%
lr = 0.5                                # learning rate: $\eta$
n = len(y)
del2 = -2*err                           # output layer $\delta_2$
Delta_W2 = -lr*del2 @ h.T               # $\Delta W2 = -\eta \delta_2 (h')^T$
new_W2 = W2 + Delta_W2 / n              # new output weights: $W2 = W2 + \Delta W2$

del1 = (W2.T @ del2)*h*(1-h)            # hidden layer $\delta_1$
del1 = del1[1:,]                        # remove from X1
Delta_W1 = -lr*del1 @ X1                # $\Delta W1 = -\eta \delta_1 x^T$
new_W1 = W1 + Delta_W1 / n              # new hidden weights: $W1 = W1 + \Delta W1$

new_h = logistic(new_W1 @ X1.T)
new_h = np.c_[np.ones(len(y)), new_h.T].T
new_o = new_W2 @ new_h
new_err = y - new_o
print(np.mean(new_err**2))              # new mean SSE