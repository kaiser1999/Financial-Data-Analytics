from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=2, max_iter=1000, solver="lbfgs",
                     activation="logistic", random_state=4012, alpha=0.0001)
model.fit(X, y)
y_pred = model.predict(X)
y_pred = np.round(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
print(pd.crosstab(y, y_pred))# confusion matrix

print(model.loss_)
L2 = 0.0001*(np.sum(model.coefs_[0]**2) + np.sum(model.coefs_[1]**2))
print((np.sum((model.predict(X) - y)**2) + L2) / 2 / len(y))
print(model.coefs_)
print(model.intercepts_)

#%%
import pandas as pd
import numpy as np

df = pd.read_csv("../Datasets/fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=2, max_iter=1000, solver="lbfgs",
                     activation="logistic", random_state=4012)
# let's fit the training data to our model
model.fit(X, y)
y_pred = model.predict(X)
y_pred = np.round(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
print(pd.crosstab(y, y_pred))# confusion matrix

print(model.coefs_)
print(model.intercepts_)

#%%
from ANNet import ANNet
import pandas as pd
import numpy as np
np.random.seed(4012)

df = pd.read_csv("../Datasets/fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]

model = ANNet(X, y, size=2, linout=True, max_iter=1000, trial=10)
y_pred = model.predict(X)
y_pred = np.round(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
print(pd.crosstab(y, y_pred))		 		# confusion matrix
print(model.coefs_)							# get weights
print(model.intercepts_)					# get bias

#%%
from ANNet import ANNet
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
np.random.seed(4012)

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4012)

model = ANNet(X_train, y_train, size=2, linout=False, max_iter=1000, trial=10)
y_pred = model.predict(X_train)

print(model.loss_)				# Best categorical cross-entropy loss + L2
enc = OneHotEncoder(handle_unknown='ignore')
one_hot_y = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
L2 = 0.0001*(np.sum(model.coefs_[0]**2) + np.sum(model.coefs_[1]**2))
print((np.sum(-one_hot_y * np.log(model.predict_proba(X_train))) + L2/2) / len(y_train))
print(pd.crosstab(y_train, y_pred))			# confusion matrix
print(model.coefs_)							# get weights $\bm{W}^{(1)}, \bm{W}^{(2)}$
print(model.intercepts_)					# get bias $\bm{b}^{(1)}, \bm{b}^{(2)}$

W1, W2 = model.coefs_
b1, b2 = model.intercepts_
W1, W2 = W1.T, W2.T
print(W1)
print(W2)
print(b1)
print(b2)

logistic = lambda x: 1/(1 + np.exp(-x))

a1 = logistic(W1 @ X_test.T + b1.reshape(-1, 1))
a2 = W2 @ a1 + b2.reshape(-1, 1)
pr = np.argmax(a2, axis=0) # logistic / np.round(a2[0]) linear
print(pd.crosstab(y_test, pr))# confusion matrix

y_pred = model.predict(X_test)
#y_pred = np.round(y_pred) # linear

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print(pd.crosstab(y_test, y_pred))# confusion matrix

#%%
from ANNet import ANNet
import pandas as pd
import numpy as np
np.random.seed(560)
np.random.seed(12966)

df_train = pd.read_csv("../Datasets/fin-ratio_train.csv", index_col=0)
df_test = pd.read_csv("../Datasets/fin-ratio_test.csv", index_col=0)
X_train, X_test = df_train.drop(columns="HSI"), df_test.drop(columns="HSI")
y_train, y_test = df_train["HSI"], df_test["HSI"]

model = ANNet(X_train, y_train, size=2, linout=False, max_iter=1000, trial=20)

print(model.coefs_)
print(model.intercepts_)

# training dataset
y_pred = model.predict(X_train)
y_pred = np.round(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred))
print(pd.crosstab(y_train, y_pred))# confusion matrix

# testing dataset
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

print(classification_report(y_test, y_pred))
print(pd.crosstab(y_test, y_pred))# confusion matrix