import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix

#%%
df = pd.read_csv("fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]
ctree = DecisionTreeClassifier(ccp_alpha=0.01)
ctree.fit(X, y)
print(export_text(ctree, feature_names=list(X.columns), 
                  show_weights=True))

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
plot_tree(ctree, feature_names=X.columns, filled=True)
fig.tight_layout()
fig.savefig("../Picture/Classification tree fin-ratio.png", dpi=200)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
scatter = ax.scatter(y, X.ln_MV, c=y, alpha=0.7,
                     cmap=ListedColormap(["blue", "red"]))
ax.legend(handles=scatter.legend_elements()[0], loc="upper left", 
          labels=["HSI", "Non-HSI"], fontsize=15)
ax.axhline(y=9.478)
ax.set_xlabel("HSI", fontsize=15)
ax.set_ylabel("ln MV", fontsize=15)
fig.tight_layout()
fig.savefig("../Picture/ln MV vs HSI.png", dpi=400)

#%%
y_hat = ctree.predict(X)
print(confusion_matrix(y_hat, y))

#%%
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris["data"], iris['target']
ctree = DecisionTreeClassifier(ccp_alpha=0.01, random_state=4012)
ctree.fit(X, y)

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
plot_tree(ctree, feature_names=iris['feature_names'], filled=True)
fig.tight_layout()
fig.savefig("../Picture/Classification tree Iris flower.png", dpi=200)

print(export_text(ctree, feature_names=iris['feature_names'], 
                  show_weights=True))

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
scatter = ax.scatter(X[:,2], X[:,3], c=y, alpha=0.7,
                     cmap=ListedColormap(["red", "blue", "green"]))
ax.legend(handles=scatter.legend_elements()[0], loc="upper left", 
          labels=list(iris['target_names']), fontsize=15)
plt.axhline(y=1.75)
plt.axvline(x=2.45)
plt.axhline(y=1.65)
plt.axvline(x=4.95)
ax.set_xlabel("Petal length", fontsize=15)
ax.set_ylabel("Petal width", fontsize=15)
fig.tight_layout()
fig.savefig("../Picture/Petal width vs Petal length.png", dpi=400)

#%%
y_hat = ctree.predict(X)
print(confusion_matrix(y_hat, y))

#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
import numpy as np

price = fetch_california_housing()
X, y = price["data"], price['target']
rtree = DecisionTreeRegressor(ccp_alpha=0.01, random_state=4012, max_depth=3)
rtree.fit(X, y)

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
plot_tree(rtree, feature_names=price['feature_names'], filled=True)

y_hat = rtree.predict(X)
print(np.mean((y - y_hat)**2))


#%%
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]
rf_clf = RandomForestClassifier(max_features=2, random_state=4002)
rf_clf.fit(X, y)
y_hat = rf_clf.predict(X)
print(confusion_matrix(y_hat, y))

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("credit default.csv")
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

(X_train, X_test, y_train, 
 y_test) = train_test_split(X, y, train_size=0.8, random_state=4012)
ctree = DecisionTreeClassifier(ccp_alpha=0.01, random_state=4012)
ctree.fit(X_train, y_train)
y_hat_dt = ctree.predict(X_test)
print(confusion_matrix(y_hat_dt, y_test))
print(classification_report(y_test, y_hat_dt))

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
plot_tree(ctree, feature_names=X.columns, filled=True)
fig.tight_layout()
fig.savefig("../Picture/Credit Default DT.png", dpi=200)

rf_clf = RandomForestClassifier(random_state=4012)
rf_clf.fit(X_train, y_train)
y_hat_rf = rf_clf.predict(X_test)
print(confusion_matrix(y_hat_rf, y_test))
print(classification_report(y_test, y_hat_rf))