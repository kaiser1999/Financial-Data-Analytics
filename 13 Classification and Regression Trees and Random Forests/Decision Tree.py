import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix

#%%
df = pd.read_csv("../Datasets/fin-ratio.csv")
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
# Regression Tree

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

# Prepare the dataset
df = pd.read_csv("../Datasets/Medicalpremium.csv")
X = df[['Age', 'Weight']]
y = df['PremiumPrice']

# Fit and plot the regression tree model
rtree = DecisionTreeRegressor(ccp_alpha=0.01, max_depth=3)
rtree.fit(X, y)
print(export_text(rtree, feature_names=list(X.columns), 
                  show_weights=True))

plt.figure(figsize=(12, 8))
plot_tree(rtree, feature_names=list(X.columns), filled=True)

plt.tight_layout()
plt.savefig("../Picture/Medical Insurance RTree.png", dpi=200)

# Plot dataset with segments and text annotations
fig, ax = plt.subplots(figsize=(8, 6))
s = ax.scatter(df['Age'], df['Weight'], c=y, cmap='rainbow')

# Add vertical lines and text annotations
ax.axvline(x=29.5, color='black', linewidth=2)
ax.axvline(x=46.5, color='black', linewidth=2)
ax.axvline(x=38.5, color='black', linewidth=2)
lines = [[(0, 119), (29.5, 119)], 
         [(24.5, 0), (24.5, 119)], [(23.0, 119), (23.0, 140)],
         [(46.5, 94.5), (70, 94.5)]]
lc = mc.LineCollection(lines, colors="black", linewidths=2)
ax.add_collection(lc)
plt.text(18.5, 127, "R1", fontsize=12)
plt.text(25.5, 127, "R2", fontsize=12)
plt.text(19.5, 85, "R3", fontsize=12)
plt.text(26, 85, "R4", fontsize=12)
plt.text(33, 90, "R5", fontsize=12)
plt.text(41.5, 90, "R6", fontsize=12)
plt.text(56, 115, "R7", fontsize=12)
plt.text(56, 71, "R8", fontsize=12)

plt.xlabel('Age')
plt.ylabel('Weight')
fig.colorbar(s, label='Premium Price')

fig.tight_layout()
fig.savefig("../Picture/Medical Insurance Scatter.png", dpi=200)

#%%
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../Datasets/fin-ratio.csv")
X = df.drop(columns="HSI")
y = df["HSI"]
rf_clf = RandomForestClassifier(max_features=2, random_state=4002)
rf_clf.fit(X, y)
y_hat = rf_clf.predict(X)
print(confusion_matrix(y_hat, y))

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("../Datasets/credit default.csv")
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