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