import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from KMeansCluster import get_bcss, best_km

iris = datasets.load_iris(return_X_y=False)
X_iris, y_iris = iris["data"], iris["target"]
df_iris = pd.DataFrame(np.c_[X_iris, y_iris], 
                       columns=iris["feature_names"] + ["Species"])
df_iris["Species"].replace(dict(zip([0,1,2], iris["target_names"])), inplace=True)
fig = plt.figure(figsize=(10, 10))              # Plot observations with color
sns.pairplot(df_iris, hue="Species", diag_kind=None,
             palette=dict(zip(iris["target_names"], ["black", "red", "green"])))

plt.savefig("../Picture/iris_cluster.png", dpi=200)

np.random.seed(4002)
km_iris = KMeans(n_clusters=3, n_init="auto")   # K-means clustering with K=3
km_iris.fit(X_iris)

print(km_iris.cluster_centers_)                 # Mean of each cluster
print(km_iris.labels_)                          # Predicted cluster labels
print(np.bincount(km_iris.labels_))             # The size of each cluster
print("Between group sum of squares:", get_bcss(km_iris, X_iris))
print("Within group sum of squares:", km_iris.inertia_)

#%%
print(pd.crosstab(km_iris.labels_, y_iris))     # Classification table

#%%
# HSI dataset
df_HSI = pd.read_csv("../Datasets/fin-ratio.csv")
X_HSI, y_HSI = df_HSI.iloc[:, :-1].values, df_HSI.HSI
fig = plt.figure(figsize=(15, 15))              # Plot observations with color
ax = sns.pairplot(df_HSI, hue="HSI", diag_kind=None, 
                  palette=dict(zip([0, 1], ["black", "red"])))

plt.savefig("../Picture/HSI_cluster.png", dpi=200)

np.random.seed(4002)
km_HSI = KMeans(n_clusters=2, n_init="auto")    # K-means clustering with K=2
km_HSI.fit(X_HSI)    

print(pd.crosstab(km_HSI.labels_, y_HSI))       # Classification table

#%%
X1 = X_iris[km_iris.labels_ == 0]   # select group by cluster label
X2 = X_iris[km_iris.labels_ == 1]
X3 = X_iris[km_iris.labels_ == 2]
print("Cluster sizes:", len(X1), len(X2), len(X3))

# Cluster means
mean_X1 = np.mean(X1, axis=0)
mean_X2 = np.mean(X2, axis=0)
mean_X3 = np.mean(X3, axis=0)
print("Cluster 1 mean:", mean_X1)
print("Cluster 2 mean:", mean_X2)
print("Cluster 3 mean:", mean_X3)

# Within group sum of squares
wcss_X1 = np.sum((X1 - mean_X1)**2)
wcss_X2 = np.sum((X2 - mean_X2)**2)
wcss_X3 = np.sum((X3 - mean_X3)**2)
print("Within group sum of squares for Cluster 1:", wcss_X1)
print("Within group sum of squares for Cluster 2:", wcss_X2)
print("Within group sum of squares for Cluster 3:", wcss_X3)
print(wcss_X1 + wcss_X2 + wcss_X3)

#%%
_, km_iris2 = best_km(X_iris, 2)                # try K=2
_, km_iris3 = best_km(X_iris, 3)                # try K=3
_, km_iris4 = best_km(X_iris, 4)                # try K=4
_, km_iris5 = best_km(X_iris, 5)                # try K=5

#%%
print(pd.crosstab(km_iris3.labels_, y_iris))    # classification table

#%%
# Cleaned HSI
df_cHSI = pd.read_csv("../Datasets/fin-ratio_cleansed.csv") # read in cleansed HSI dataset
X_cHSI, y_cHSI = df_cHSI.iloc[:, :-1].values, df_cHSI.HSI
_, km_cHSI2 = best_km(X_cHSI, 2)                # try K=2
_, km_cHSI3 = best_km(X_cHSI, 3)                # try K=3
_, km_cHSI4 = best_km(X_cHSI, 4)                # try K=4
_, km_cHSI5 = best_km(X_cHSI, 5)                # try K=5

#%%
print(pd.crosstab(km_cHSI2.labels_, y_cHSI))    # classification table

df_cHSI_km2 = df_cHSI.copy()
df_cHSI_km2.HSI = km_cHSI2.labels_
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# boxplots for each variable
for i in range(len(df_cHSI_km2.columns) - 1):
    j, k = i // 3, i % 3
    sns.boxplot(data=df_cHSI_km2, ax=axs[j,k],
                x="HSI", y=df_cHSI_km2.columns[i])

plt.tight_layout()
plt.savefig("../Picture/HSI_boxplots.png", dpi=200)

#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from KMeansCluster import best_km

df_Cred = pd.read_csv('../Datasets/Credit Card Customer Data.csv')
df_Cred = df_Cred.drop(['Sl_No', 'Customer Key'], axis=1)
X = df_Cred.values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

_ , km_cCred2 = best_km(X, 2) # try K = 2
_ , km_cCred3 = best_km(X, 3) # try K = 3
_ , km_cCred4 = best_km(X, 4) # try K = 4
_ , km_cCred5 = best_km(X, 5) # try K = 4

#%%
plt_cmap = sns.color_palette("viridis", n_colors=3)
centroids = km_cCred3.cluster_centers_

fig, ax = plt.subplots(figsize=(10, 7))
# scatter plot w.r.t first two feature variables
for i in range(len(np.unique(km_cCred3.labels_))):
    idx = np.where(km_cCred3.labels_ == i)[0]
    ax.scatter(X[idx,0], X[idx,1], color=plt_cmap[i], s=50, 
               label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='orange', 
            s=150)
plt.legend(fontsize=15)
plt.xlabel(df_Cred.columns[0], fontsize=15)
plt.ylabel(df_Cred.columns[1], fontsize=15)
plt.title('Results of K-means Clustering', fontsize=20)

plt.tight_layout()
plt.savefig("../Picture/Credit_card_Kmeans.png", dpi=200)

#%%
df_Cred_km3 = df_Cred.copy()
df_Cred_km3["Total_interactions"] = (df_Cred["Total_visits_bank"] + 
                                     df_Cred["Total_visits_online"] + 
                                     df_Cred["Total_calls_made"])
df_Cred_km3["clust"] = km_cCred3.predict(X) + 1
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# boxplots for each variable
for i in range(len(df_Cred_km3.columns) - 1):
    j, k = i // 3, i % 3
    sns.boxplot(data=df_Cred_km3, ax=axs[j,k], palette="viridis", 
                x="clust", y=df_Cred_km3.columns[i])
    
plt.tight_layout()
plt.savefig("../Picture/Credit_card_HSI_boxplots.png", dpi=200)