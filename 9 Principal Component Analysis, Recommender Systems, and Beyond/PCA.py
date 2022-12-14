import numpy as np
import pandas as pd

d = pd.read_csv("us-rate.csv")  # Read in data

label = ["1m","3m","6m","9m","12m","18m","2y",
         "3y","4y","5y","7y","10y","15y"]
d.columns = label                   # Apply labels
np.set_printoptions(precision=2)    # Display the number of 2 digits
print(np.corrcoef(d, rowvar=False)) # Compute correlation matrix

#%%
from sklearn.decomposition import PCA

np.set_printoptions(precision=4)    # Display the number of 4 digits
pca = PCA()
standard_d = (d - np.mean(d, axis=0))/np.std(d, axis=0)
pca.fit(standard_d)
# components_: an ndarray of shape (n_components, n_features)
print(pca.components_[:6,:].T)

#%%
pc1 = pca.components_[0,:]    # Save the loading of 1st PC
pc2 = pca.components_[1,:]    # Save the loading of 2nd PC

print(pc1 @ pc1)
print(pc2 @ pc2)
print(pc1 @ pc2)

#%%

s2 = pca.explained_variance_   # save the variance of all PC to s2
print(np.round(s2, 4))         # Display the variances of all PC's
print(sum(s2))
# Proportion of variance explained by PC's
print(np.round(pca.explained_variance_ratio_, 4))
# Cumulative sum of proportion of variance
print(np.cumsum(pca.explained_variance_ratio_))

#%%
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 10, 10
plt.rcParams.update({'font.size': 15})

PC_comp = np.arange(pca.n_components_) + 1
plt.plot(PC_comp, pca.explained_variance_ratio_*sum(s2), 
         "bo-", linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance')

plt.savefig("PCA scree plot.png", dpi=400)

#%%
pc1 = pca.components_[0,:]    # Save the loading of 1st PC
pc2 = -pca.components_[1,:]   # Save the loading of 2nd PC
pc3 = -pca.components_[2,:]   # Save the loading of 3rd PC

# Multi-frame for plotting
fig, axs = plt.subplots(3, 1, figsize=(10,15))
axs[0].plot(PC_comp, pc1, "bo-")	# blue-circle-line
axs[0].set_ylim(-0.65, 0.65)
axs[0].set_ylabel("PC1")
axs[1].plot(PC_comp, pc2, "ro-")	# red-circle-line
axs[1].set_ylim(-0.65, 0.65)
axs[1].set_ylabel("PC2")
axs[2].plot(PC_comp, pc3, "co-")	# cyan-circle-line
axs[2].set_ylim(-0.65, 0.65)
axs[2].set_ylabel("PC3")

plt.savefig("PCA first 3 components.png", dpi=400)

#%%
from seaborn import pairplot

score = standard_d @ pca.components_.T   # save scores of all PC's
score.columns = [f"PC{i}" for i in PC_comp]
print(score.iloc[:,:3])
plots = pairplot(score.iloc[:,:3], diag_kind="hist", height=5)
plots.savefig("PCA first 3 scores pairplot.png", dpi=400)

#%%

def pr_comp(X, center=True, scale=True, method="SD"):
    feature_name = X.columns.values
    if center:
        X = X - np.mean(X, axis=0)
    if scale:
        X = X/np.std(X, axis=0)
    matr = np.cov(X, rowvar=False)
    if method=="SD":
        w, v = np.linalg.eig(matr)
        std, loadings = np.sqrt(w), v
    else:
        u, s, vh = np.linalg.svd(matr, full_matrices=True)
        std, loadings = np.sqrt(s), vh.T
    
    loadings = pd.DataFrame(loadings, index=feature_name)
    loadings.columns = [f"PC{i}" for i in range(1, np.shape(X)[1]+1)]
    score = X @ loadings
    # std is population standard deviation, different from R
    return {"loadings": loadings, "sdev":std, "scores":score}

pca2 = pr_comp(d, method="123")
print(np.allclose(pca2["sdev"], np.sqrt(s2)))

#%%

import pandas as pd
import numpy as np
	
d = pd.read_csv("us stocks.csv", index_col=0)
r = d.apply(np.log).diff().dropna()

spx = pd.read_csv("spx.csv", index_col=0)

from sklearn.decomposition import PCA
pca = PCA(1).fit(r)
pc1 = pd.Series(index=r.columns, data=pca.components_[0])
	
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = 13, 8
plt.rcParams.update({'font.size': 15})

# select 10 Equities to invest base on the loadings of PC1
n_stock = 10
investment = 10_000
short_r = 0.5
short_interest = 0.08
base_fee = 2.05

select_long = pc1.nlargest(n_stock).index
print(pc1.nlargest(n_stock))

select_short = pc1.nsmallest(n_stock).index
print(pc1.nsmallest(n_stock))

pca_long_n = np.floor(investment/d[select_long].iloc[0,:])
pca_long_remain = (investment*n_stock 
                   - d[select_long].iloc[0,:] @ pca_long_n)
pca_long_com = sum(pca_long_n.apply(lambda x: max(base_fee, 0.013*x)))
pca_long_stock = d[select_long].iloc[1:,:] @ pca_long_n
pca_long_val = pca_long_stock - pca_long_com + pca_long_remain

pca_short_n = np.floor(investment*short_r/d[select_short].iloc[0,:])
pca_short_remain = d[select_short].iloc[0,:] @ pca_short_n
pca_short_com = sum(pca_short_n.apply(lambda x: max(base_fee, 0.013*x)))
pca_short_stock = d[select_short].iloc[1:,:] @ pca_short_n
date = pd.to_datetime(d.index.values, format="%d/%m/%Y")
short_days = (date[1:] - date[0]).days
pca_short_interest = short_interest * short_days/365 * pca_short_remain
pca_short_val = (investment*n_stock + pca_short_remain - pca_short_stock 
                 - pca_short_com - pca_short_interest)

market_n = np.floor(investment*n_stock/spx.iloc[0].values)
market_remain = investment*n_stock - spx.iloc[0]*market_n
market_com = max(base_fee, 0.013*market_n)
market_stock = market_n*spx.iloc[1:]
market_val = market_stock + market_remain - market_com

plt.rcParams["figure.figsize"] = 15, 10

compare = pd.concat([pca_long_val, pca_short_val, market_val], axis=1)
compare.plot(linewidth=2, color=["blue", "red", "black"])
plt.legend(["PCA long", "PCA short", "S&P500"])
plt.ylim(6.5*investment, 14.5*investment)

plt.savefig("PCA stock selection.png", dpi=400)