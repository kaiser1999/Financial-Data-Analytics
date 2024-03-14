import numpy as np
import pandas as pd

d = pd.read_csv("../Datasets/us-rate.csv")  # Read in data

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

plt.savefig("../Picture/PCA scree plot.png", dpi=400)

#%%
pc1 = pca.components_[0,:]    # Save the loading of 1st PC
pc2 = -pca.components_[1,:]   # Save the loading of 2nd PC
pc3 = -pca.components_[2,:]   # Save the loading of 3rd PC

# Multi-frame for plotting
fig, axs = plt.subplots(3, 1, figsize=(10,17))
axs[0].plot(PC_comp, pc1, "bo-")	# blue-circle-line
axs[0].set_ylim(-0.65, 0.65)
axs[0].set_ylabel("PC1")
axs[1].plot(PC_comp, pc2, "ro-")	# red-circle-line
axs[1].set_ylim(-0.65, 0.65)
axs[1].set_ylabel("PC2")
axs[2].plot(PC_comp, pc3, "co-")	# cyan-circle-line
axs[2].set_ylim(-0.65, 0.65)
axs[2].set_ylabel("PC3")

plt.tight_layout()
plt.savefig("../Picture/PCA first 3 components.png", dpi=400)

#%%
from seaborn import pairplot

score = standard_d @ pca.components_.T   # save scores of all PC's
score.columns = [f"PC{i}" for i in PC_comp]
print(score.iloc[:,:3])
plots = pairplot(score.iloc[:,:3], diag_kind="hist", height=5)
plots.savefig("../Picture/PCA first 3 scores pairplot.png", dpi=400)

#%%
# compute the covariance matrix and extract the variance
var = np.diag(np.cov(d, ddof=1, rowvar=False))
print(np.sqrt(var))

#%%
import pandas as pd
import numpy as np
	
df = pd.read_csv("../Datasets/us_Open_2020H.csv", index_col=0)
r = df.apply(np.log).diff()
r_mask_train = pd.to_datetime(r.index, format='%d/%m/%Y') < "2020-04-01"
r_train = r.loc[r_mask_train].dropna()

from sklearn.decomposition import PCA
# PCA: Input data is centered but not scaled for each feature
pca = PCA(n_components=1).fit(r_train)
pc1 = pd.Series(index=r.columns, data=pca.components_[0])

# Select 10 equities to invest base on the loadings of pc1
n_stock = 10
smallest_n = pc1.nsmallest(n_stock).index
largest_n = pc1.nlargest(n_stock).index

pd.set_option('display.max_columns', None)
df_train_last = sum(r_mask_train) - 1 # python starts idx from 0
print(df[smallest_n].iloc[[0, df_train_last],:].diff())
print(df[largest_n].iloc[[0, df_train_last],:].diff())

#%%
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = 15, 10
plt.rcParams.update({'font.size': 15})

investment = 100_000
def buy_and_hold(test_price, n_stock=n_stock):
    n_units = np.floor(investment/n_stock/test_price.iloc[0,:])
    commission = sum(n_units.apply(lambda x: max(2.05, 0.013*x)))
    stocks_amount = test_price.iloc[0,:] @ n_units
    price_path = test_price.iloc[1:,:] @ n_units
    remain = investment - 2*commission
    return remain + (price_path - stocks_amount)
    
spx = pd.read_csv("../Datasets/SPX_Open_2020H.csv", index_col=0)
equity_price = df.loc[np.invert(r_mask_train)]
index_price = spx.loc[np.invert(r_mask_train)]

pca_bnh_smallest = buy_and_hold(equity_price[smallest_n], n_stock)
pca_bnh_largest = buy_and_hold(equity_price[largest_n], n_stock)
market_bnh = buy_and_hold(index_price, 1)

compare = pd.concat([pca_bnh_smallest, pca_bnh_largest, 
                     market_bnh], axis=1)
compare.plot(linewidth=2, color=["blue", "red", "black"])
plt.legend(["PCA smallest", "PCA largest", "S&P500"])
plt.ylim(0.9*investment, 3*investment)

plt.tight_layout()
plt.savefig("../Picture/PCA stock selection.png", dpi=400)