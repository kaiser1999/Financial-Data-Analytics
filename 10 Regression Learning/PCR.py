from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.datasets import load_boston
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 10, 10
plt.rcParams.update({'font.size': 15})

X, y = load_boston(return_X_y=True)
# Split the data into training and testing dataset in ratio of 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=4012)

mu_train = np.mean(X_train, axis=0)
sigma_train = np.std(X_train, axis=0)
scale_X_train = (X_train - mu_train)/sigma_train
scale_X_test = (X_test - mu_train)/sigma_train

pca = PCA()
pca.fit(scale_X_train)

s2 = pca.explained_variance_   # save the variance of all PC to s2
PC_comp = np.arange(pca.n_components_) + 1
plt.plot(PC_comp, pca.explained_variance_ratio_*sum(s2), 
         "bo-", linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance')

plt.savefig("Boston PCA scree plot.png", dpi=400)

print(np.cumsum(pca.explained_variance_ratio_))

#%%
n_comp = 9 # for 95% explained variance
pcr_model = LinearRegression()
scores_X_train = pca.transform(scale_X_train)
pcr_model.fit(scores_X_train[:,:n_comp], y_train)

scores_X_test = pca.transform(scale_X_test)
y_hat_pcr = pcr_model.predict(scores_X_test[:,:n_comp])

#%%

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_hat_lr = lr_model.predict(X_test)

print(np.sqrt(np.mean((y_test - y_hat_pcr)**2)))	# RMSE
print(np.sqrt(np.mean((y_test - y_hat_lr)**2)))     # RMSE

#%%
from sklearn.preprocessing import StandardScaler

# threshold model for dummay variables chas
chas_train = X_train[:,3]
chas_test = X_test[:,3]
print(np.unique(chas_train, return_counts=True))

# remove dummy feature chas from training and testing dataset
X_train_chas = np.delete(X_train, 3, 1)
X_test_chas = np.delete(X_test, 3, 1)

X_train_0 = X_train_chas[chas_train==0,:]
X_train_1 = X_train_chas[chas_train==1,:]

X_test_0 = X_test_chas[chas_test==0,:]
X_test_1 = X_test_chas[chas_test==1,:]

scaler_0 = StandardScaler().fit(X_train_0)
scale_X_train_0 = scaler_0.transform(X_train_0)
scale_X_test_0 = scaler_0.transform(X_test_0)

scaler_1 = StandardScaler().fit(X_train_1)
scale_X_train_1 = scaler_1.transform(X_train_1)
scale_X_test_1 = scaler_1.transform(X_test_1)

pca_0 = PCA()
pca_0.fit(scale_X_train_0)

s2_0 = pca_0.explained_variance_   # save the variance of all PC to s2
PC_comp_0 = np.arange(pca_0.n_components_) + 1
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].plot(PC_comp_0, pca_0.explained_variance_ratio_*sum(s2_0), 
            "bo-", linewidth=2)
axs[0].set_title('Scree Plot')
axs[0].set_xlabel('Principal Component')
axs[0].set_ylabel('Variance')

pca_1 = PCA()
pca_1.fit(scale_X_train_1)

s2_1 = pca_1.explained_variance_   # save the variance of all PC to s2
PC_comp_1 = np.arange(pca_1.n_components_) + 1
axs[1].plot(PC_comp_1, pca_1.explained_variance_ratio_*sum(s2_1), 
            "bo-", linewidth=2)
axs[1].set_title('Scree Plot')
axs[1].set_xlabel('Principal Component')
axs[1].set_ylabel('Variance')

fig.savefig("Boston threshold PCA scree plot.png", dpi=400)

#%%
print(np.cumsum(pca_0.explained_variance_ratio_))
print(np.cumsum(pca_1.explained_variance_ratio_))
n_comp_0, n_comp_1 = 10 , 7
pcr_0_model = LinearRegression()
scores_X_train_0 = pca_0.transform(scale_X_train_0)
pcr_0_model.fit(scores_X_train_0[:,:n_comp_0], y_train[chas_train==0])

scores_X_test_0 = pca_0.transform(scale_X_test_0)
y_hat_pcr_0 = pcr_0_model.predict(scores_X_test_0[:,:n_comp_0])

pcr_1_model = LinearRegression()
scores_X_train_1 = pca_1.transform(scale_X_train_1)
pcr_1_model.fit(scores_X_train_1[:,:n_comp_1], y_train[chas_train==1])

scores_X_test_1 = pca_1.transform(scale_X_test_1)
y_hat_pcr_1 = pcr_1_model.predict(scores_X_test_1[:,:n_comp_1])

joint_SSE = np.concatenate(((y_test[chas_test==0] - y_hat_pcr_0)**2,
                           (y_test[chas_test==1] - y_hat_pcr_1)**2))

print(np.sqrt(np.mean(joint_SSE)))