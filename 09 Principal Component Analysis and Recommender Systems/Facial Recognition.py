# Operating System to link with the folder in which the images are saved
import os
import numpy as np
import cv2              # OpenCV = OPEN source for Computer Vision library
# tqdm = taqadum, meaning the processing passed for each iteration [===>...]
from tqdm import tqdm, trange

SUB_FOLDERS = [f"s{i+1}" for i in range(40)]
train_img, test_img = [], []
for sub_folder in SUB_FOLDERS:
	# join() = JOIN the text strings by slash "/"
    path = os.path.join("../Datasets/ATT", sub_folder)
    # For all image filenames in the current working directory
    for i, img_Name in enumerate(tqdm(os.listdir(path))):
        img_Path = os.path.join(path, img_Name)
        img_array = cv2.imread(img_Path, cv2.IMREAD_COLOR)
        gray_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        rescale_array = gray_array.flatten()
        # take the first image in every sub_folder as test
        if i == 0:
            test_img.append(rescale_array)
        else:
            train_img.append(rescale_array)

train_img, test_img = np.array(train_img).T, np.array(test_img).T
mu_train = np.mean(train_img, axis=1).reshape(-1, 1)
std_train = np.std(train_img, axis=1).reshape(-1, 1)
rescale_train = (train_img - mu_train)/std_train
rescale_test = (test_img - mu_train)/std_train

#%%
# method 1: using numpy
_, n = rescale_train.shape
lamb, U = np.linalg.eig(rescale_train.T @ rescale_train / n)
H = U.T @ rescale_train.T
# normalize each eigenvector
H = H / np.sqrt(np.sum(H**2, axis=1).reshape(-1, 1))
print(H.shape)                  # a n x (hw) matrix
print(sum(lamb))

CumSum_s2 = np.cumsum(lamb)/sum(lamb)
idx = np.where(CumSum_s2 > 0.9)[0][0]
tilde_H = H[:idx, :]
print(tilde_H.shape)            # m x (hw) matrix

#%% 
# method 2: using sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9).fit(rescale_train.T)
# components_: an ndarray of shape (n_components, n_features)
tilde_H = pca.components_
print(tilde_H.shape)            # m x (hw) matrix

#%%
import matplotlib.pyplot as plt

score_train = tilde_H @ rescale_train   # m x n matrix
score_test = tilde_H @ rescale_test     # m x n matrix
m, n_test = score_test.shape
'''
ncol = 10
fig, axes = plt.subplots(nrows=8, ncols=ncol, figsize=(6*ncol,8*8))
pred_dist = []
for i in trange(n_test):
    distance = np.linalg.norm(score_train - 
                              score_test[:,i].reshape(-1, 1), axis=0)
    idx_train = np.argmin(distance)

    j = 2*i
    ax1 = axes[j // ncol][j % ncol]
    ax1.imshow(test_img[:,i].reshape(112, 92), cmap='gray')
    ax1.axis('off')
    ax1.set_title("Test image", fontsize=45)
    ax2 = axes[(j+1) // ncol][(j+1) % ncol]
    ax2.imshow(train_img[:,idx_train].reshape(112, 92), 
               cmap='gray')
    ax2.axis('off')
    ax2.set_title(f"Distance {distance[idx_train]:.3f}", fontsize=45)
    pred_dist.append(distance[idx_train])
    
plt.tight_layout()
fig.savefig("../Picture/PCA_Facial.png")

#%%
col = ["blue"] * n_test
col[34] = "red"
fig = plt.figure(figsize=(10, 7))
plt.bar(np.arange(n_test), pred_dist, ec="black", color=col, alpha=0.7)
plt.xlabel("Index", fontsize=15)
plt.ylabel("Distance", fontsize=15)
plt.axhline(65, ls="--", color="black")

plt.tight_layout()
fig.savefig("../Picture/PCA_Facial bar.png", dpi=200)
'''

#%%
y_train = np.array([np.repeat(x, 9) for x in range(40)]).reshape(-1)
K = 3

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=K, metric="cosine", 
                             weights="distance")
model.fit(score_train.T, y_train)
distances, indices = model.kneighbors(score_test.T)
indices = model.predict(score_test.T)
print(indices)
proba = model.predict_proba(score_test.T)
print(proba[34])                # 34 is wrong

#%%

y_hat, y_weight = [], []
norm_train = np.linalg.norm(score_train, axis=0)
norm_test  = np.linalg.norm(score_test, axis=0)
for i in range(len(score_test.T)):
    similarity = (score_train.T @ score_test[:,i] / 
                  (norm_train * norm_test[i]))
    idx = np.argpartition(similarity, -K)        # cloest to theta=0
    P_K = similarity[idx[-K:]]
    weight = np.zeros(len(np.unique(y_train)))
    for m in range(len(np.unique(y_train))):
        weight[m] = np.sum(1 / (1 - P_K[y_train[idx[-K:]] == m]))
    
    idx_train = np.argmax(weight)
    y_hat.append(idx_train)
    y_weight.append(weight)

print(y_hat)
print(y_weight[34] / np.sum(y_weight[34]))       # 34 is wrong

#%%
'''
y_train = np.array([np.repeat(x, 9) for x in range(40)]).reshape(-1)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)
model.fit(score_train.T, y_train)
distances, indices = model.kneighbors(score_test.T)
print(indices.reshape(-1) // 9)     # reshape to array for 1-NN
print(distances[34])                # 34 is a mismatch / 35-th individual

#%%
K = 3
y_hat, y_weight = [], []
norm_train = np.linalg.norm(score_train, axis=0)
norm_test  = np.linalg.norm(score_test, axis=0)
for i in range(len(score_test.T)):
    similarity = (score_train.T @ score_test[:,i] / 
                  (norm_train * norm_test[i]))
    idx = np.argpartition(similarity, -K) # cloest to theta=1
    P_K = similarity[idx[-K:]]
    weight = np.zeros(len(np.unique(y_train)))
    for m in range(len(np.unique(y_train))):
        weight[m] = np.sum(P_K[y_train[idx[-K:]] == m])
        
    idx_train = np.argmax(weight)
    y_hat.append(idx_train)
    y_weight.append(weight)

print(y_hat)
print(y_weight[34])
'''