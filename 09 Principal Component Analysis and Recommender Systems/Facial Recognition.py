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
    path = os.path.join("ATT", sub_folder)
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

train_img, test_img = np.array(train_img), np.array(test_img)
mu_train = np.mean(train_img, axis=0)
std_train = np.std(train_img, axis=0)
rescale_train = (train_img - mu_train)/std_train
rescale_test = (test_img - mu_train)/std_train

#%%
# method 1: using numpy
lamb, U = np.linalg.eig(rescale_train @ rescale_train.T / len(rescale_train))
H = rescale_train.T @ U
H = H / np.sqrt(np.sum(H**2, axis=0))   # normalize each eigenvector
print(H.shape)                  # a (hw) x n matrix
print(sum(lamb))

CumSum_s2 = np.cumsum(lamb)/sum(lamb)
idx = np.where(CumSum_s2 > 0.9)[0][0]
tilde_H = H[:, :idx]
print(tilde_H.shape)            # a (hw) x m matrix

#%% 
# method 2: using sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9).fit(rescale_train)
# components_: an ndarray of shape (n_components, n_features)
tilde_H = pca.components_.T
print(tilde_H.shape)            # a (hw) x m matrix

#%%
import matplotlib.pyplot as plt

score_train = rescale_train @ tilde_H
score_test = rescale_test @ tilde_H
ncol = 10
fig, axes = plt.subplots(nrows=8, ncols=ncol, figsize=(6*ncol,8*8))
for i in trange(len(score_test)):
    distance = np.linalg.norm(score_train - score_test[i], axis=1)
    idx_train = np.argmin(distance)

    j = 2*i
    ax1 = axes[j // ncol][j % ncol]
    ax1.imshow(test_img[i].reshape(112, 92), cmap='gray')
    ax1.axis('off')
    ax1.set_title("Test image", fontsize=45)
    ax2 = axes[(j+1) // ncol][(j+1) % ncol]
    ax2.imshow(train_img[idx_train].reshape(112, 92), 
               cmap='gray')
    ax2.axis('off')
    ax2.set_title(f"Distance {distance[idx_train]:.3f}", fontsize=45)
    
plt.tight_layout()
fig.savefig("../Picture/PCA_Facial.png")