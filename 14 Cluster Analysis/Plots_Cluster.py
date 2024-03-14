import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=3)
kmeans = KMeans(n_clusters=3, n_init="auto")
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#%%
def k_means_clustering(X_train, n_clusters):
    centers = X_train[np.random.choice(np.arange(len(X_train)), n_clusters)]
    history_centers, history_labels = [], []
    while True:
        distance_2 = np.zeros((len(X_train), len(centers)))
        for i in range(len(centers)):
            distance_2[:,i] = ((X_train - centers[i])**2).sum(axis=1)
        
        labels = distance_2.argmin(axis=1)
        centers_new = np.array([X_train[labels == i].mean(0) for i in range(n_clusters)])
        
        history_centers.append(centers)
        history_labels.append(labels)
        if np.all(centers == centers_new):
            break
        centers = centers_new
    
    # skip the very last update, since it is similar to the last one
    return history_centers, history_labels

np.random.seed(8)
history_centers, history_labels = k_means_clustering(X, 3)

#%%
for n_iter in range(len(history_centers)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=history_labels[n_iter], s=50, cmap='rainbow')
    #ax.set_xlim(xmin=-4.5, xmax=3.5)
    #ax.set_ylim(ymin=-12.5, ymax=2.2)
    plt.xlabel(r"$x^{(1)}$", fontsize=15)
    plt.ylabel(r"$x^{(2)}$", fontsize=15)
    
    centers = history_centers[n_iter]
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1, marker='s')
    kmeans.cluster_centers_ = centers
    DecisionBoundaryDisplay.from_estimator(kmeans, X, alpha=0.2, ax=ax, response_method="predict", cmap=None)
    
    plt.tight_layout()
    fig.savefig(f"../Picture/K_means_iter_{n_iter}.png", dpi=200)

print(len(history_centers))

#%%
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=2000, noise=0.1, random_state=3)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.scatter(X[:, 1], X[:, 0], c=y, s=50, cmap='rainbow')
plt.xlabel(r"$x^{(1)}$", fontsize=15)
plt.ylabel(r"$x^{(2)}$", fontsize=15)
plt.tight_layout()
fig.savefig("../Graphs/K_means_moon_original.png", dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.scatter(X[:, 1], X[:, 0], c=y_kmeans, s=50, cmap='rainbow')
plt.xlabel(r"$x^{(1)}$", fontsize=15)
plt.ylabel(r"$x^{(2)}$", fontsize=15)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=200, alpha=1, marker='s')
plt.tight_layout()
fig.savefig("../Picture/K_means_moon.png", dpi=200)

#%%
# K-means segmentation

import numpy as np
import matplotlib.pyplot as plt
import cv2
 
# Read in the image
image = cv2.imread("../Picture/HK_2.jpg")
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(30, 20), dpi=200)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.imshow(image)

#%%

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
# then perform k-means clustering with number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 5
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
 
fig = plt.figure(figsize=(30, 20), dpi=200)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.imshow(segmented_image)
plt.tight_layout()
plt.savefig(f"../Picture/HK_2_5.png")