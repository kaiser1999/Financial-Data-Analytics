import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import datetime as dt
from KMeansCluster import best_km

((X_train, y_train), 
 (X_test, y_test)) = tf.keras.datasets.mnist.load_data()

N_train = len(X_train)                  # 60000 training samples
N_test = len(X_test)                    # 10000 test samples
X_train = X_train.reshape(N_train, -1).astype('float32') / 255.0
X_test = X_test.reshape(N_test, -1).astype('float32') / 255.0
K = 10

#%%
start = dt.now()
_, km = best_km(X_train, K)                     # K-means with K=10
y_hat_km = km.predict(X_test)
tab = confusion_matrix(y_hat_km, y_test)        # Confusion matrix
print(tab)

pos_dict = dict(zip(np.arange(K), np.argmax(tab, axis=1)))
print(pos_dict)
y_pred = [pos_dict[k] for k in y_hat_km]
tab = confusion_matrix(y_pred, y_test)          # Confusion matrix
print(tab)
print(sum(tab.diagonal()) / len(y_test))        # Accuracy
print(dt.now() - start)                         # K-means

#%%
from sklearn.decomposition import PCA

start = dt.now()
pca = PCA(n_components=.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f'Total number of components: {pca.n_components_}')

_, km_pca = best_km(X_train_pca, K)             # K-means with K=10
y_hat_pca = km_pca.predict(X_test_pca)
tab = confusion_matrix(y_hat_pca, y_test)       # Confusion matrix
print(tab)

pos_dict = dict(zip(np.arange(K), np.argmax(tab, axis=1)))
print(pos_dict)
y_pred = [pos_dict[k] for k in y_hat_pca]
tab = confusion_matrix(y_pred, y_test)          # Confusion matrix
print(tab)
print(sum(tab.diagonal()) / len(y_test))        # Accuracy
print(dt.now() - start)                         # PCA + K-means

#%%
from sklearn.neighbors import KNeighborsClassifier

start = dt.now()
knn = KNeighborsClassifier(n_neighbors=3)       # KNN with K=3
knn.fit(X_train, y_train)
y_hat_knn = knn.predict(X_test)
tab = confusion_matrix(y_hat_knn, y_test)       # Confusion matrix
print(tab)
print(sum(tab.diagonal()) / len(y_test))        # Accuracy
print(dt.now() - start)                         # KNN

#%%
start = dt.now()
knn_pca = KNeighborsClassifier(n_neighbors=3)   # KNN with K=3
knn_pca.fit(X_train_pca, y_train)
y_hat_pca = knn_pca.predict(X_test_pca)
tab = confusion_matrix(y_hat_pca, y_test)       # Confusion matrix
print(tab)
print(sum(tab.diagonal()) / len(y_test))        # Accuracy
print(dt.now() - start)                         # PCA + KNN