import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

((X_train, y_train), 
 (X_test, y_test)) = tf.keras.datasets.mnist.load_data()

N_train = len(X_train)                  # 60000 training samples
N_test = len(X_test)                    # 10000 test samples
X_train = X_train.reshape(N_train, -1).astype('float32')
X_test = X_test.reshape(N_test, -1).astype('float32')

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

pca = PCA(n_components=.95)
pca.fit(X_train_scale)
print(f'Total number of components used: {pca.n_components_}')

X_train_pca = pca.transform(X_train_scale)
X_test_pca = pca.transform(X_test_scale)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_pca, y_train)
y_hat_pca = clf.predict(X_test_pca)

# Table of predictions and prediction accuracy
tab = confusion_matrix(y_test, y_hat_pca)               # Confusion matrix
print(tab)

# No need to inter-change labels cuz mu is first initialized by class
print(np.sum(tab.diagonal()) / len(y_test))             # Accuracy

#%%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='saga', max_iter=10000, C=50)
clf.fit(X_train, y_train)
y_hat_logit = clf.predict(X_test)

# Table of predictions and prediction accuracy
tab = confusion_matrix(y_test, y_hat_logit)             # Confusion matrix
print(tab)

# No need to inter-change labels cuz mu is first initialized by class
print(np.sum(tab.diagonal()) / len(y_test))             # Accuracy