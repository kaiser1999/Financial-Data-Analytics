import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import datetime as dt

((X_train, y_train), 
 (X_test, y_test)) = tf.keras.datasets.mnist.load_data()

N_train = len(X_train)                  # 60000 training samples
N_test = len(X_test)                    # 10000 test samples
X_train = X_train.reshape(N_train, -1).astype('float32') / 255.0
X_test = X_test.reshape(N_test, -1).astype('float32') / 255.0

start = dt.now()
from sklearn.linear_model import LogisticRegression

np.random.seed(4002)
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
y_hat_logit = clf.predict(X_test)

# Table of predictions and prediction accuracy
tab = confusion_matrix(y_hat_logit, y_test)     # Confusion matrix
print(tab)
print(sum(tab.diagonal()) / len(y_test))        # Accuracy
print(dt.now() - start)                  # Logistic regression

#%%
start = dt.now()
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f'Total number of components: {pca.n_components_}')

np.random.seed(4002)
clf_pca = LogisticRegression(max_iter=10000)
clf_pca.fit(X_train_pca, y_train)
y_hat_pca = clf_pca.predict(X_test_pca)

# Table of predictions and prediction accuracy
tab = confusion_matrix(y_hat_pca, y_test)       # Confusion matrix
print(tab)
print(sum(tab.diagonal()) / len(y_test))        # Accuracy
print(dt.now() - start)                  # PCA + Logistic regression