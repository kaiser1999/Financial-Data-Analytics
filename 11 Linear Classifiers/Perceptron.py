import numpy as np

def perceptron(X, y, eta=1e-3, n_iter=1e5):
    X, y = np.array(X), np.array(y)
    X_1 = np.c_[np.ones(len(y)), X]
    n, p = X_1.shape
    omega = np.repeat(1/p, p)           # initialize weight vector            
    for it in range(int(n_iter)):
        i = it % n
        if i == 0:                      # Python index starts from 0
            new_idx = np.random.choice(np.arange(n), n, replace=False)
        
        j = new_idx[i]
        if y[j] * omega @ X_1[j] < 0:   # misclassified
            omega += eta * y[j] * X_1[j]
    
    return omega, np.sign(X_1 @ omega)

#%%
import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../Datasets/fin-ratio.csv")
X, y = df.drop(columns=["HSI"]), df["HSI"]
y = y.replace(0, -1)				    # transform the output to {-1, 1}

np.random.seed(4002)
omega, y_pred = perceptron(X, y, eta=0.01, n_iter=1e6)
conf = confusion_matrix(y_pred, y) 	    # confusion matrix
print(conf)
print(sum(np.diag(conf))/len(y))
print(omega)