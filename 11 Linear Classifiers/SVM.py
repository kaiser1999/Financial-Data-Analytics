import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

df = pd.read_csv("../Datasets/credit_scoring_sample.csv")
df = df.dropna()

y = df.SeriousDlqin2yrs.values
X = df.drop(columns="SeriousDlqin2yrs").to_numpy()
(X_train, X_test, 
 y_train, y_test) = train_test_split(X, y, test_size=0.2, 
                                     random_state=4002)

#%%
svm_lnr = SVC(kernel="linear", C=1)
svm_lnr.fit(X_train, y_train)
ypred = svm_lnr.predict(X_test)
print(accuracy_score(ypred, y_test))
print(confusion_matrix(ypred, y_test))

#%%
# exp(-gamma \norm{x-x'}**2); gamma = 1/(2*sigma2)
gamma = lambda sig: 1/(2*sig**2)
sigma = [0.5, 1, 5, 10, 50, 100]

train_acc, test_acc = list(), list()
for sig in tqdm(sigma, total=len(sigma)):
    svm_rbf = SVC(kernel="rbf", C=1, gamma=gamma(sig))
    svm_rbf.fit(X_train, y_train)
    ypred_train = svm_rbf.predict(X_train)
    ypred_test = svm_rbf.predict(X_test)
    train_acc.append(accuracy_score(ypred_train, y_train))
    test_acc.append(accuracy_score(ypred_test, y_test))

colnames = ["sigma", "gamma", "train-accuarcy", "test-accuracy"]
print(pd.DataFrame(dict(zip(colnames, [sigma, gamma(np.array(sigma)), 
                                       train_acc, test_acc]))))

#%%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
plt.plot(sigma, test_acc, c="orange", linewidth=2)
plt.plot(sigma, train_acc, c="blue", linewidth=2)
plt.ylim(0.75, 1)
plt.xlabel("sigma", fontsize=15)
plt.ylabel("accuracy", fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/SVM RBF sigma train_test.png", dpi=200)

fig = plt.figure(figsize=(10, 8))
plt.plot(sigma, test_acc, c="orange", linewidth=2)
plt.xlabel("sigma", fontsize=15)
plt.ylabel("accuracy", fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/SVM RBF credit scoring.png", dpi=200)