import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%%
d = pd.read_csv("fin-ratio.csv")
X, y = d.drop(columns="HSI"), d.HSI

# split data into training and testing sets with a ratio 8:2
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=4012)


# Build a Gaussian Naive Bayes classifier with training sets
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
y_pred = gnb_clf.predict(X_test)
print(confusion_matrix(y_pred, y_test))

# mean and variance of each feature variable
print(gnb_clf.theta_)                   # mean
print(gnb_clf.var_)                     # variance