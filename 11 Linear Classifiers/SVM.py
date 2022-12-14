import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("fin-ratio.csv")   # read in dataset
X = df.drop(columns="HSI").values
y = df.HSI.values

# split data into training and testing sets with a ratio of 7:3
(X_train, X_test, 
 y_train, y_test) = train_test_split(X, y, test_size=0.3, 
                                     random_state=4002)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_clf = svm.SVC(kernel="linear", C=1)
svm_clf.fit(X_train, y_train)
ypred = svm_clf.predict(X_test)
print(confusion_matrix(ypred, y_test))

#%%
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

fig = plt.figure(figsize=(10, 10))
values, ranges = 1.5, 0.75
values_dict = {0: values, 3:values, 4:values, 5:values}
ranges_dict = {0: ranges, 3:ranges, 4:ranges, 5:ranges}

# Plot Decision Region using mlxtend
plot_decision_regions(X=X_train, y=y_train, clf=svm_clf, 
                      legend=1, scatter_kwargs={'s':120}, 
                      feature_index=(1,2),
                      filler_feature_values=values_dict,
                      filler_feature_ranges=ranges_dict)
plt.xlabel('ln_MV', size=20)
plt.ylabel('CFTP', size=20)
plt.show()
#plt.savefig(f"SVM sigma {sig}.png", dpi=500)