from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

X, y = load_boston(return_X_y=True)
LSTAT = X[:,-1]

# Split the data into training and testing dataset in ratio of 8:2
LSTAT_train, LSTAT_test, y_train, y_test = train_test_split(LSTAT, y, 
                                                            test_size=0.2, 
                                                            random_state=4002)

# Build the model
poly = PolynomialFeatures(degree=5)
LSTAT_train_5 = poly.fit_transform(LSTAT_train.reshape(-1, 1))
LSTAT_test_5 = poly.fit_transform(LSTAT_test.reshape(-1, 1))

model = sm.OLS(y_train, LSTAT_train_5).fit()
print(model.summary())

# Model predictions
predict = model.predict(LSTAT_test_5)

# Model performance
from sklearn.metrics import r2_score, mean_squared_error
print(mean_squared_error(y_test, predict, squared=False))   # RMSE
print(r2_score(y_test, predict))    # R2

fig, ax = plt.subplots(figsize=(15, 10))
plt.scatter(LSTAT_train, y_train, edgecolors="black", 
            color="white", s=20)

line = np.linspace(1, 40, 100)
line_5 = poly.fit_transform(line.reshape(-1, 1))
plt.plot(line, model.predict(line_5), color="red")
fig.savefig("Polynomial.png", dpi=200)