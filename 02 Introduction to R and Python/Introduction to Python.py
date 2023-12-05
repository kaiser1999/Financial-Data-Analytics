import numpy as np
import pandas as pd

x = 1
if x > 0:
	print("x is positive")
	print("Time to begin with machine learning!")

# Read csv in Python
data = pd.read_csv("../Datasets/fin-ratio.csv")
# Write csv in Python
data.to_csv("../Datasets/fin-ratio_new.csv")

#%%
# Function in Python
from My_Function import MyFun

print(MyFun(2))
print(MyFun(2) + 10000)

# For loop in Python
for i in range(10):
  print(i)

# Assign data to x except the label y = HSI stock or not
x = data.drop(data.columns[-1], axis=1)
print(data.columns[-1])

# Compute sample means and sample variances
print(x.mean())
print(x.var())

# Compute sample covariance matrix
print(x.cov())

#%%
import numpy as np

A = np.array([1, 3, 5, 2, 6, 4, 2, 3, 1]).reshape(3,3)
B = np.array([3, 1, 2, 4, 2, 8, 1, 3, 1]).reshape(3,3)
print(A)
print(B)

# Matrix Multiplication
print(np.dot(A, B))
print(A @ B)

# Hadamard Product
print(np.multiply(A, B))
print(A * B)

# Inverse
print(np.linalg.inv(A))

#%%
import matplotlib.pyplot as plt
import scipy.stats as stats

HSBC = pd.read_csv("../Datasets/0005.HK.csv")
X = HSBC["Adj Close"]
log_returns = np.log(X) - np.log(X.shift(1))
log_returns = log_returns.iloc[1:]

fig = plt.figure(figsize=(13,5))
plt.subplot(1, 2, 1)
plt.hist(log_returns, bins="sturges", ec='black')
plt.title("Histogram_HSBC")

plt.subplot(1, 2, 2)

stats.probplot(log_returns, dist="norm", plot=plt)
plt.title("QQNorm_HSBC")
plt.tight_layout()
plt.savefig("../Graphs/HSBC Norm python.png")
plt.show()

#%%
import numpy as np
import scipy.stats as stats

u = log_returns
print(stats.kstest(u, "norm", args=(np.mean(u), np.std(u))))

def KS_test(x, func=stats.norm, *args):
    n = len(x)
    S_n = np.arange(n+1)/n
    z = np.sort(x)
    Phi_z = func.cdf(z, *args)
    Term1 = S_n[1:] - Phi_z
    Term2 = S_n[:-1] - Phi_z
    KS = np.max([Term1, Term2])
    crit = np.sqrt(-0.5*np.log(0.05/2)/n)
    return({"KS-stat": KS, "critical value":crit})

print(KS_test(u, stats.norm, np.mean(u), np.std(u)))

#%%
import numpy as np
import scipy.stats as stats

print(stats.jarque_bera(log_returns))

def JB_test(x):
    u = x - np.mean(x)                          # remove mean
    n = len(u)                                  # sample size
    s = np.std(u)                               # compute population sd
    sk = sum(u**3)/(n*s**3)                     # compute skewness
    ku = sum(u**4)/(n*s**4)-3                   # excess kurtosis
    JB = n*(sk**2/6+ku**2/24)                   # JB test stat
    p = 1 - stats.chi2.cdf(JB,2)                # p-value
    return({"JB-stat": JB, "p-value": p})       # output 

print(JB_test(log_returns))

#%%
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../Datasets/Pearson.txt", sep="\t")
model = LinearRegression()
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)
model.fit(x, y)
print({"intercept": model.intercept_, "slope": model.coef_})
y_pred = model.predict(x)
plt.figure(figsize=(5, 5))
plt.scatter(x, y, edgecolors="black", color="white", s=20)
plt.plot(x, y_pred, color="red")
plt.xlabel("Father")
plt.ylabel("Son")
plt.title("Heights of fathers and their full grown son (in inches)")
plt.tight_layout()
plt.savefig("../Graphs/Pearson python.png", dpi=200)
plt.show()