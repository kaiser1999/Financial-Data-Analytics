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