x = 1
if x > 0:
	print("x is positive")
	print("Time to begin with machine learning!")
    
#%%
import tensorflow as tf

#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
import pandas as pd

# Read csv in Python
data = pd.read_csv('fin-ratio.csv')
# Write csv in Python
data.to_csv("fin-ratio_new.csv")

#%%
# Assign data to x except the label y = HSI stock or not
x = data.drop(data.columns[-1], axis=1)
# data.columns[-1] = HSI, axis=1 means column (axis=0 for row)
# Compute sample means and sample variances
print(x.mean())
print(x.var())

# Compute sample covariance matrix
print(x.cov())

#%%
from My_Function import MyFun

MyFun(2)

#%%
# For loop in Python
for i in range(5):
	print(i)
    
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
print(np.linalg.inv(A))			# LINear ALGebra