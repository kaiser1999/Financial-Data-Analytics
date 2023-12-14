import pandas as pd
import numpy as np
np.set_printoptions(precision=4) # control display into 4 digis

HSI_2002 = pd.read_csv("../Datasets/fin-ratio.csv")
print(HSI_2002.columns.values)

X_2002 = HSI_2002.drop(columns="HSI").values # A 680x6 data matrix

mu_2002 = np.mean(X_2002, axis=0) # Mean vector
print(mu_2002)

S_2002 = np.cov(X_2002, rowvar=False) # Covariance matrix
print(S_2002)

R_2002 = np.corrcoef(X_2002, rowvar=False) # Correlation matrix
print(R_2002)

#%%
np.set_printoptions(precision=4) # control display into 4 digis

print(np.linalg.det(np.linalg.inv(S_2002))) # |A^-1| = 1/|A|
print(1/np.linalg.det(S_2002))

eig_val_2002, H_2002 = np.linalg.eig(S_2002)
print(eig_val_2002)
print(H_2002)

print(np.round(H_2002.T @ H_2002, 3)) # H^T H = I

print(H_2002[:,1] @ H_2002[:,2]) # h_1^T h_2 = 0

print(np.round(H_2002.T @ S_2002 @ H_2002, 3)) # H^T A H = D
D_2002 = np.diag(eig_val_2002)
print(D_2002)

sqrt_S_2002 = H_2002 @ np.sqrt(D_2002) @ H_2002.T # H D^1/2 H^T = A^1/2
print(sqrt_S_2002 @ sqrt_S_2002)
print(S_2002)