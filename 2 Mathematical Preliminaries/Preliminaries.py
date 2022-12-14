import pandas as pd
import numpy as np
np.set_printoptions(precision=4) # control display into 4 digis

HSI_2002 = pd.read_csv("Financial ratio_2002.csv")
print(HSI_2002.columns.values)

X_2002 = HSI_2002.drop(columns="HSI").values # A 680x6 data matrix

mu_2002 = np.mean(X_2002, axis=0) # Mean vector
print(mu_2002)

S_2002 = np.cov(X_2002, rowvar=False) # Covariance matrix
print(S_2002)

R_2002 = np.corrcoef(X_2002, rowvar=False) # Correlation matrix
print(R_2002)

HSI_2018 = pd.read_csv("Financial ratio_2018.csv")
print(HSI_2018.columns.values)

X_2018 = HSI_2018.drop(columns="HSI").values # A 680x6 data matrix

mu_2018 = np.mean(X_2018, axis=0) # Mean vector
print(mu_2018)

S_2018 = np.cov(X_2018, rowvar=False) # Covariance matrix
print(S_2018)

R_2018 = np.corrcoef(X_2018, rowvar=False) # Correlation matrix
print(R_2018)


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

#%%
import statsmodels.api as sm

X_2018, y_2018 = HSI_2018.drop(columns="HSI"), HSI_2018["HSI"]

X_intercept_2018 = sm.add_constant(X_2018)

logit_model = sm.GLM(y_2018, X_intercept_2018, family=sm.families.Binomial()).fit()
print(logit_model.summary())

#%%

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4002)

def pse_uni_gen(a=16807, c=1, m=2**31-1, seed=123456789, 
                size=1, burn_in=1000):
	x = [(a*seed + c) % m]
	for i in range(1, size + burn_in):
		x.append((a*x[-1] + c) % m)
	return np.array(x[burn_in:]) / m

def pse_uniform_gen(lower=0, upper=1, seed=123456789, 
                    size=1, burn_in=1000):
    U = pse_uni_gen(seed=seed, size=size, burn_in=burn_in)
    return lower + (upper - lower)*U  

pse_sample = pse_uniform_gen(lower=0, upper=1, size=10000)
built_in_sample = np.random.rand(10000)
plt.figure(figsize=(8,6))
plt.hist(pse_sample, ec='black', bins="sturges", 
         alpha=0.7, label="Pseudo Generator")
plt.hist(built_in_sample, ec='black', bins="sturges", 
         alpha=0.7, label="Numpy Generator")
plt.legend(loc='upper right')
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
np.random.seed(4002)

def pseudo_exp_gen(lamb, seed=123456789, size=1, burn_in=1000):
	U = pse_uniform_gen(seed=seed, size=size, burn_in=burn_in)
	X = -(1/lamb)*np.log(1-U)
	return X

pse_sample = pseudo_exp_gen(lamb=1, size=10000)
built_in_sample = np.random.exponential(1, size=10000)
plt.figure(figsize=(8,6))
plt.hist(pse_sample, ec='black', bins="sturges", 
         alpha=0.7, label="Pseudo Generator")
plt.hist(built_in_sample, ec='black', bins="sturges", 
         alpha=0.7, label="Numpy Generator")
x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), 10000)
plt.plot(x, expon.pdf(x)*10000)
plt.legend(loc='upper right')
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4002)

def pseudo_normal_gen(mu=0.0, sigma=1.0, seed=123456789, 
                      size=1, burn_in=1000):
    U = pse_uniform_gen(seed=seed, size=2*size, burn_in=burn_in)
    U1, U2 = U[:size], U[size:]
    Z0 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi * U2)
    Z1 = np.sqrt(-2*np.log(U1))*np.sin(2*np.pi * U2)
    return Z0*sigma + mu
	
pse_sample = pseudo_normal_gen(mu=0, sigma=1, size=10000)
built_in_sample = np.random.normal(0, 1, size=10000)
plt.figure(figsize=(8,6))
plt.hist(pse_sample, ec='black', bins="sturges", 
         alpha=0.7, label="Pseudo Generator")
plt.hist(built_in_sample, ec='black', bins="sturges", 
         alpha=0.7, label="Numpy Generator")
plt.legend()
plt.show()