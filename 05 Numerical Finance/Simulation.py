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
plt.savefig("../Picture/prnguni.png", dpi=200)

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
_, bins, _ = plt.hist(pse_sample, alpha=0.7, label="Pseudo Generator", 
                      color="blue", ec='black', bins="sturges")
plt.hist(built_in_sample, alpha=0.7, label="Numpy Generator", 
         color="red", ec='black', bins=bins)
x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), 10000)
plt.plot(x, expon.pdf(x)*10000)
plt.legend(loc='upper right')
plt.savefig("../Picture/prngexp.png", dpi=200)

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
_, bins, _ = plt.hist(pse_sample, alpha=0.7, label="Pseudo Generator", 
                      color="blue", ec='black', bins="sturges")
plt.hist(built_in_sample, alpha=0.7, label="Numpy Generator", 
         color="red", ec='black', bins=bins)
plt.legend()
plt.savefig("../Picture/prngnorm.png", dpi=200)
