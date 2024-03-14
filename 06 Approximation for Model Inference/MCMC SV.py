import pandas as pd
import numpy as np

import pymc as pm

#%%
df = pd.read_csv("../Datasets/stock_1999_2002.csv", index_col=0)
y = np.log(1 + df.pct_change()).iloc[1:,:]
y_HSBC, y_CLP, y_CK = y.HSBC.values, y.CLP.values, y.CK.values

np.random.seed(4002)
with pm.Model() as model:
    '''
    mu, sigma = np.array([0, 0]), np.array([100, 1])
    lower, upper = np.array([-1e10, -1]), np.array([1e10, 1])
    rho = pm.TruncatedNormal("rho", mu=mu, sigma=sigma, 
                             lower=lower, upper=upper, shape=2)
    '''
    rho = pm.Uniform("rho", lower=-1, upper=1, shape=2)
    nu = pm.InverseGamma("nu", alpha=2.5, beta=0.025)
    h = pm.AR("h", rho=rho, sigma=pm.math.sqrt(nu), constant=True, shape=len(y_HSBC))
    y = pm.Normal("y", mu=0, sigma=pm.math.exp(h/2), observed=y_HSBC)

    trace = pm.sample(draws=2000, tune=1000, random_seed=4002, cores=1)

#%%
print(trace)
print(trace.summary())