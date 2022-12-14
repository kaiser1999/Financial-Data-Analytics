import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate
import seaborn as sns

d = pd.read_csv("stock_1999_2002.csv")
returns = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return
n_sim = 1000

N_cop_dist = GaussianMultivariate()
N_cop_dist.fit(returns)
# Generate random return samples based on multi-variate normal copula
np.random.seed(4002)
return_sim_N = N_cop_dist.sample(n_sim)

sns.pairplot(pd.concat([returns.assign(label='return'), 
                        return_sim_N.assign(label='sim_return')]), 
             hue='label', diag_kind='kde', palette=['orange', 'blue'])

#%%

from copulas.univariate import StudentTUnivariate

t_cop_dist = GaussianMultivariate(distribution={
    "HSBC": StudentTUnivariate,
    "CLP": StudentTUnivariate,
    "CK": StudentTUnivariate})

t_cop_dist.fit(returns)
# Generate random return samples based on multi-variate t copula
np.random.seed(4002)
return_sim_t = t_cop_dist.sample(n_sim)

sns.pairplot(pd.concat([returns.assign(label='return'), 
                        return_sim_t.assign(label='sim_return')]), 
             hue='label', diag_kind='kde', palette=['orange', 'blue'])