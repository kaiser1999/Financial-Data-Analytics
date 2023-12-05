import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate
import seaborn as sns
import matplotlib.pyplot as plt

d = pd.read_csv("stock_1999_2002.csv", index_col=0)
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

plt.tight_layout()
plt.savefig("../Picture/Gaussian_Copula.png", dpi=200)

#%%
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d

def Empirical_QQ_plot(sim_data, returns, col=None):
    n_stocks = len(returns.columns)
    n_days   = len(returns)
    if col is None: col = ["black"] * n_stocks
    
    fig, axes = plt.subplots(ncols=n_stocks, figsize=(10*n_stocks, 10))
    for k in range(n_stocks):
        sim_data_k = sim_data.iloc[:,k]
        returns_k  = returns.iloc[:,k]
        sim_data_k_quantile = edf.ECDF(sim_data_k)(sim_data_k)
        inverted_k_edf = interp1d(sim_data_k_quantile, sim_data_k)
        i = (np.arange(1, n_days + 1) - 0.5) / n_days
        q = inverted_k_edf(i)
        
        b, w = np.linalg.lstsq(np.c_[np.ones(n_days), q], 
        					   np.sort(returns_k), rcond=None)[0]
        axes[k].scatter(q, np.sort(returns_k), color=col[k],
                        facecolor="white", marker="o", s=50)
        axes[k].plot(q, w*q + b, color="blue", linewidth=2)
        axes[k].set_xlabel("Empirical quantiles", fontsize=15)
        axes[k].set_ylabel("Returns quantiles", fontsize=15)
        axes[k].set_title(f"{d.columns.values[k]} Empirical Q-Q Plot", 
                          fontsize=20)
    
    return fig

total_sim = 1e5; col = ["blue", "orange", "green"]

#%%
np.random.seed(4002)
sim_N = N_cop_dist.sample(int(total_sim))
fig = Empirical_QQ_plot(sim_N, returns, col=col)

fig.tight_layout()
fig.savefig("../Picture/Empirical N QQ Plot.png", dpi=200)
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

plt.tight_layout()
plt.savefig("../Picture/t_Copula.png", dpi=200)

#%%
np.random.seed(4002)
sim_t = t_cop_dist.sample(int(total_sim))
fig = Empirical_QQ_plot(sim_t, returns, col=col)

fig.tight_layout()
fig.savefig("../Picture/Empirical t QQ Plot.png", dpi=200)