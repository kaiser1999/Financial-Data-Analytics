import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

d = pd.read_csv("stock_1999_2002.csv", index_col=0)
returns = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return
n_sim = int(1e5)

# QQ plot for empirical marginals
col = ["blue", "orange", "green"]
n_days, n_stocks = returns.shape
i = (np.arange(1, n_days + 1) - 0.5) / n_days
fig, axes = plt.subplots(ncols=n_stocks, figsize=(10*n_stocks, 10))
for k in range(n_stocks):
    returns_k = returns.iloc[:,k]
    q = np.quantile(returns_k, i, method='interpolated_inverted_cdf')
    com = returns.columns[k]
    b, w = np.linalg.lstsq(np.c_[np.ones(n_days), q], 
                           np.sort(returns_k), rcond=None)[0]
    axes[k].scatter(q, np.sort(returns_k), color=col[k],
                facecolor="white", marker="o", s=50)
    axes[k].plot(q, w*q + b, color="blue", linewidth=2)
    axes[k].set_xlabel("Empirical quantiles", fontsize=15)
    axes[k].set_ylabel("Returns quantiles", fontsize=15)
    axes[k].set_title(f"Q-Q Plot with {com}'s returns", fontsize=20)

fig.tight_layout()
fig.savefig("../Picture/Empirical_Marginals.png", dpi=200)

from statsmodels.distributions.empirical_distribution import ECDF
def empirical_marginals(x):
    return np.apply_along_axis(lambda z: ECDF(z)(z), axis=0, arr=x)

def empirical_quantile(p, samples):
    p, samples = np.array(p), np.array(samples)
    q = np.empty(p.shape)
    for k in range(p.shape[1]):
        q[:,k] = np.quantile(samples[:,k], p[:,k], 
                             method='interpolated_inverted_cdf')
    
    return q

# using empirical as the marginal distribution
emp_u = empirical_marginals(returns)
    
#%%
from copulae import NormalCopula

N_cop_dist = NormalCopula(dim=len(d.columns))
N_cop_dist.fit(emp_u, verbose=0)
# Generate random samples based on gaussian copula
u_sim_N = pd.DataFrame(N_cop_dist.random(n_sim, seed=4002),
                       columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(u_sim_N[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'blue'})
print(np.corrcoef(u_sim_N, rowvar=False))
print(np.corrcoef(returns, rowvar=False))

plt.tight_layout()
plt.savefig("../Picture/Gaussian_sample_Copula.png", dpi=200)

# Get back returns based on the random samples
return_sim_N = pd.DataFrame(empirical_quantile(u_sim_N, returns), 
                            columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(return_sim_N[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'green'})

plt.tight_layout()
plt.savefig("../Picture/Gaussian_Copula.png", dpi=200)
#%%
def Mahalanobis2(X):
    X = np.array(X)
    mu = np.mean(X, axis=0)
    inv_Sig = np.linalg.inv(np.cov(X, rowvar=False))
    X_minus_mu = X - mu
    return np.sum((X_minus_mu @ inv_Sig) * X_minus_mu, axis=1)

def Mahalanobis_QQ_Plot(sim_data, returns, col="blue"):
    n_days = len(returns)
    sim_Mahalanobis = Mahalanobis2(sim_data)
    raw_Mahalanobis = Mahalanobis2(returns)
    
    i = (np.arange(1, n_days + 1) - 0.5) / n_days
    q = np.quantile(sim_Mahalanobis, i, 
                    method='interpolated_inverted_cdf')
    
    fig = plt.figure(figsize=(10, 10))
    b, w = np.linalg.lstsq(np.c_[np.ones(n_days), q], 
                           np.sort(raw_Mahalanobis), rcond=None)[0]
    plt.scatter(q, np.sort(raw_Mahalanobis), color=col,
                    facecolor="white", marker="o", s=50)
    plt.plot(q, w*q + b, color="blue", linewidth=2)
    plt.xlabel("Empirical copula quantiles", fontsize=15)
    plt.ylabel("Returns quantiles", fontsize=15)
    plt.title("Squared Mahalanobis Q-Q Plot with empirical marginals", 
              fontsize=20)
    
    return fig

#%%

fig = Mahalanobis_QQ_Plot(return_sim_N, returns, col="blue")

fig.tight_layout()
fig.savefig("../Picture/Empirical N QQ Plot.png", dpi=200)

from scipy import stats

d2 = Mahalanobis2(returns)
i = (np.arange(1, n_days+1)-0.5)/n_days
q = stats.chi2.ppf(i, 3)

fig = plt.figure(figsize=(10, 10))
b, w = np.linalg.lstsq(np.vstack([np.ones(n_days), q]).T, np.sort(d2), 
                       rcond=None)[0]
plt.scatter(q, np.sort(d2), color="blue", marker="o", s=50)
plt.plot(q, w*q + b, color="blue", linewidth=2)
plt.title("Chi2 Q-Q Plot", fontsize=20)

plt.tight_layout()
plt.savefig("../Picture/MVN Chi2 Plot.png", dpi=200)

#%%
from copulae import StudentCopula

t_cop_dist = StudentCopula(dim=len(d.columns))
t_cop_dist.fit(emp_u, verbose=0)
print(t_cop_dist.params.rho)
print(t_cop_dist.params.df)

# Generate random samples based on t-copula
u_sim_t = pd.DataFrame(t_cop_dist.random(n_sim, seed=4002),
                       columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(u_sim_N[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'blue'})

plt.tight_layout()
plt.savefig("../Picture/t_sample_Copula.png", dpi=200)

# Get back returns based on the random samples
return_sim_t = pd.DataFrame(empirical_quantile(u_sim_t, returns), 
                            columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(return_sim_t[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'green'})

plt.tight_layout()
plt.savefig("../Picture/t_Copula.png", dpi=200)

#%%
fig = Mahalanobis_QQ_Plot(return_sim_t, returns, col="orange")

fig.tight_layout()
fig.savefig("../Picture/Empirical t QQ Plot.png", dpi=200)