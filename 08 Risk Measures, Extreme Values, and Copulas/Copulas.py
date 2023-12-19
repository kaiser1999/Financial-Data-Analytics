import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

d = pd.read_csv("../Datasets/stock_1999_2002.csv", index_col=0)
returns = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return
n_sim = int(1e5)

# Q-Q plot for empirical marginals
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
    return np.apply_along_axis(lambda z: ECDF(z, side="left")(z), 
                               axis=0, arr=x)

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
print(np.corrcoef(u_sim_N, rowvar=False, ddof=1))
print(np.corrcoef(returns, rowvar=False, ddof=1))

plt.tight_layout()
#plt.savefig("../Picture/Gaussian_sample_Copula.png", dpi=200)

# Get back returns based on the random samples
return_sim_N = pd.DataFrame(empirical_quantile(u_sim_N, returns), 
                            columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(return_sim_N[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'green'})

plt.tight_layout()
#plt.savefig("../Picture/Gaussian_Copula.png", dpi=200)
#%%
def Mahalanobis2(X):
    X = np.array(X)
    mu = np.mean(X, axis=0)
    inv_Sig = np.linalg.inv(np.cov(X, rowvar=False))
    X_minus_mu = X - mu
    return np.sum((X_minus_mu @ inv_Sig) * X_minus_mu, axis=1)

def QQ_Plot(sim_data, raw_data, col="blue"):
    n_days = len(raw_data)
    
    i = (np.arange(1, n_days + 1) - 0.5) / n_days
    q = np.quantile(sim_data, i, method='interpolated_inverted_cdf')
    
    fig = plt.figure(figsize=(10, 10))
    b, w = np.linalg.lstsq(np.c_[np.ones(n_days), q], 
                           np.sort(raw_data), rcond=None)[0]
    plt.scatter(q, np.sort(raw_data), color=col,
                    facecolor="white", marker="o", s=50)
    plt.plot(q, w*q + b, color="blue", linewidth=2)
    plt.xlabel("Empirical copula quantiles", fontsize=15)
    plt.ylabel("Returns quantiles", fontsize=15)
    plt.title("Squared Mahalanobis Q-Q Plot with empirical marginals", 
              fontsize=20)
    
    return fig

returns_md2 = Mahalanobis2(returns)

#%%
sim_N_md2 = Mahalanobis2(return_sim_N)
fig = QQ_Plot(sim_N_md2, returns_md2, col="blue")

fig.tight_layout()
fig.savefig("../Picture/Empirical N QQ Plot.png", dpi=200)

from scipy import stats

i = (np.arange(1, n_days+1)-0.5)/n_days
q = stats.chi2.ppf(i, 3)

fig = plt.figure(figsize=(10, 10))
b, w = np.linalg.lstsq(np.vstack([np.ones(n_days), q]).T, 
                       np.sort(returns_md2), rcond=None)[0]
plt.scatter(q, np.sort(returns_md2), color="blue", marker="o", s=50)
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
sns.pairplot(u_sim_t[1:1000], diag_kind="kde", 
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
sim_t_md2 = Mahalanobis2(return_sim_t)
fig = QQ_Plot(sim_t_md2, returns_md2, col="orange")

fig.tight_layout()
fig.savefig("../Picture/Empirical t QQ Plot.png", dpi=200)

#%%
n_days = len(returns)
i = (np.arange(1, n_days + 1) - 0.5) / n_days
q_N = np.quantile(sim_N_md2, i, method='interpolated_inverted_cdf')
q_t = np.quantile(sim_t_md2, i, method='interpolated_inverted_cdf')

b_N, w_N = np.linalg.lstsq(np.c_[np.ones(n_days), q_N], 
                           np.sort(returns_md2), rcond=None)[0]
b_t, w_t = np.linalg.lstsq(np.c_[np.ones(n_days), q_t], 
                           np.sort(returns_md2), rcond=None)[0]

r2_N = (np.sort(returns_md2) - (b_N + w_N*q_N))**2
r2_t = (np.sort(returns_md2) - (b_t + w_t*q_t))**2

fig = plt.figure(figsize=(10, 10), dpi=200)
# plot theoretical quantiles starting from 10
idx_start = min(min(np.where(q_N>10)[0]), min(np.where(q_t>10)[0]))
print(idx_start)
plt.plot(np.sort(r2_N[idx_start:-1]), "ro", markersize=10)
plt.plot(np.sort(r2_t[idx_start:-1]), "bx", markersize=10)
plt.xlabel("Index", fontsize=15)
plt.ylabel("Squared residuals", fontsize=15)
plt.legend(["Gaussian copula", "t-copula"], fontsize=15)