import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

period = "1999_2002"
period = "2006_2009"
d = pd.read_csv(f"../Datasets/stock_{period}.csv", index_col=0)
returns = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return
n_days, n_stocks = returns.shape
n_sim = int(1e5)

# compute pseudo observations
from scipy.stats import rankdata
pse_u = rankdata(returns, method='average', axis=0) / (n_days + 1)

# Q-Q plot for pseudo observations
col = ["blue", "orange", "green"]
q = (np.arange(1, n_days + 1) - 0.5) / n_days
fig, axes = plt.subplots(ncols=n_stocks, figsize=(10*n_stocks, 10))
for k in range(n_stocks):
    pse_u_k = pse_u[:,k]
    com = returns.columns[k]
    b, w = np.linalg.lstsq(np.c_[np.ones(n_days), q], 
                           np.sort(pse_u_k), rcond=None)[0]
    axes[k].scatter(q, np.sort(pse_u_k), color=col[k],
                facecolor="white", marker="o", s=50)
    axes[k].plot(q, w*q + b, color="blue", linewidth=2)
    axes[k].set_xlabel("Theoretical quantiles", fontsize=15)
    axes[k].set_ylabel("Returns quantiles", fontsize=15)
    axes[k].set_title(f"Q-Q Plot with {com}'s returns", fontsize=20)

fig.tight_layout()
fig.savefig(f"../Picture/Pseudo_Marginals_{period}.png", dpi=200)

def pseudo_quantile(p, samples):
    p, samples = np.array(p), np.array(samples)
    q = np.empty(p.shape)
    for k in range(p.shape[1]):
        q[:,k] = np.quantile(samples[:,k], p[:,k], 
                             method='interpolated_inverted_cdf')
    
    return q

#%%
from copulae import NormalCopula

N_cop_dist = NormalCopula(dim=len(d.columns))
N_cop_dist.fit(pse_u, verbose=0, to_pobs=False)
# Generate random samples based on gaussian copula
u_sim_N = pd.DataFrame(N_cop_dist.random(n_sim, seed=4002),
                       columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(u_sim_N[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'blue'})
print(np.corrcoef(u_sim_N, rowvar=False, ddof=1))
print(np.corrcoef(returns, rowvar=False, ddof=1))

plt.tight_layout()
plt.savefig(f"../Picture/Gaussian_sample_Copula_{period}.png", dpi=200)

# Get back returns based on the random samples
return_sim_N = pd.DataFrame(pseudo_quantile(u_sim_N, returns), 
                            columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(return_sim_N[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'green'})

plt.tight_layout()
plt.savefig(f"../Picture/Gaussian_Copula_{period}.png", dpi=200)
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
    plt.scatter(q, np.sort(raw_data), color=col,
                facecolor="white", marker="o", s=50)
    plt.axline([0, 0], [1, 1], color=col, linewidth=2)
    plt.xlabel("Bootstrapped quantiles", fontsize=15)
    plt.ylabel("Sample quantiles", fontsize=15)
    plt.title("Copula Q-Q Plot", fontsize=20)
    
    return fig

returns_md2 = Mahalanobis2(returns)

#%%
sim_N_md2 = Mahalanobis2(return_sim_N)
fig = QQ_Plot(sim_N_md2, returns_md2, col="blue")

fig.tight_layout()
fig.savefig(f"../Picture/Pseudo N QQ Plot_{period}.png", dpi=200)

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
plt.savefig(f"../Picture/MVN Chi2 Plot_{period}.png", dpi=200)

print(stats.kstest(np.sqrt(returns_md2), np.sqrt(sim_N_md2), 
                   method="asymp"))
print(stats.kstest(np.sqrt(returns_md2), stats.chi2.cdf, 
                   args=(3,), method="asymp"))

#%%
from copulae import StudentCopula

t_cop_dist = StudentCopula(dim=len(d.columns))
t_cop_dist.fit(pse_u, verbose=0, to_pobs=False)
print(t_cop_dist.params.rho)
print(t_cop_dist.params.df)

# Generate random samples based on t-copula
u_sim_t = pd.DataFrame(t_cop_dist.random(n_sim, seed=4002),
                       columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(u_sim_t[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'blue'})

plt.tight_layout()
plt.savefig(f"../Picture/t_sample_Copula_{period}.png", dpi=200)

# Get back returns based on the random samples
return_sim_t = pd.DataFrame(pseudo_quantile(u_sim_t, returns), 
                            columns=returns.columns)
# only plot the first 1000 samples
sns.pairplot(return_sim_t[1:1000], diag_kind="kde", 
             plot_kws={'alpha': 0.5, 'color': 'green'})

plt.tight_layout()
plt.savefig(f"../Picture/t_Copula_{period}.png", dpi=200)

#%%
sim_t_md2 = Mahalanobis2(return_sim_t)
fig = QQ_Plot(sim_t_md2, returns_md2, col="orange")

fig.tight_layout()
fig.savefig(f"../Picture/Pseudo t QQ Plot_{period}.png", dpi=200)

print(stats.kstest(np.sqrt(returns_md2), np.sqrt(sim_t_md2), 
                   method="asymp"))

#%%
n_days = len(returns)
i = (np.arange(1, n_days + 1) - 0.5) / n_days
q_N = np.quantile(sim_N_md2, i, method='interpolated_inverted_cdf')
q_t = np.quantile(sim_t_md2, i, method='interpolated_inverted_cdf')

a = 15
# find the common index where both coordinates > a
sort_returns_md2 = np.sort(returns_md2)
idx = set.intersection(set(np.where(q_N>a)[0]), 
                       set(np.where(q_t>a)[0]),
                       set(np.where(sort_returns_md2>a)[0]))
idx_start = min(idx)
print(idx_start)

fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
# skip the largest (last) entry
ax.scatter(q_N[idx_start:], sort_returns_md2[idx_start:], 
            color="blue", facecolor="white", marker="o", s=50)
ax.plot(q_t[idx_start:], sort_returns_md2[idx_start:], 
        "x", color="orange", markersize=10)
x = np.linspace(*ax.get_xlim())
ax.plot(x, x, color="black", linewidth=2)
ax.set_xlabel("Bootstrapped quantiles", fontsize=15)
ax.set_ylabel("Sample quantiles", fontsize=15)
ax.set_title("Copula Q-Q Plot", fontsize=20)
ax.legend(["Gaussian copula", "t-copula"], fontsize=15)

fig.tight_layout()
fig.savefig(f"../Picture/Pseudo QQ Plot_{period}.png", dpi=200)

# residuals with respect to the 45-degree line
resid2_N = (sort_returns_md2 - q_N)**2
resid2_t = (sort_returns_md2 - q_t)**2

fig = plt.figure(figsize=(10, 10), dpi=200)
# plot the largest 50 and skip the largest (last) entry
plt.plot(np.sort(resid2_N)[-50:-1], "bo", 
         markerfacecolor="white", markersize=10)
plt.plot(np.sort(resid2_t)[-50:-1], "x", color="orange", markersize=10)
plt.xlabel("Index", fontsize=15)
plt.ylabel("Squared residuals", fontsize=15)
plt.legend(["Gaussian copula", "t-copula"], fontsize=15)

plt.tight_layout()
fig.savefig(f"../Picture/Squared residuals plot 49_{period}.png", dpi=200)

#%%
fig = plt.figure(figsize=(10, 10), dpi=200)
# plot the largest 50
plt.plot(np.sort(resid2_N)[-50:], "bo", 
         markerfacecolor="white", markersize=10)
plt.plot(np.sort(resid2_t)[-50:], "x", color="orange", markersize=10)
plt.xlabel("Index", fontsize=15)
plt.ylabel("Squared residuals", fontsize=15)
plt.legend(["Gaussian copula", "t-copula"], fontsize=15)

plt.tight_layout()
fig.savefig(f"../Picture/Squared residuals plot_{period}.png", dpi=200)

#%%
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
# skip the largest (last) entry
ax.scatter(q_N[idx_start:], sort_returns_md2[idx_start:], 
            color="blue", facecolor="white", marker="o", s=50)
ax.plot(q_t[idx_start:], sort_returns_md2[idx_start:], 
        "x", color="orange", markersize=10)
x = np.linspace(*ax.get_xlim())
ax.plot(x, x, color="black", linewidth=2)
ax.set_xlabel("Bootstrapped quantiles", fontsize=15)
ax.set_ylabel("Sample quantiles", fontsize=15)
ax.set_title("Copula Q-Q Plot", fontsize=20)
ax.legend(["Gaussian copula", "t-copula"], fontsize=15)

b_N, w_N = np.linalg.lstsq(np.vstack([np.ones(n_days), q_N]).T, 
                           sort_returns_md2, rcond=None)[0]
ax.plot(x, w_N*x + b_N, color="blue", linewidth=2)

b_t, w_t = np.linalg.lstsq(np.vstack([np.ones(n_days), q_t]).T, 
                           sort_returns_md2, rcond=None)[0]
ax.plot(x, w_t*x + b_t, color="orange", linewidth=2)

fig.tight_layout()
fig.savefig(f"../Picture/Pseudo QQ Plot with references_{period}.png", dpi=200)