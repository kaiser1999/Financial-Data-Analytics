import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

d = pd.read_csv("stock_1999_2002.csv")
u = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return

d.plot(subplots=True, layout=(3,1), figsize=(10, 10))
u.plot(subplots=True, layout=(3,1), sharey=True, figsize=(10, 10))

fig, axs = plt.subplots(3, 2, figsize=(10,15))
axs[0,0].hist(u.HSBC, color="blue", ec='black', bins="sturges")
axs[0,0].set_title("Histogram of HSBC Return")
stats.probplot(u.HSBC, dist="norm", plot=axs[0,1])
axs[0,1].set_title("Normal Q-Q Plot of HSBC Return")
axs[0,1].get_lines()[0].set_color('blue')

axs[1,0].hist(u.CLP, color="orange", ec='black', bins="sturges")
axs[1,0].set_title("Histogram of CLP Return")
stats.probplot(u.CLP, dist="norm", plot=axs[1,1])
axs[1,1].set_title("Normal Q-Q Plot of CLP Return")
axs[1,1].get_lines()[0].set_color('orange')

axs[2,0].hist(u.CK, color="green", ec='black', bins="sturges")
axs[2,0].set_title("Histogram of CK Return")
stats.probplot(u.CK, dist="norm", plot=axs[2,1])
axs[2,1].set_title("Normal Q-Q Plot of CK Return")
axs[2,1].get_lines()[0].set_color('green')
fig.tight_layout()

#%%

print(stats.shapiro(u.HSBC))
print(stats.shapiro(u.CLP))
print(stats.shapiro(u.CK))

#%%
print(stats.kstest(u.HSBC, stats.norm.cdf, 
                   args=(np.mean(u.HSBC), np.std(u.HSBC))))
print(stats.kstest(u.CLP, stats.norm.cdf,
                   args=(np.mean(u.CLP), np.std(u.CLP))))
print(stats.kstest(u.CK, stats.norm.cdf,
                   args=(np.mean(u.CK), np.std(u.CK))))

#%%

def JB_test(u):
    z = u - np.mean(u)  # Remove mean
    n = len(z) # Sample size
    s = np.std(z) # Population standard deviation
    sk = sum(z**3) / (n*s**3) # Skewness
    ku = sum(z**4) / (n*s**4) - 3 # Excess Kurtosis
    JB = n * (sk**2/6 + ku**2/24) # JB test statistics
    p = 1 - stats.chi2.cdf(JB, 2) # chi-squared p-value
    return ({"JB-test": JB, "p-value": p})

print(stats.jarque_bera(u.HSBC))
print(JB_test(u.HSBC))

print(stats.jarque_bera(u.CLP))
print(JB_test(u.CLP))

print(stats.jarque_bera(u.CK))
print(JB_test(u.CK))

#%%

def QQt_plot(u, color="blue", comp="", ax=None):
    z = u - np.mean(u)  # Remove mean
    sz = np.sort(z) # Sort z
    n = len(z) # Sample size
    s = np.std(z) # Population standard deviation
    ku = sum(z**4) / (n*s**4) - 3 # Excess Kurtosis
    nu = 6/ku + 4 # degrees of freedom
    i = (np.arange(1, n+1)-0.5)/n # create a vector of percentile
    q = stats.t.ppf(i, nu)
    
    b, w = np.linalg.lstsq(np.vstack([np.ones(n), q]).T, sz, 
                           rcond=None)[0]
    ax.scatter(q, sz, color=color)
    ax.plot(q, w*q + b, color="red")
    ax.set_title(f"Self-defined t Q-Q Plot of {comp} Return")
    return nu

fig, axs = plt.subplots(3, 2, figsize=(10,15))
df_HSBC = QQt_plot(u.HSBC, color="blue", comp="HSBC", ax=axs[0,0])
stats.probplot(u.HSBC, dist="t", sparams=df_HSBC, plot=axs[0,1])
axs[0,1].set_title("t Q-Q Plot of HSBC Return")
axs[0,1].get_lines()[0].set_color('blue')

df_CLP = QQt_plot(u.CLP, color="orange", comp="CLP", ax=axs[1,0])
stats.probplot(u.CLP, dist="t", sparams=df_CLP, plot=axs[1,1])
axs[1,1].set_title("t Q-Q Plot of CLP Return")
axs[1,1].get_lines()[0].set_color('orange')

df_CK = QQt_plot(u.CK, color="green", comp="CK", ax=axs[2,0])
stats.probplot(u.CK, dist="t", sparams=df_CK, plot=axs[2,1])
axs[2,1].set_title("t Q-Q Plot of CK Return")
axs[2,1].get_lines()[0].set_color('green')
fig.tight_layout()

#%%
print([df_HSBC, df_CLP, df_CK])

print(stats.kstest(u.HSBC, stats.t.cdf, args=(df_HSBC,)))
print(stats.kstest(u.CLP, stats.t.cdf, args=(df_CLP,)))
print(stats.kstest(u.CK, stats.t.cdf, args=(df_CK,)))

#%%
n = 180
u_180 = u.iloc[len(u)-n:, :]
mu_180 = np.mean(u_180)
S_180 = np.cov(u_180, rowvar=False)

z_180 = (u_180 - mu_180).values.reshape(n, -1)
d2_180 = np.diagonal(z_180 @ np.linalg.inv(S_180) @ z_180.T)
sd2_180 = np.sort(d2_180)
i = (np.arange(1, n+1)-0.5)/n
q = stats.chi2.ppf(i, 3)

print(mu_180)
print(z_180)
print(d2_180)

fig = plt.figure()
b, w = np.linalg.lstsq(np.vstack([np.ones(n), q]).T, sd2_180, 
                       rcond=None)[0]
plt.scatter(q, sd2_180, color="blue")
plt.plot(q, w*q + b, color="blue")
plt.title("Chi2 Q-Q Plot")

print(stats.kstest(d2_180, stats.chi2.cdf, args=(3,)))

#%%

print(np.corrcoef(u_180, rowvar=False))

#%%
hist_kwds = {'color':'blue', 'bins':'sturges', 'ec':'black'}
axes = pd.plotting.scatter_matrix(u, figsize=(10,10), color="blue",
                                  hist_kwds=hist_kwds)
new_labels = [round(float(i.get_text()), 2) for i in 
              axes[0,0].get_yticklabels()]
axes[0,0].set_yticklabels(new_labels)
plt.tight_layout()

#%%

fig, axs = plt.subplots(4, 3, figsize=(15,20))
axs[0,0].hist(d.HSBC, color="blue", ec='black', bins="sturges")
axs[0,0].set_title("Histogram of HSBC Price")
stats.probplot(d.HSBC, dist="norm", plot=axs[1,0])
axs[1,0].set_title("Normal Q-Q Plot of HSBC Price")
axs[1,0].get_lines()[0].set_color('blue')
pd.plotting.lag_plot(d.HSBC, lag=1, ax=axs[2,0], c="blue")
axs[2,0].set_title("1-day Lagged Plot of HSBC Price")
pd.plotting.lag_plot(u.HSBC, lag=1, ax=axs[3,0], c="blue")
axs[3,0].set_title("1-day Lagged Plot of HSBC Return")

axs[0,1].hist(d.CLP, color="orange", ec='black', bins="sturges")
axs[0,1].set_title("Histogram of CLP Price")
stats.probplot(d.CLP, dist="norm", plot=axs[1,1])
axs[1,1].set_title("Normal Q-Q Plot of CLP Price")
axs[1,1].get_lines()[0].set_color('orange')
pd.plotting.lag_plot(d.CLP, lag=1, ax=axs[2,1], c="orange")
axs[2,1].set_title("1-day Lagged Plot of CLP Price")
pd.plotting.lag_plot(u.CLP, lag=1, ax=axs[3,1], c="orange")
axs[3,1].set_title("1-day Lagged Plot of CLP Return")

axs[0,2].hist(d.CK, color="green", ec='black', bins="sturges")
axs[0,2].set_title("Histogram of CK Price")
stats.probplot(d.CK, dist="norm", plot=axs[1,2])
axs[1,2].set_title("Normal Q-Q Plot of CK Price")
axs[1,2].get_lines()[0].set_color('green')
pd.plotting.lag_plot(d.CK, lag=1, ax=axs[2,2], c="green")
axs[2,2].set_title("1-day Lagged Plot of CK Price")
pd.plotting.lag_plot(u.CK, lag=1, ax=axs[3,2], c="green")
axs[3,2].set_title("1-day Lagged Plot of CK Return")
plt.tight_layout()

#%%

from statsmodels.graphics.tsaplots import plot_acf
fig, axs = plt.subplots(3, 3, figsize=(15,15))
plot_acf(d.HSBC, alpha=None, c="blue", title="ACF Plot of HSBC Price", ax=axs[0,0])
plot_acf(u.HSBC, alpha=None, c="blue", title="ACF Plot of HSBC Return", ax=axs[1,0])
plot_acf(u.HSBC**2, alpha=None, c="blue", title="ACF Plot of Squared HSBC Return", ax=axs[2,0])

plot_acf(d.CLP, alpha=None, c="orange", title="ACF Plot of CLP Price", ax=axs[0,1])
plot_acf(u.CLP, alpha=None, c="orange", title="ACF Plot of CLP Return", ax=axs[1,1])
plot_acf(u.CLP**2, alpha=None, c="orange", title="ACF Plot of Squared CLP Return", ax=axs[2,1])

plot_acf(d.CK, alpha=None, c="green", title="ACF Plot of CK Price", ax=axs[0,2])
plot_acf(u.CK, alpha=None, c="green", title="ACF Plot of CK Return", ax=axs[1,2])
plot_acf(u.CK**2, alpha=None, c="green", title="ACF Plot of Squared CK Return", ax=axs[2,2])

#%%
np.random.seed(4012)
mu_180 = np.mean(u_180)
S_180 = np.cov(u_180, rowvar=False)
C_180 = np.linalg.cholesky(S_180)
s0 = d.iloc[-1,:]
s_pred = []
for i in range(90):
    z = np.random.randn(3)
    v = mu_180 + C_180.T @ z
    s1 = s0 * (1 + v)
    s_pred.append(s1.values)
    s0 = s1

df_pred = pd.DataFrame(np.array(s_pred), columns=d.columns.values + "_pred")
df_pred.index = np.arange(len(d)+1, len(d)+90+1)

pd.merge(d, df_pred, how='outer', left_index=True, 
         right_index=True).plot(figsize=(10, 7))

