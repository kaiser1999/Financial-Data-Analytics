import pandas as pd
import numpy as np

d = pd.read_csv("stock_1999_2002.csv", index_col=0)

x_n = d.iloc[-1,:]           # select the last obs
w = [40000, 30000, 30000]    # investment amount on each stock
p_0 = sum(w)	             # total investment amount
w_s = w/x_n                  # no. of shares bought at day n

h_sim = (d/d.shift(1) * x_n)[1:]
p_n = h_sim @ w_s            # portfolio value at day n

loss = p_0 - p_n	         # loss
VaR_sim = np.quantile(loss, 0.99) # 1-day 99% V@R
print(VaR_sim)

#%%

from arch import arch_model

d = pd.read_csv("stock_1999_2002.csv", index_col=0) # read in data file
t = d["HSBC"]               # select HSBC
n = len(d)		            # no. of obs
x_n = t.iloc[-1]            # select the last obs
ns = n-1                    # number of scenarios
u = np.array((t-t.shift(1))/t.shift(1))[1:]  # stock returns

# fit the GARCH(1,1) model
res_HSBC = arch_model(u, mean='Zero', vol='GARCH', p=1, q=1).fit()
omega, alpha, beta = res_HSBC.params.values
nu = [omega/(1 - alpha - beta)] # long run variance
for i in range(1, len(u)):
    nu.append(omega + alpha*u[i-1]**2 + beta*nu[-1])
    
p_0 = 100000                # initial portfolio value
w_s = p_0/x_n               # shares owned on day n

# Fitted variance on day n 
var_n = omega + alpha*u[-1]**2 + beta*nu[-1]

t_i = np.array(t[1:ns])
t_i_1 = np.array(t[0:(ns-1)])
var_i = nu[1:ns]

h_sim = x_n*(t_i_1+(t_i-t_i_1)*np.sqrt(var_n/var_i))/t_i_1

p_n = h_sim * w_s           # portfolio value
loss_GARCH = p_0 - p_n      # loss
VaR_GARCH = np.quantile(loss_GARCH, 0.99)   # 1-day 99% VaR
print(VaR_GARCH)

#%%

from scipy.optimize import minimize, Bounds

# set lower and upper bounds for omega, alpha, beta
bds = Bounds(lb=[1e-15, 1e-15, 1e-15], ub=[1-1e-15, 1-1e-15, 1-1e-15])
bds_2 = [(1e-15,1-1e-15),(1e-15,1-1e-15),(1e-15, 1-1e-15)]
# stationarity constraint: omega + alpha + beta < 1
cons = [{'type': 'ineq', 'fun': lambda x: 1-x[1]-x[2]}]

def GARCH_11(x, r):
    omega, alpha, beta = x
    nu = omega + alpha*np.mean(r**2) + beta*np.mean(r**2) # nu_1
    log_like = -1/2*(np.log(2*np.pi) + np.log(nu) + r[0]**2/nu)
    for i in range(1, len(r)):
        nu = omega + alpha*r[i-1]**2 + beta*nu
        log_like -= 1/2*(np.log(2*np.pi) + np.log(nu) + r[i]**2/nu)
    return -log_like

d = pd.read_csv("stock_1999_2002.csv") # read in data file
t = d["HSBC"]               # select HSBC
n = len(d)		            # no. of obs
x_n = t.iloc[-1]            # select the last obs
ns = n-1                    # number of scenarios
u = np.array((t-t.shift(1))/t.shift(1))[1:]  # stock returns

# fit the GARCH(1,1) model with two stage optimization
x0 = [0.05, 0.04, 0.9]
model1 = minimize(GARCH_11, x0, args=(u), method='L-BFGS-B', bounds=bds, tol=1e-20)
model2 = minimize(GARCH_11, model1.x, args=(u), method='SLSQP', constraints=cons, bounds=bds_2, tol=1e-20)

omega, alpha, beta = model2.x
nu = [omega + alpha*np.mean(u**2) + beta*np.mean(u**2)]
for i in range(1, len(u)):
    nu.append(omega + alpha*u[i-1]**2 + beta*nu[-1])
    
p_0 = 100000                # initial portfolio value
w_s = p_0/x_n               # shares owned on day n

# Fitted variance on day n 
var_n = omega + alpha*u[-1]**2 + beta*nu[-1]

t_i = np.array(t[1:ns])
t_i_1 = np.array(t[0:(ns-1)])
var_i = nu[1:ns]

h_sim = x_n*(t_i_1+(t_i-t_i_1)*np.sqrt(var_n/var_i))/t_i_1

p_n = h_sim * w_s           # portfolio value
loss_GARCH = p_0 - p_n      # loss
VaR_GARCH = np.quantile(loss_GARCH, 0.99)   # 1-day 99% VaR
print(VaR_GARCH)

#%%

from scipy.stats import norm

d = pd.read_csv("stock_1999_2002.csv", index_col=0)
u = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return
S = np.cov(u, rowvar=False)	            # sample cov. matrix
w = np.array([40000, 30000, 30000])     # investment amount on each stock
delta_p = u @ w	                        # Delta P
sd_p = np.std(delta_p, ddof=1)		    # sample sd of portfolio (empirical)
VaR_N = norm.ppf(0.99)*sd_p	            # 1-day 99% V@R with normal
print(VaR_N)
print(norm.ppf(0.99)*np.sqrt(w.T @ S @ w)) # z x sqrt(w.T S w)

#%%
from scipy.stats import t

ku = sum((delta_p/sd_p)**4)/len(delta_p)-3
nu = round(6/ku+4)
VaR_t = t.ppf(0.99, nu)*sd_p*np.sqrt((nu-2)/nu) # 1-day 99% V@R with t
print(VaR_t)

#%%

from scipy.optimize import minimize

u = 3.2                                 # threshold value
m = np.mean(loss)                       # mean loss
s = np.std(loss)		                # sd loss
z = (loss-m)/s		                    # standardize loss
z_u = z[z>u]                            # select z>u
n_u = len(z_u)                          # no. of z>u
print(n_u)

def n_log_lik(theta, y):                # theta=(xi, beta)
    xi, beta = theta
    return(len(y)*np.log(beta)+(1/xi+1)*sum(np.log(1+xi*y/beta)))

theta_0 = [0.2, 0.01]                   # initial theta_0
# min -ve log_likelihood
res = minimize(n_log_lik, theta_0, method='Nelder-Mead', args=(z_u-u,))
xi, beta = res.x                        # MLE theta=(xi, beta)
print(res.x)
print(-res.fun)                         # max value
q = 0.99
VaR = u+(beta/xi)*((len(z)*(1-q)/n_u)**(-xi)-1)
print(VaR)
VaR_EVT = m+VaR*s                       # 1day 99% V@R by EVT
print(VaR_EVT)

#%%
from scipy.stats import binom

m = np.arange(11)
print(np.round(1-binom.cdf(m, 250, 0.01), 4))

#%%

x_n = d.iloc[-1,:]                      # select the last obs
w = np.array([40000, 30000, 30000])     # investment amount on each stock
p_0 = sum(w)                            # total investment amount
w_s = w/x_n                             # no. of shares bought at day n

ns = 250                                # 250 days
x_250 = d.iloc[-ns:,:]                  # recent 250 days
ps_250 = x_250 @ w_s.T                  # portfolio value
ps_250 = np.append(ps_250, p_0)	        # add total amount 
loss_250 = -np.diff(ps_250, axis=0)     # compute daily loss

print(sum(loss_250 > VaR_sim))          # no. of exceptions
print(sum(loss_250 > VaR_GARCH))
print(sum(loss_250 > VaR_N))
print(sum(loss_250 > VaR_t))
print(sum(loss_250 > VaR_EVT))

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 13, 10
plt.rcParams.update({'font.size': 15})

plt.hist(-loss_250, ec='black', bins=50, alpha=0.8)
plt.xlim(-4500, 3200)
plt.axvline(x=-VaR_sim, c="blue")
plt.axvline(x=-VaR_GARCH, c="red")
plt.axvline(x=-VaR_N, c="green")
plt.axvline(x=-VaR_t, c="gray")
plt.axvline(x=-VaR_EVT, c="orange")

plt.text(-VaR_sim, 1.5, "-VaR_sim", ha='center', fontsize="small")
plt.text(-VaR_GARCH, 1.5, "-VaR_GARCH", ha='center', fontsize="small")
plt.text(-VaR_N, 15, "-VaR_N", ha='center', fontsize="small")
plt.text(-VaR_t, 8, "-VaR_t", ha='center', fontsize="small")
plt.text(-VaR_EVT, 15, "-VaR_EVT", ha='center', fontsize="small")

plt.tight_layout()
plt.savefig("../Picture/Backtesting VaR.png", dpi=200)

#%%

# expected shortfall
print(np.mean(loss[loss > VaR_sim]))    # expected shortfall
print(np.mean(loss[loss > VaR_GARCH]))
print(np.mean(loss_GARCH[loss_GARCH > VaR_GARCH]))

mu = np.mean(-delta_p)
sig = np.std(-delta_p, ddof=1)
# normal
from scipy.stats import norm
print(mu + sig/0.01*norm.pdf(norm.ppf(0.01)))

# student t
from scipy.stats import t
Term1 = (nu + t.ppf(0.01, nu)**2)/(nu-1)
Term2 = t.pdf(t.ppf(0.01, nu), nu)/0.01
print(mu + sig*np.sqrt((nu-2)/nu)*Term1*Term2)

# extreme value theorem
EVT = VaR + (beta + xi*(VaR - u))/(1 - xi)
print(mu + sig*EVT)

# distribution free
K = int(np.floor(n*0.01))
sort_loss = np.sort(loss)[::-1]
Term1 = 1/(0.01*n)*sum(sort_loss[:(K-1)])
Term2 = (1-(K-1)/(0.01*n))*sort_loss[K-1]
print(Term1 + Term2)