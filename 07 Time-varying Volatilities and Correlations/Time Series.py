import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

d = pd.read_csv("../Datasets/stock_1999_2002.csv", index_col=0)
u = np.diff(d, axis=0) / d.iloc[:-1, :] # Arithmetic return
u.columns = d.columns.values + "_Return"

#%%
# compute 90-day moving sd and 180-day moving sd
u1, u2, u3 = u["HSBC_Return"], u["CLP_Return"], u["CK_Return"]

s_90, s_180 = u.rolling(90).std(ddof=1), u.rolling(180).std(ddof=1)

fig, ax = plt.subplots(figsize=(15,8))
ax.plot(s_90["HSBC_Return"], c="blue", label="s_90")
ax.plot(s_180["HSBC_Return"], c="red", label="s_180")
ax.set_title("Simple moving standard deviation of HSBC", fontsize=20)
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.legend(fontsize=15)

fig.tight_layout()
fig.savefig("../Picture/HSBC moving sd.png", dpi=200)

#%%
from arch import arch_model

res_HSBC = arch_model(u1, mean='Zero', vol='GARCH', p=1, q=1).fit()
print(res_HSBC.params.values)
print(res_HSBC.loglikelihood) # compute log-likelihood value

#%%
omega, alpha, beta = res_HSBC.params.values
print(omega / (1 - alpha - beta))

#%%
print(res_HSBC.summary())
#res_HSBC.plot()

#%%
from scipy.optimize import minimize
from scipy.optimize import Bounds

# set lower and upper bounds for omega, alpha, beta
bds = Bounds(lb=[1e-15,]*3, ub=[np.inf] + [1-1e-15,]*2)
# stationarity constraint: alpha + beta < 1
cons = [{'type': 'ineq', 'fun': lambda x: 1-x[1]-x[2]}]

def GARCH_11(x, r):
    omega, alpha, beta = x
    nu = omega + alpha*np.mean(r**2) + beta*np.mean(r**2) # nu_1
    log_like = -1/2*(np.log(2*np.pi) + np.log(nu) + r[0]**2/nu)
    for i in range(1, len(r)):
        nu = omega + alpha*r[i-1]**2 + beta*nu
        log_like -= 1/2*(np.log(2*np.pi) + np.log(nu) + r[i]**2/nu)
    return -log_like

def GARCH_11_MLE(r):
    x0 = [0.05, 0.04, 0.9]
    
    # two stage optimization: L-BFGS-B only support bounded constraints
    model1 = minimize(GARCH_11, x0, args=(r), method='L-BFGS-B', 
                      bounds=bds, tol=1e-20)
    model2 = minimize(GARCH_11, model1.x, args=(r), method='SLSQP', 
                      constraints=cons, bounds=bds, tol=1e-20)
    return model2.x, model2.fun			# negative log-likelihood

model_HSBC, llike_HSBC = GARCH_11_MLE(u1)
print(model_HSBC, llike_HSBC)

#%%
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# Residuals u_i/sigma_i ~ N(0, 1)
omega, alpha, beta = model_HSBC
nu = [omega + alpha*np.mean(u1**2) + beta*np.mean(u1**2)]
for i in range(1, len(u1)):
    nu.append(omega + alpha*u1[i-1]**2 + beta*nu[-1])

resid_HSBC = u1/np.sqrt(nu)

fig, axs = plt.subplots(2, 2, figsize=(15,15))
vol = pd.Series(np.sqrt(nu), index=d.index[1:])
axs[0,0].plot(vol, c="blue")
axs[0,0].set_title("Conditional SD")
axs[0,0].xaxis.set_major_locator(ticker.MultipleLocator(400))
plot_acf(u1**2, alpha=0.05, c="blue", 
         title="ACF Plot of Squared HSBC Return", ax=axs[0,1])
plot_acf(resid_HSBC**2, alpha=0.05, c="blue", 
         title="ACF Plot of Squared Residuals", ax=axs[1,0])
stats.probplot(resid_HSBC, dist="norm", plot=axs[1,1])
axs[1,1].set_title("QQ Plot of Standardized Residuals")
fig.tight_layout()
fig.savefig("../Picture/HSBC diagnosis.png", dpi=200)

from statsmodels.stats.diagnostic import acorr_ljungbox

print(acorr_ljungbox(u1**2, lags=[15]))
print(acorr_ljungbox(np.array(resid_HSBC)**2, lags=[15]))

#%%
sig = pd.Series(np.sqrt(nu), index=d.index[1:])
df = pd.concat([sig, s_90["HSBC_Return"], s_180["HSBC_Return"]], 
               axis=1)
df.columns = ["nu", "s_90", "s_180"]

fig, axs = plt.subplots(1, 1, figsize=(15,8))
df.plot(ax=axs, color=["green", "blue", "red"])
axs.xaxis.set_major_locator(ticker.MultipleLocator(200))
axs.set_title("HSBC volatilities", fontsize=20)

fig.tight_layout()
fig.savefig("../Picture/HSBC GARCH sd.png", dpi=200)

#%%
print(u.iloc[-1, :])
print(np.corrcoef(u.iloc[-90:, :], rowvar=False))
print(np.cov(u.iloc[-90:, :], rowvar=False))

#%%
'''
test = arch_model(u1, mean='Zero', vol='GARCH', p=1, q=1, dist="t").fit()
print(test.params.values)
print(test.summary())
#test.plot(axs=ax)
'''
#%%
# Numbers in the book
omega, alpha, beta = model_HSBC

print(nu[-1])
print(np.sqrt(nu[-1]))

V_L = omega/(1-alpha-beta)
print(V_L)

gamma = -np.log(alpha + beta)
T = 10
future_nu = V_L + (1-np.exp(-10*gamma))/(10*gamma)*(nu[-1]-V_L)
print(future_nu)
print(np.sqrt(future_nu))
print(np.sqrt(252*future_nu))

#%%
model_CLP, llike_CLP = GARCH_11_MLE(u2)
model_CK, llike_CK = GARCH_11_MLE(u3)

coef = np.array([model_HSBC, model_CLP, model_CK])
print(coef)
print(np.mean(coef, axis=0))

#%%
# Numbers in the book
omega, alpha, beta = np.mean(coef, axis=0)
u1T, u2T, u3T = u.iloc[-1,:]
Var_MAT = np.cov(u.iloc[-90:, :], rowvar=False)
sig_11 = omega + alpha*u1T**2 + beta*Var_MAT[0,0]
sig_22 = omega + alpha*u2T**2 + beta*Var_MAT[1,1]
sig_33 = omega + alpha*u3T**2 + beta*Var_MAT[2,2]
sig_12 = omega + alpha*u1T*u2T + beta*Var_MAT[0,1]
sig_13 = omega + alpha*u1T*u3T + beta*Var_MAT[0,2]
sig_23 = omega + alpha*u2T*u3T + beta*Var_MAT[1,2]
rho_12 = sig_12/np.sqrt(sig_11*sig_22)
rho_13 = sig_13/np.sqrt(sig_11*sig_33)
rho_23 = sig_23/np.sqrt(sig_22*sig_33)

print(u.iloc[-1,:])
print(Var_MAT)

print(f'{sig_11:.8f}')
print(f'{sig_22:.8f}')
print(f'{sig_33:.8f}')
print(f'{sig_12:.8f}')
print(f'{sig_13:.8f}')
print(f'{sig_23:.8f}')
print(f'{rho_12:.8f}')
print(f'{rho_13:.8f}')
print(f'{rho_23:.8f}')

#%%
L_bds = [(1e-15, np.inf), (1e-15, 1-1e-15), (1e-15, 1-1e-15), 
         (1e-15, 1-1e-15)]

def L_GARCH_11(x, r):
    omega, alpha, beta, theta = x
    nu = np.mean(r**2)  # following rugarch in R
    log_like = -1/2*(np.log(2*np.pi) + np.log(nu) + r[0]**2/nu)
    for i in range(1, len(r)):
        nu = omega + alpha*r[i-1]**2 + beta*nu + theta*r[i-1]**2*(r[i-1]<0)
        log_like -= 1/2*(np.log(2*np.pi) + np.log(nu) + r[i]**2/nu)
    return -log_like

def L_GARCH_11_MLE(r):
    # stationarity constraint: alpha + beta + theta*gamma < 1
    fun = lambda x: 1-x[1]-x[2]-x[3]*np.mean(r < 0)
    L_cons = [{'type': 'ineq', 'fun': fun}]
    
    x0 = [0.05, 0.03, 0.9, 0.005]
    model1 = minimize(L_GARCH_11, x0, args=(r), method='L-BFGS-B', 
                      bounds=L_bds, tol=1e-20)
    model2 = minimize(L_GARCH_11, model1.x, args=(r), method='SLSQP', 
                      constraints=L_cons, bounds=L_bds, tol=1e-20)
    return model2.x, model2.fun

print(L_GARCH_11_MLE(u1))   # HSBC
print(L_GARCH_11_MLE(u2))   # CLP
print(L_GARCH_11_MLE(u3))   # CK

#%%
# set lower and upper bounds for theta_1, theta_2
DCC_bds = Bounds(lb=[1e-15, 1e-15], ub=[1-1e-15, 1-1e-15])
# stationarity constraint: theta_1 + theta_2 < 1
DCC_cons = [{'type': 'ineq', 'fun': lambda x: 1-sum(x)}]
from numpy.linalg import det, inv

def DCC_GARCH_11(x, eta):
    theta_1, theta_2 = x
    
    bar_Sigma = np.cov(eta, rowvar=False)
    Q = bar_Sigma
    R = np.diag(np.diag(Q)**(-1/2)) @ Q @ np.diag(np.diag(Q)**(-1/2))
    func = np.log(det(R)) + eta[1,:] @ inv(R) @ eta[1,:]
    for i in range(1, np.shape(eta)[0]):
        z = eta[i-1,:].reshape(-1, 1)
        Q = ((1-theta_1-theta_2)*bar_Sigma + theta_2*Q + theta_1*z @ z.T)
        R = np.diag(np.diag(Q)**(-1/2)) @ Q @ np.diag(np.diag(Q)**(-1/2))
        
        func += np.log(det(R)) + eta[i,:] @ inv(R) @ eta[i,:]
      
    return func

def resid(x, r):
    omega, alpha, beta = x
    nu = [omega + alpha*np.mean(r**2) + beta*np.mean(r**2)]
    for i in range(1, len(r)):
        nu.append(omega + alpha*r[i-1]**2 + beta*nu[-1])
    
    return r/np.sqrt(nu)

eta = np.array([resid(model_HSBC, u1), resid(model_CLP, u2), 
                resid(model_CK, u3)]).T

x0 = [0.1, 0.4]
model1 = minimize(DCC_GARCH_11, x0, args=(eta), method='L-BFGS-B', 
                  bounds=DCC_bds, tol=1e-20)
model2 = minimize(DCC_GARCH_11, model1.x, args=(eta), method='SLSQP', 
                  constraints=DCC_cons, bounds=DCC_bds, tol=1e-20)
print(model2.x)

#%%
# Example of using R code in Python
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

rugarch = importr('rugarch')

garch_spec = rugarch.ugarchspec(
            mean_model=robjects.r('list(armaOrder=c(0, 0), include.mean=FALSE)'),
            variance_model=robjects.r('list(model="eGARCH", garchOrder=c(1,1))'),
            distribution_model='norm')
        # Used to convert training set to R list for model input
numpy2ri.activate()
# Train R GARCH model on returns as %
garch_fitted = rugarch.ugarchfit(spec=garch_spec, data=np.array(u1))


EstMdl_R = np.array(garch_fitted.slots['fit'].rx2('coef'))

print(EstMdl_R)

garch_fitted = rugarch.ugarchfit(spec=garch_spec, data=np.array(u2))

EstMdl_R = np.array(garch_fitted.slots['fit'].rx2('coef'))

print(EstMdl_R)

garch_fitted = rugarch.ugarchfit(spec=garch_spec, data=np.array(u3))

EstMdl_R = np.array(garch_fitted.slots['fit'].rx2('coef'))

print(EstMdl_R)

numpy2ri.deactivate()