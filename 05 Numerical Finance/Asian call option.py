import numpy as np
import pandas as pd

# Initialize variables
S_0 = 10; K = 8; r = 0.05; sigma = 0.3; T = 1
M_lst = [1e1, 1e2, 1e3]; n = 1e5

def Sim_Asian(n, M, S_0, K, r, sigma, T, theta):
    delta_t = T / M
    
    def Euler(z, S_t, theta):
        return S_t*(1 + r*delta_t) + sigma*S_t**(theta/2)*np.sqrt(delta_t)*z
    
    def Milstein(z, S_t, theta):
        return (Euler(z, S_t, theta) + 
                sigma**2*theta/2*S_t**(theta-1)*delta_t/2*(z**2-1))
    
    def Exact(z, S_t, theta=2):
        return S_t*np.exp((r - sigma**2/2)*delta_t + sigma*np.sqrt(delta_t)*z)
    
    # Asian_call
    S_Eul = S_0; avg_Eul = S_0 / (M+1)
    S_Mil = S_0; avg_Mil = S_0 / (M+1)
    if (theta == 2):
        S_Ext = S_0; avg_Ext = S_0 / (M+1)
    else:
        S_Ext = 0; avg_Ext = 0
    
    for m in range(int(M)):
        z = np.random.normal(size=int(n))
        S_Eul = Euler(z, S_Eul, theta)
        avg_Eul += S_Eul/(M+1)
        
        S_Mil = Milstein(z, S_Mil, theta)
        avg_Mil += S_Mil/(M+1)
      
        if (theta == 2):
            S_Ext = Exact(z, S_Ext, theta)
            avg_Ext += S_Ext/(M+1)
    
    return {"Eul": {"price": S_Eul, "payoff": np.exp(-r*T)*np.maximum(avg_Eul-K, 0)}, 
            "Mil": {"price": S_Mil, "payoff": np.exp(-r*T)*np.maximum(avg_Mil-K, 0)}, 
            "Ext": {"price": S_Ext, "payoff": np.exp(-r*T)*np.maximum(avg_Ext-K, 0)}}

#%%
import matplotlib.pyplot as plt

BS_results = pd.DataFrame({"M": M_lst, "n": n, "Asian_Eul": 0.0, 
                           "Asian_Mil": 0.0, "Asian_Ext": 0.0})
for i in range(len(M_lst)):
    M = M_lst[i]
    np.random.seed(4002)
    Asian_BS = Sim_Asian(n, M, S_0, K, r, sigma, T, 2)
    BS_results.iloc[i, -3] = np.mean(Asian_BS["Eul"]["payoff"])
    BS_results.iloc[i, -2] = np.mean(Asian_BS["Mil"]["payoff"])
    BS_results.iloc[i, -1] = np.mean(Asian_BS["Ext"]["payoff"])
    
    Eul_diff = Asian_BS["Ext"]["price"] - Asian_BS["Eul"]["price"]
    Mil_diff = Asian_BS["Ext"]["price"] - Asian_BS["Mil"]["price"]
    print(np.ptp(Eul_diff), np.ptp(Mil_diff), np.ptp(Eul_diff)/np.ptp(Mil_diff))
    
    fig, axes = plt.subplots(ncols=2, figsize=(16,7))
    axes[0].hist(Eul_diff, alpha=0.7, color="blue", ec='black', bins="sturges")
    axes[0].set_title("Euler Scheme", fontsize=15)
    axes[1].hist(Mil_diff, alpha=0.7, color="red", ec='black', bins="sturges")
    axes[1].set_title("Milstein Scheme", fontsize=15)
    
    #fig.tight_layout()
    #fig.savefig(f"../Picture/Asian option M_{int(M)}.png", dpi=200)

print(BS_results)

#%%
M = 1e3; n_lst = [1e3, 1e4, 1e5, 1e6, 1e7]
BS_results = pd.DataFrame({"M": M, "n": n_lst, "Asian_Eul": 0.0, 
                           "Asian_Mil": 0.0, "Asian_Ext": 0.0})

for i in range(len(n_lst)):
    n = n_lst[i]
    np.random.seed(4002)
    Asian_BS = Sim_Asian(n, M, S_0, K, r, sigma, T, 2)
    BS_results.iloc[i, -3] = np.mean(Asian_BS["Eul"]["payoff"])
    BS_results.iloc[i, -2] = np.mean(Asian_BS["Mil"]["payoff"])
    BS_results.iloc[i, -1] = np.mean(Asian_BS["Ext"]["payoff"])
    
    Eul_diff = Asian_BS["Ext"]["price"] - Asian_BS["Eul"]["price"]
    Mil_diff = Asian_BS["Ext"]["price"] - Asian_BS["Mil"]["price"]
    print(np.ptp(Eul_diff), np.ptp(Mil_diff), np.ptp(Eul_diff)/np.ptp(Mil_diff))

print(BS_results)

#%%
from tqdm import trange
pd.set_option('display.max_columns', None)  # print all columns

def control_variate(x, y, mu_y):
    s_cov = np.cov(x, y, ddof=1)[0][1]
    s_var = np.var(y, ddof=1)
    return np.mean(x) - s_cov/s_var * (np.mean(y) - mu_y)

theta = 1; epochs = 100
CEV_results = pd.DataFrame({"M": M_lst, "n": n, "theta": theta, 
                            "mu_Eul": 0.0, "mu_Mil": 0.0,
                            "mu_Eul_cv": 0.0, "mu_Mil_cv": 0.0,
                            "sd_Eul": 0.0, "sd_Mil": 0.0,
                            "sd_Eul_cv": 0.0, "sd_Mil_cv": 0.0,})
np.random.seed(4002)
benchmark = Sim_Asian(1e6, 1e4, S_0, K, r, sigma, T, theta)
Asian_Eul, Asian_Mil = np.empty(epochs), np.empty(epochs)
Asian_Eul_cv, Asian_Mil_cv = np.empty(epochs), np.empty(epochs)
mu_y = np.mean(benchmark["Eul"]["price"])
for i in range(len(M_lst)):
    M = M_lst[i]
    np.random.seed(4002)
    for j in trange(epochs, desc=f"M = {M}"):
        Asian_theta = Sim_Asian(n, M, S_0, K, r, sigma, T, theta)
        
        Asian_Eul[j] = np.mean(Asian_theta["Eul"]["payoff"])
        Asian_Mil[j] = np.mean(Asian_theta["Mil"]["payoff"])
        Asian_Eul_cv[j] = control_variate(Asian_theta["Eul"]["payoff"],
                                          Asian_theta["Eul"]["price"],
                                          mu_y)
        Asian_Mil_cv[j] = control_variate(Asian_theta["Mil"]["payoff"],
                                          Asian_theta["Mil"]["price"],
                                          mu_y)
        
    CEV_results.iloc[i, -8:] = [np.mean(Asian_Eul), np.mean(Asian_Mil),
                                np.mean(Asian_Eul_cv), 
                                np.mean(Asian_Mil_cv),
                                np.std(Asian_Eul, ddof=1), 
                                np.std(Asian_Mil, ddof=1),
                                np.std(Asian_Eul_cv, ddof=1), 
                                np.std(Asian_Mil_cv, ddof=1)]

print(CEV_results)
print(np.mean(benchmark["Eul"]["payoff"]), 
      np.mean(benchmark["Mil"]["payoff"]))

#%%
theta = 1.8; epochs = 100
CEV_results = pd.DataFrame({"M": M_lst, "n": n, "theta": theta, 
                            "mu_Eul": 0.0, "mu_Mil": 0.0,
                            "mu_Eul_cv": 0.0, "mu_Mil_cv": 0.0,
                            "sd_Eul": 0.0, "sd_Mil": 0.0,
                            "sd_Eul_cv": 0.0, "sd_Mil_cv": 0.0,})
np.random.seed(4002)
benchmark = Sim_Asian(1e6, 1e4, S_0, K, r, sigma, T, theta)
Asian_Eul, Asian_Mil = np.empty(epochs), np.empty(epochs)
Asian_Eul_cv, Asian_Mil_cv = np.empty(epochs), np.empty(epochs)
mu_y = np.mean(benchmark["Eul"]["price"])
for i in range(len(M_lst)):
    M = M_lst[i]
    np.random.seed(4002)
    for j in trange(epochs, desc=f"M = {M}"):
        Asian_theta = Sim_Asian(n, M, S_0, K, r, sigma, T, theta)
        
        Asian_Eul[j] = np.mean(Asian_theta["Eul"]["payoff"])
        Asian_Mil[j] = np.mean(Asian_theta["Mil"]["payoff"])
        Asian_Eul_cv[j] = control_variate(Asian_theta["Eul"]["payoff"],
                                          Asian_theta["Eul"]["price"],
                                          mu_y)
        Asian_Mil_cv[j] = control_variate(Asian_theta["Mil"]["payoff"],
                                          Asian_theta["Mil"]["price"],
                                          mu_y)
        
    CEV_results.iloc[i, -8:] = [np.mean(Asian_Eul), np.mean(Asian_Mil),
                                np.mean(Asian_Eul_cv), np.mean(Asian_Mil_cv),
                                np.std(Asian_Eul, ddof=1), np.std(Asian_Mil, ddof=1),
                                np.std(Asian_Eul_cv, ddof=1), np.std(Asian_Mil_cv, ddof=1)]

print(CEV_results)
print(np.mean(benchmark["Eul"]["payoff"]), np.mean(benchmark["Mil"]["payoff"]))
