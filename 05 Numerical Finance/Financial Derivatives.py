import numpy as np
import pandas as pd

# Initialize variables
S_0 = 10; K = 8; r = 0.05; sigma = 0.3; T = 1

def Sim_greek(h_d, h_v, n, M, S_0, K, r, sigma, T, theta, seed=4002):
    delta_t = T / M
    def Milstein(z, S_t, r, sigma):
        return (S_t*(1+r*delta_t) + sigma*S_t**(theta/2)*np.sqrt(delta_t)*z + 
                sigma**2*theta/2*S_t**(theta-1)*delta_t/2*(z**2-1))

    def Euro_call(S_t, K, r, sigma, T):
        np.random.seed(seed)
        for m in range(M):
            z = np.random.randn(n)
            S_t = Milstein(z, S_t, r, sigma)
        return np.mean(np.exp(-r*T) * np.maximum(S_t - K, 0))

    Y = Euro_call(S_0, K, r, sigma, T)

    # Estimate delta using forward and central difference method
    Y_S0_neg = Euro_call(S_0-h_d, K, r, sigma, T)
    Y_S0_pos = Euro_call(S_0+h_d, K, r, sigma, T)

    # Estimate vega using forward and central difference method
    Y_sig_neg = Euro_call(S_0, K, r, sigma-h_v, T)
    Y_sig_pos = Euro_call(S_0, K, r, sigma+h_v, T)

    return {"delta": {"forward": (Y_S0_pos - Y)/h_d, 
                      "central": (Y_S0_pos - Y_S0_neg)/(2*h_d)},
            "vega": {"forward": (Y_sig_pos - Y)/h_v, 
                     "central": (Y_sig_pos - Y_sig_neg)/(2*h_v)}}

#%%
from scipy.stats import norm
from tqdm import trange

# Compute the exact delta and vega for theta=2 (BS)
d_plus = (np.log(S_0/K) + (r+sigma**2/2)*T) / (sigma*np.sqrt(T))
exact_delta = norm.cdf(d_plus)
exact_vega = S_0 * np.sqrt(T) * norm.pdf(d_plus)
print("When \u03B8=2, the exact delta and vega are")
print(exact_delta)
print(exact_vega)

h_delta = np.arange(0.5, 0.05 - 0.001, -0.05)
h_vega = np.arange(0.05, 0.005 - 0.0001, -0.005)
n, M = int(1e6), int(1e4)
theta_lst = [2, 1.8, 1]

for theta in theta_lst:
    delta_finite = pd.DataFrame({"h": h_delta, "theta": theta, 
                                 "delta_forward": 0.0, 
                                 "delta_central": 0.0})
    vega_finite = pd.DataFrame({"h": h_vega, "theta": theta, 
                                "vega_forward": 0.0, "vega_central": 0.0})
    for i in trange(len(h_delta)):
        h_d, h_v = h_delta[i], h_vega[i]
        Euro_CEV = Sim_greek(h_d, h_v, n, M, S_0, K, r, sigma, T, theta)
        delta_finite.iloc[i, -2:] = Euro_CEV["delta"].values()
        vega_finite.iloc[i, -2:] = Euro_CEV["vega"].values()
    
    print(delta_finite)
    print(vega_finite)

#%%
def greek_pathwise(n, M, S_0, K, r, sigma, T, theta, seed=4002):
    delta_t = T / M
    y_t = 1; v_t = 0; S_t = S_0
    np.random.seed(seed)
    for m in range(int(M)):
        dw_t = np.sqrt(delta_t) * np.random.normal(size=int(n))
        y_t += r*y_t*delta_t + sigma*theta/2*S_t**(theta/2-1)*y_t*dw_t
        v_t += (r*v_t*delta_t + 
                S_t**(theta/2-1)*(S_t + sigma*theta/2*v_t)*dw_t)
        S_t += r*S_t*delta_t + sigma*S_t**(theta/2)*dw_t
    
    return {"delta": np.mean(np.exp(-r*T)*(S_t > K) * y_t), 
            "vega": np.mean(np.exp(-r*T)*(S_t > K) * v_t)}

n_size = np.arange(500, 100000+1, 500)

import matplotlib.pyplot as plt

M = 1e4; theta = 2
delta_pathwise, vega_pathwise = [], []
# Estimate delta and vega with different n
for i in range(len(n_size)):
    Euro_CEV = greek_pathwise(n_size[i], M, S_0, K, r, sigma, T, theta)
    delta_pathwise.append(Euro_CEV["delta"])
    vega_pathwise.append(Euro_CEV["vega"])

plt.figure(figsize=(10, 7))
plt.plot(n_size, delta_pathwise, "ro-", markersize=7, 
         linewidth=1.5, label="pathwise differentiation")
plt.axhline(y=exact_delta, color="blue", linewidth=1.8, label="exact")
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("delta", fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/pathwise_delta_2.png", dpi=200)

plt.figure(figsize=(10, 7))
plt.plot(n_size, vega_pathwise, "ro-", markersize=7, linewidth=1.5, 
         label="pathwise differentiation")
plt.axhline(y=exact_vega, color="blue", linewidth=1.8, label="exact")
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("vega", fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/pathwise_vega_2.png", dpi=200)

#%%
theta = 1.8
delta_pathwise, vega_pathwise = [], []
# Estimate delta and vega with different n
for i in range(len(n_size)):
    Euro_CEV = greek_pathwise(n_size[i], M, S_0, K, r, sigma, T, theta)
    delta_pathwise.append(Euro_CEV["delta"])
    vega_pathwise.append(Euro_CEV["vega"])

plt.figure(figsize=(10, 7))
plt.plot(n_size, delta_pathwise, "ro-", markersize=7, linewidth=1.5)
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("delta", fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/pathwise_delta_1_8.png", dpi=200)

plt.figure(figsize=(10, 7))
plt.plot(n_size, vega_pathwise, "ro-", markersize=7, linewidth=1.5)
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("vega", fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/pathwise_vega_1_8.png", dpi=200)

#%%
theta = 1
delta_pathwise, vega_pathwise = [], []
# Estimate delta and vega with different n
for i in range(len(n_size)):
    Euro_CEV = greek_pathwise(n_size[i], M, S_0, K, r, sigma, T, theta)
    delta_pathwise.append(Euro_CEV["delta"])
    vega_pathwise.append(Euro_CEV["vega"])

plt.figure(figsize=(10, 7))
plt.plot(n_size, delta_pathwise, "ro-", markersize=7, linewidth=1.5)
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("delta", fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/pathwise_delta_1.png", dpi=200)

plt.figure(figsize=(10, 7))
plt.plot(n_size, vega_pathwise, "ro-", markersize=7, linewidth=1.5)
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("vega", fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/pathwise_vega_1.png", dpi=200)

#%%
np.random.seed(4012)
delta_pathwise, vega_pathwise = [], []
delta_likelihood, vega_likelihood = [], []
mu = (r-sigma**2/2)*T; sd = sigma*np.sqrt(T)

# Estimate delta and vega with different n
for i in range(len(n_size)):
    # Generate the Black-Scholes sample
    w_T = np.sqrt(T) * np.random.normal(size=int(n_size[i]))
    S_T = S_0 * np.exp(mu + sigma * w_T)
    
    d_payoff = np.exp(-r * T) * (S_T > K)
    # Estimate delta and vega using pathwise differentiation method
    delta_pathwise.append(np.mean(d_payoff * S_T/S_0))
    vega_pathwise.append(np.mean(d_payoff * S_T*(w_T - sigma*T)))
    
    payoff = np.exp(-r * T) * np.maximum(S_T - K, 0)
    # Estimate delta and vega using likelihood ratio method
    delta_likelihood.append(np.mean(payoff * w_T/(S_0*sigma*T)))
    vega_likelihood.append(np.mean(payoff * ((w_T**2/T - 1)/sigma - w_T)))

#%%
# Plot the graph of estimated delta and exact delta
plt.figure(figsize=(10, 7))
plt.plot(n_size, delta_pathwise, "ro-", markersize=7, 
         linewidth=1.5, label="pathwise differentiation")
plt.plot(n_size, delta_likelihood, "ko-", markersize=7, linewidth=1.5, 
         label="likelihood ratio")
plt.axhline(y=exact_delta, color="blue", linewidth=1.8, label="exact")
plt.ylim((0.75, 0.95))
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("delta", fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/likelihood_delta.png", dpi=200)

# Plot the graph of estimated vega and exact vega
plt.figure(figsize=(10, 7))
plt.plot(n_size, vega_pathwise, "ro-", markersize=7, linewidth=1.5, 
         label="pathwise differentiation")
plt.plot(n_size, vega_likelihood, "ko-", markersize=7, linewidth=1.5, 
         label="likelihood ratio")
plt.axhline(y=exact_vega, color="blue", linewidth=1.8, label="exact")
plt.ylim((0, 3.6))
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("vega", fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/likelihood_vega.png", dpi=200)