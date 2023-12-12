import numpy as np
import pandas as pd
from scipy.stats import norm

# Initialize variables
S_0 = 10; q = 0; sigma = 0.7; r = 0; T = 1; K = 8; h = 0.3

# Compute the exact Delta
d_plus = (np.log(S_0/K) + (r-q+sigma**2/2)*T) / (sigma*np.sqrt(T))
delta = norm.cdf(d_plus)
vega = S_0 * np.sqrt(T) * norm.pdf(d_plus)
print(delta)
print(vega)

#%%
np.random.seed(4012)

# Create a data frame to store results
results = pd.DataFrame(columns=["h", "n", "delta", 
                                "est_forward_delta", 
                                "est_central_delta",
                                "vega", "est_forward_vega", 
                                "est_central_vega"])

def sim_call(n, S_0, q, sigma, r, T, K):
    # Generate the Black-Scholes sample
    S_T = S_0 * np.exp(norm.rvs(loc=(r-q-sigma**2/2)*T, 
                                scale=sigma*np.sqrt(T), size=int(n)))
    return np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))

n_size = [1e6, 1e8]
# Estimate Delta by forward and central difference with different n
for n in n_size:
    Y_bar_S0 = sim_call(n, S_0, q, sigma, r, T, K)
    
    # Estimate delta using forward and central difference method
    Y_bar_S0_minus_h = sim_call(n, S_0-h, q, sigma, r, T, K)
    Y_bar_S0_plus_h = sim_call(n, S_0+h, q, sigma, r, T, K)
    
    est_forward_delta = (Y_bar_S0_plus_h - Y_bar_S0) / h
    est_central_delta = (Y_bar_S0_plus_h - Y_bar_S0_minus_h) / (2*h)
    
    # Estimate vega using forward and central difference method
    Y_bar_sigma_minus_h = sim_call(n, S_0, q, sigma-h, r, T, K)
    Y_bar_sigma_plus_h = sim_call(n, S_0, q, sigma+h, r, T, K)
    
    est_forward_vega = (Y_bar_sigma_plus_h - Y_bar_S0) / h
    est_central_vega = (Y_bar_sigma_plus_h - Y_bar_sigma_minus_h)/(2*h)
  
    # Store the results in the data frame
    output = pd.DataFrame({"h": [h], "n": [int(n)], "delta": [delta], 
                           "est_forward_delta": [est_forward_delta], 
                           "est_central_delta": [est_central_delta],
                           "vega": [vega],
                           "est_forward_vega": [est_forward_vega], 
                           "est_central_vega": [est_central_vega]})
    results = pd.concat([results, output])

#%%
# Print the result of delta and vega
print(results[["h", "n", "delta", "est_forward_delta",
               "est_central_delta"]])
print(results[["h", "n", "vega", "est_forward_vega",
               "est_central_vega"]])

# Relative error for forward and central difference methods
print(abs(results.est_forward_delta.values - delta) / delta * 100)
print(abs(results.est_central_delta.values - delta) / delta * 100)
print(abs(results.est_forward_vega.values - vega) / vega * 100)
print(abs(results.est_central_vega.values - vega) / vega * 100)

#%%
np.random.seed(4012)
n_size = np.arange(400, 30000, 400)
est_pathwise_delta, est_pathwise_vega = [], []
est_lr_delta, est_lr_vega = [], []
params = {"loc": (r-q-sigma**2/2)*T, "scale": sigma*np.sqrt(T)}
discount = np.exp(-r * T)

# Estimate Delta by pathwise method with different n
for i in range(len(n_size)):
    # Generate the Black-Scholes sample
    S_T = S_0 * np.exp(norm.rvs(**params, size=int(n_size[i])))
    
    # Estimate Delta using pathwise method
    result = (S_T > K) * S_T / S_0
    est_pathwise_delta.append(discount * np.mean(result))
    
    # Estimate Vega using pathwise method
    result = (S_T > K) * S_T *(np.log(S_T/S_0)-(r-q+sigma**2/2)*T)
    est_pathwise_vega.append(discount / sigma * np.mean(result))
    
    # Estimate Delta using likelihood ratio method
    h = (np.log(S_T/ S_0) - ((r - sigma**2/2)*T)) / (sigma*np.sqrt(T))
    result = np.maximum(S_T - K, 0) * h / (S_0 * sigma * np.sqrt(T))
    est_lr_delta.append( discount * np.mean(result))
  
    # Estimate Vega using likelihood ratio method
    result = np.maximum(S_T-K, 0)*(-1/sigma +h**2/sigma -np.sqrt(T)*h)
    est_lr_vega.append(discount * np.mean(result))
    
import matplotlib.pyplot as plt

# Plot the graph of estimated Delta and exact Delta
plt.figure(figsize=(10, 7))
plt.plot(n_size, est_pathwise_delta, "ro-", markersize=7, 
         linewidth=1.5, label="pathwise estimate")
plt.plot(n_size, est_lr_delta, "ko-", markersize=7, linewidth=1.5, 
         label="likelihood ratio estimate")
plt.axhline(y=delta, color='blue', linewidth=1.8, label="exact")
plt.ylim((0.5, 1))
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("Delta", fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("likelihood_delta.png", dpi=200)

# Plot the graph of estimated Vega and exact Vega
plt.figure(figsize=(10, 7))
plt.plot(n_size, est_pathwise_vega, "ro-", markersize=7, linewidth=1.5, 
         label="pathwise estimate")
plt.plot(n_size, est_lr_vega, "ko-", markersize=7, linewidth=1.5, 
         label="likelihood ratio estimate")
plt.axhline(y=vega, color='blue', linewidth=1.8, label="exact")
plt.ylim((1.5, 4.5))
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("Vega", fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("likelihood_vega.png", dpi=200)