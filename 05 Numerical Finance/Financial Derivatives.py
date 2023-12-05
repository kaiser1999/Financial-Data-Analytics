import numpy as np
import pandas as pd
from scipy.stats import norm

# Initialize variables
S_0 = 10; q = 0; sigma = 0.7; r = 0; T = 1; K = 9; h = 0.3
params = {"loc": (r-q-sigma**2/2)*T, "scale": sigma*np.sqrt(T)}
discount = np.exp(-r * T)

# Compute the exact Delta
d_plus = (np.log(S_0/K) + (r-q+sigma**2/2)*T) / (sigma*np.sqrt(T))
delta = norm.cdf(d_plus)
print(delta)

#%%
np.random.seed(4012)

# Create a data frame to store results
results = pd.DataFrame(columns=["h", "n", "delta", 
                                "est_forward_diff", 
                                "est_central_diff"])
n_size = [1e6, 1e8]
# Estimate Delta by forward and central difference with different n
for n in n_size:
    # Generate the Black-Scholes sample
    S_T = S_0 * np.exp(norm.rvs(**params, size=int(n)))
    Y_bar_S0 = discount * np.mean(np.maximum(S_T - K, 0))
  
    S_T_minus_h = (S_0 - h) * np.exp(norm.rvs(**params, size=int(n)))
    Y_bar_S0_minus_h = discount * np.mean(np.maximum(S_T_minus_h - K, 0))
  
    S_T_plus_h = (S_0 + h) * np.exp(norm.rvs(**params, size=int(n)))
    Y_bar_S0_plus_h = discount * np.mean(np.maximum(S_T_plus_h - K, 0))
  
    # Estimate using forward difference method
    est_forward_diff = (Y_bar_S0_plus_h - Y_bar_S0) / h
  
    # Estimate using central difference method
    est_central_diff = (Y_bar_S0_plus_h - Y_bar_S0_minus_h) / (2*h)
  
    # Store the results in the data frame
    output = pd.DataFrame({"h": [h], "n": [int(n)], "delta": [delta], 
                           "est_forward_diff": [est_forward_diff], 
                           "est_central_diff": [est_central_diff]})
    results = pd.concat([results, output])

print(results)

# Relative error for forward and central difference methods
print(abs(results.est_forward_diff.values - delta) / delta * 100)
print(abs(results.est_central_diff.values - delta) / delta * 100)

#%%
np.random.seed(4012)

n_size = np.arange(400, 20000, 400)
est_pathwise = []

# Estimate Delta by pathwise method with different n
for i in range(len(n_size)):
    # Generate the Black-Scholes sample
    S_T = S_0 * np.exp(norm.rvs(**params, size=int(n_size[i])))
    
    # Estimate using pathwise method
    result = (S_T > K) * S_T / S_0
    est_pathwise.append(discount * np.mean(result))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
# Plot the graph of estimated Delta and exact Delta
plt.plot(n_size, est_pathwise, "ro-", markerfacecolor='white',
         linewidth=1.5, markersize=10, label="pairwise estimate")
plt.ylim((0.4, 0.9))
plt.xlabel("number of paths", fontsize=15)
plt.ylabel("Delta", fontsize=15)
plt.axhline(y=delta, color='b', linestyle='-', label="exact")
plt.legend(fontsize=15)

plt.tight_layout()
plt.savefig("../Picture/Pathwise_estimate.png", dpi=200)