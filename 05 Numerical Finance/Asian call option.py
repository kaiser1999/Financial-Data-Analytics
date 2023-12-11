import numpy as np
import pandas as pd

# Initialize variables
S_0 = 10; r = 0; sigma = 0.7; T = 1; K = 8
M_values = [50, 100, 150]
path_values = [1e4, 1e5, 1e6]

# Create a data frame to store results
results = pd.DataFrame({"M": np.repeat(M_values, len(path_values)),
                        "Path": np.tile(path_values, len(M_values)), 
                        "Estimate": 0.0})

np.random.seed(4002)
for index, row in results.iterrows():
    M, n = int(row.M), int(row.Path)
    
    delta_t = T / M
    prices = S_0
    avg_price = S_0 / (M+1)
    for m in range(M):
        z = np.random.normal(size=n)
        prices *= 1 + r*delta_t + sigma*np.sqrt(delta_t)*z
        avg_price += prices / (M+1)
    
    payoff_T = np.exp(-r*T) * np.maximum(avg_price - K, 0)
    results.iloc[index, -1] = np.mean(payoff_T)

print(results)