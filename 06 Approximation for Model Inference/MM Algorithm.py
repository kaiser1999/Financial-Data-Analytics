import numpy as np

def MM_quantile(x, q, tol=1e-18, maxit=1e4):
    n = len(x)
    mu_old = np.mean(x)
    for i in range(int(maxit)):
        w = 1/np.abs(x - mu_old)
        mu_new = (sum(w * x) + (2*q - 1)*n)/sum(w)
        if np.isnan(mu_new): return mu_old
        if abs(mu_new - mu_old) < tol: return mu_new
        mu_old = mu_new
  
    return mu_new

#%%
# same as Risk_Measure.py
import pandas as pd
import numpy as np

d = pd.read_csv("../Datasets/stock_1999_2002.csv", index_col=0)

x_n = d.iloc[-1,:]           # select the last obs
w = [40000, 30000, 30000]    # investment amount on each stock
p_0 = sum(w)	             # total investment amount
w_s = w/x_n                  # no. of shares bought at day n

h_sim = (d/d.shift(1) * x_n)[1:]
p_n = h_sim @ w_s            # portfolio value at day n

loss = p_0 - p_n	         # loss

#%%
VaR_sim = MM_quantile(loss, 0.99) # 1-day 99% V@R
print(VaR_sim)