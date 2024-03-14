import numpy as np
from scipy.special import beta
np.random.seed(4002)

n, psi = 10000, []
for i in range(1000):
    u = np.random.rand(n)
    psi.append(np.sum(u**3 * (1-u)**2.5) / n)
    
print(np.mean(psi))
print(beta(4, 3.5))
print(np.var(psi, ddof=1))

#%%
import numpy as np
np.random.seed(4002)

n, psi_A = 10000, []
for i in range(1000):
    u = np.random.rand(np.int(n/2))
    v = 1 - u
    psi_A.append(np.sum(u**3 * (1-u)**2.5 + v**3 * (1-v)**2.5) / n)
    
print(np.mean(psi_A))
print(np.var(psi_A, ddof=1))

#%%
import numpy as np
np.random.seed(4002)

n, psi_C = 10000, []
mu_y = 2/7
for i in range(1000):
    u = np.random.rand(n)
    x, y = u**3 * (1-u)**2.5, (1-u)**2.5
    # Using sample variance and sample covariance
    s_cov = np.cov(x, y, ddof=1)[0][1]
    s_var = np.var(y, ddof=1)
    psi_C.append(np.mean(x) - s_cov/s_var * (np.mean(y) - mu_y))

print(np.mean(psi_C))
print(np.var(psi_C, ddof=1))

#%%
import numpy as np
np.random.seed(4002)

n, psi_MC = 10000, []
mu_y = np.array([2/7, 1/4])
for i in range(1000):
    u = np.random.rand(n)
    x, y = u**3 * (1-u)**2.5, np.vstack([(1-u)**2.5, u**3])
    # Using sample variance and sample covariance
    s_cov = np.cov(x, y, ddof=1)[0,1:]
    diff_y = np.mean(y, axis=1) - mu_y
    inv_s_var = np.linalg.inv(np.cov(y, ddof=1))
    psi_MC.append(np.mean(x) - s_cov @ inv_s_var @ diff_y)

print(np.mean(psi_MC))
print(np.var(psi_MC, ddof=1))

#%%
import numpy as np
np.random.seed(4002)

n, psi_S, J = 10000, [], 10
for i in range(1000):
    u_j = np.random.rand(np.int(n/J), J)
    u_j = (u_j + np.arange(10).reshape(1, -1))/J
    psi_S.append(np.sum(u_j**3 * (1-u_j)**2.5) / n)

print(np.mean(psi_S))
print(np.var(psi_S, ddof=1))