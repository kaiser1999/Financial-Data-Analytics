import numpy as np
import pandas as pd
from scipy.stats import norm
import mpmath
from scipy.optimize import brentq

def f(t, eps, lamb):
    x = eps + lamb*t
    z = 1/np.sqrt(2*np.pi*t)*np.exp(-x**2/(2*t)) - lamb*norm.cdf(-x/np.sqrt(t))
    return 2*np.exp(2*lamb*eps)*z

def Laplace_G(s, eps, lamb):
    a1 = lamb - mpmath.sqrt(lamb**2 + 2*s)
    a2 = lamb + mpmath.sqrt(lamb**2 + 2*s)
    return 2*(mpmath.exp(2*a1*eps) - mpmath.exp(2*lamb*eps))/(a2*mpmath.exp(a1*eps) - a1*mpmath.exp(a2*eps))

def gamma(t, eps, lamb):
    return float(mpmath.invertlaplace(lambda s: Laplace_G(s, eps, lamb), t, method='talbot'))

def eta(t, eps, lamb):
    return f(t, eps, lamb) - f(t, 0, lamb)

def varphi(t, eps, lamb):
    return eta(t, eps, lamb) - gamma(t, eps, lamb)

def Feq(t, eps, lamb):
    return varphi(t/eps**2, 1, lamb)

def fzero(eps, lamb, a, b):
    return brentq(lambda t: Feq(t, eps, lamb), a, b)

#%%

def Roots_Table(eps, lamb):
    eps_lamb = lamb * eps
    try:
        if eps_lamb > -0.1205 and eps_lamb < 0:
            nRoot = 2
            Root = np.zeros(nRoot)
            # s_1 = Root[1] < Root[0] = s_2
            Root[0] = fzero(eps, eps_lamb, 1.7*eps**2, -eps**2/(2*eps_lamb))
            Root[1] = fzero(eps, eps_lamb, 0.00001, 1.7*eps**2)
        elif eps_lamb > 0:
            Root = fzero(eps, eps_lamb, 0.00001, np.max([1, 1.5*(eps**2)]))
        else: #eps_lamb <= -0.1205 and eps_lamb = 0:
            Root = np.infty
    except:
        print(eps, lamb)
        Root = np.infty
    return Root

LAMBDAS = [-18, -17, -14, -11, -8, -5, -2, -0.5, -0.1, -0.05, 
           0.05, 0.1, 0.5, 2, 5, 8, 11, 14, 17, 18]
EPSILONS = [0.01, 0.025, 0.05, 0.1, 0.5, 2, 5, 8, 11, 14, 17, 21, 22]


results = {}
for lamb in LAMBDAS:
    eps_vec = []
    for eps in EPSILONS:
        eps_vec.append(Roots_Table(eps, lamb))
    
    results[lamb] = eps_vec

df = pd.DataFrame(results, columns=LAMBDAS, index=EPSILONS)
df.to_csv("Roots_Table.csv")
