import numpy as np
from scipy.stats import norm
import mpmath
import matplotlib.pyplot as plt

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

def fzero(alpha, p, x, sig=0.3):
    eps = -np.log(p)/sig
    lamb = alpha/sig - sig/2
    
    result = []
    for t in x:
        result.append(Feq(t, eps, lamb))
    return result

x = np.linspace(0.01, 5, num=100)

plt.plot(x, fzero(-0.01, 0.8, x), c="blue")
plt.plot(x, fzero(0.008, 0.75, x), c="red")
plt.plot(x, fzero(0.1, 0.8, x), c="gray")
plt.plot(x, fzero(0.02, 0.8, x), c="orange")

plt.ylim(bottom=-2)
plt.axhline(y=0, color='black', linestyle='-')
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
plt.gca().set_yticks([0])
plt.gca().set_xticks([0, 5])
plt.gca().set_xticklabels([0, "T"])
plt.ylabel(r'$\varphi(s)$', rotation=0)
plt.xlabel(r'$s$')

plt.rcParams["figure.figsize"] = 13, 10
plt.rcParams.update({'font.size': 15})

plt.savefig("Optimal roots.png", dpi=400)