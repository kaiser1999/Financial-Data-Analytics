import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 12, 8
plt.rcParams.update({'font.size': 15})

def NW_estimator(x0, x, y, h, kernel=norm.pdf):
    x0, x = np.asarray(x0)[:, None], np.asarray(x)
    f_x = kernel((x0 - x)/h)/h
    return f_x @ y / np.sum(f_x, axis=1)

np.random.seed(4012)
n = 500
eps = np.random.normal(loc=0, scale=2, size=n)
m = lambda x: x**2 * np.cos(x)
x = np.random.normal(loc=2, scale=4, size=n)
y = m(x) + eps
h = 0.5    # Bandwidth

XGrid = np.linspace(start=-15, stop=15, num=100)

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
plt.scatter(x, y, edgecolors="black", color="white", s=20)
p1, = plt.plot(XGrid, m(XGrid), color='b')
p2, = plt.plot(XGrid, NW_estimator(x0=XGrid, x=x, y=y, h=h), color='r')

plt.xlabel("x0")
plt.ylabel("y")
plt.legend(handles=[p1, p2], labels=["True regression", "Nadaraya-Watson"],
           loc="upper center")
plt.savefig("Kernel Regression Example.png", dpi=200)