#%%
# SVM
#%%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4002)

n_data_points = 200

rad = 0.5
thk = 0.2
sep = 0.2

# We use random radius in the interval [rad, rad+thk]
#  and random angles from 0 to pi radians.
r1 = np.random.rand(n_data_points)*thk+rad
a1 = np.random.rand(n_data_points)*np.pi

r2 = np.random.rand(n_data_points)*thk+rad
a2 = np.random.rand(n_data_points)*np.pi+np.pi

r = np.append(r1, r2)
a = np.append(a1, a2)

# In order to plot it we convert it to cartesian:
x1, y1 = r*np.cos(a), r*np.sin(a)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel(r'$x_1$', fontsize=20)
ax.set_ylabel(r'$x_2$', fontsize=20)
plt.scatter(x1, y1, marker='.', linewidths=0.1, c="red", s=100)

rad = 1.5
thk = 0.2
sep = 0.2

# We use random radius in the interval [rad, rad+thk]
#  and random angles from 0 to pi radians.
r1 = np.random.rand(n_data_points)*thk+rad
a1 = np.random.rand(n_data_points)*np.pi

r2 = np.random.rand(n_data_points)*thk+rad
a2 = np.random.rand(n_data_points)*np.pi+np.pi

r = np.append(r1, r2)
a = np.append(a1, a2)

# In order to plot it we convert it to cartesian:
x2, y2 = r*np.cos(a), r*np.sin(a)

plt.scatter(x2, y2, marker='.', linewidths=0.1, c="blue", s=100)
plt.savefig("../Picture/SVM_scatter.png", dpi=200)

z1 = x1**2 + y1**2
z2 = x2**2 + y2**2
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, y1, z1, c="red", s=50)
ax.scatter(x2, y2, z2, c="blue", s=50)
ax.set_xlabel(r'$x_1$', fontsize=20)
ax.set_ylabel(r'$x_2$', fontsize=20)
ax.set_zlabel(r'$K$', fontsize=20)
ax.set_zlim(0, 3)

ax.view_init(15, 35)
plt.tight_layout()
plt.savefig("../Picture/SVM_scatter3D.png", dpi=200) # need self crop

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 7))
ax1.scatter(x1**2, y1**2, marker='.', linewidths=0.1, c="red", s=100)
ax1.scatter(x2**2, y2**2, marker='.', linewidths=0.1, c="blue", s=100)
ax1.set_xlabel(r'$x_1^2$', fontsize=20)
ax1.set_ylabel(r'$x_2^2$', fontsize=20)

ax2.scatter(x1**2, np.sqrt(2)*x1*y1, marker='.', linewidths=0.1, c="red", s=100)
ax2.scatter(x2**2, np.sqrt(2)*x2*y2, marker='.', linewidths=0.1, c="blue", s=100)
ax2.set_xlabel(r'$x_1^2$', fontsize=20)
ax2.set_ylabel(r'$\sqrt{2} x_1 x_2$', fontsize=20)

plt.tight_layout()
plt.savefig("../Picture/SVM_scatter rbf.png", dpi=200)

#%%
np.random.seed(4002)
x1 = np.random.randn(12, 2)
x2 = np.random.randn(12, 2)*0.5 - 0.5

fig = plt.figure(figsize=(10, 10))
plt.scatter(x1[:,0], x1[:,1], linewidths=1.5, c="blue")
plt.scatter(x2[:,0], x2[:,1], linewidths=1.5, c="red")

from sklearn import svm
from mlxtend.plotting import plot_decision_regions

X = np.vstack((x1, x2))
y = np.array([0]*12 + [1]*12)

sig = 1
clf = svm.SVC(gamma=1/2/sig**2)
clf.fit(X, y)

fig = plt.figure(figsize=(10, 10))
# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(X=X, y=y, clf=clf, legend=1, scatter_kwargs={'s':120})

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel(r'$x_1$', size=20)
plt.ylabel(r'$x_2$', size=20)
#plt.title(f'SVM Decision Region Boundary with sigma={sig}', size=20)
plt.tight_layout()
plt.savefig(f"../Picture/SVM sigma {sig}.png", dpi=500)

sig = 0.3
clf = svm.SVC(gamma=1/2/sig**2)
clf.fit(X, y)

fig = plt.figure(figsize=(10, 10))
# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(X=X, y=y, clf=clf, legend=1, scatter_kwargs={'s':120})

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel(r'$x_1$', size=20)
plt.ylabel(r'$x_2$', size=20)
#plt.title(f'SVM Decision Region Boundary with sigma={sig}', size=20)
plt.tight_layout()
plt.savefig(f"../Picture/SVM sigma {sig}.png", dpi=500)