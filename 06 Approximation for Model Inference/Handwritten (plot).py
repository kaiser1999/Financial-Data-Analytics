import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Limit the pixel value between 0 and 1 to avoid computational explode
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#%%
np.random.seed(4002)

nrows, ncols = 10, 10
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(100,100))

idx = np.random.choice(range(len(y_train)), size=nrows*ncols)
for i in range(nrows):
    for j in range(ncols):
        axes[i,j].imshow(X_train[i*nrows+j], aspect="auto", cmap='gray_r')
        axes[i,j].set_axis_off()

fig.tight_layout()
fig.savefig("../Picture/handwriting.png", dpi=20)

#%%
np.random.seed(4002)

digit = 2
n_plot = 10

select = np.random.choice(np.where(y_train == digit)[0], size=100)
fig, axes = plt.subplots(nrows=1, ncols=n_plot+1, figsize=((n_plot+1)*10,10))
for i in range(n_plot):
    axes[i].imshow(X_train[select[i]], aspect="auto", cmap='gray_r')
    axes[i].set_axis_off()

X_mean = np.mean(X_train[select], axis=0)
axes[-1].imshow(X_mean, aspect="auto", cmap='gray_r')

fig.tight_layout()
fig.savefig("../Picture/EM_handwriting digit 2.png", dpi=200)