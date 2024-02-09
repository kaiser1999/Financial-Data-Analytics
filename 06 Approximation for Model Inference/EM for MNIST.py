import numpy as np
import tensorflow as tf
from tqdm import trange
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

((X_train, y_train), 
 (X_test, y_test)) = tf.keras.datasets.mnist.load_data()
# Limit the pixel value between 0 and 1 to avoid computational explode
X_train[np.where(X_train!=0)] = 1
X_test[np.where(X_test!=0)] = 1
M = len(np.unique(y_train))             # 10: numbers from 0 to 9
img_dim = X_train.shape[1:]             # 28x28 pixels
N_train = len(X_train)                  # 60000 training samples
N_test = len(X_test)                    # 10000 test samples

#%%
def loglike_func(post, prior, mu):
    loglike = 0
    mu = np.clip(mu, a_min=1e-323, a_max=1 - 1e-323)
    for m in range(M):
        # X_train / log_sums: (n, img1, img2); mu: (M, img1, img2)
        log_sums = np.sum(X_train*np.log(mu[m]) + 
                          (1 - X_train)*np.log(1 - mu[m]), axis=(1,2))
        loglike += np.sum(post[:, m]*(np.log(prior[m]) + log_sums))
    
    return loglike

#%%
np.random.seed(4002)
prior = np.random.rand(M)
prior /= np.sum(prior)
mu = np.empty((M,) + img_dim)
for m in range(M):
    digit = X_train[np.where(y_train == m)[0]]
    mu[m] = np.mean(digit, axis=0)

#%%
epochs = 10
prior_record = np.empty((M, epochs))
loglike_record = np.empty((epochs))
t = trange(epochs, desc="log likelihood: 00000")
for ep in t:
    # Compute posterior probabilities
    nominator = np.zeros((M, N_train))
    for m in range(M):
        binm = mu[m]**X_train * (1 - mu[m])**(1 - X_train)
        nominator[m] = prior[m] * np.prod(binm, axis=(1,2))
        
    # denominator = sum nominator
    posterior = nominator.T / np.sum(nominator, axis=0).reshape(-1, 1)
    loglike_record[ep] = loglike_func(posterior, prior, mu)
    t.set_description(f"log likelihood: {np.round(loglike_record[ep])}")
    
    # Update mu and prior
    for m in range(M):
        px = np.sum(posterior[:,m].reshape((-1, 1, 1)) * X_train, axis=0)
        mu[m] = px / np.sum(posterior[:,m])
        prior[m] = np.sum(posterior[:,m]) / N_train
        
    prior = prior/np.sum(prior)
    prior_record[:, ep] = prior.T

#%%
y_prob = np.zeros((M, len(X_test)))
for m in range(M):
    binm = mu[m]**X_test * (1 - mu[m])**(1 - X_test)
    y_prob[m,:] = prior[m] * np.prod(binm, axis=(1,2))

y_hat = np.array(y_prob).argmax(axis=0)

# Table of predictions and prediction accuracy
tab = confusion_matrix(y_test, y_hat)                  # Confusion matrix
print(tab)

# No need to inter-change labels cuz mu is first initialized by class
print(np.sum(tab.diagonal()) / len(y_test))             # Accuracy

#%%
# Plot for the log-likelihood for all 10 iterations
plt.figure(figsize=(10, 8))
plt.plot(np.arange(len(loglike_record))+1, loglike_record, 'x-', linewidth=3)
plt.xlabel("Iterations", fontsize=15)
plt.ylabel("Log-likelihood", fontsize=15)
plt.title("BMM EM Algoritm", fontsize=15)
plt.tight_layout()
plt.savefig("../Picture/Python log-likelihood_BMM.png", dpi=200)

#%%
fig = plt.figure(figsize=(16, 12))
for m in range(M):
    y = fig.add_subplot(3, 4, m+1)                      # rows, columns, index
    y.imshow(mu[m,:,:], aspect='auto', cmap="gray_r")

plt.tight_layout()
fig.savefig("../Picture/Python Number 0-9_BMM.png", dpi=200)