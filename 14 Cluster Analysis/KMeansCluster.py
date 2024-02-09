import numpy as np
from sklearn.cluster import KMeans

def get_bcss(km, X):
    ng = np.bincount(km.labels_)                # size of each cluster
    return sum(ng*np.sum((np.mean(X, axis=0) - km.cluster_centers_)**2, axis=1))

# Try several values of K, choose K so that stat. is maximized
def kmstat(X, K):
    km = KMeans(n_clusters=K, n_init="auto").fit(X)
    n = X.shape[0]                      # sample size
    wcss = km.inertia_                  # within group ss
    bcss = get_bcss(km, X)              # between group ss
    return ((n - K) * bcss) / ((K - 1) * wcss), km

# Try KMeans() several times and output the best trial
def best_km(X, K, trial=5, seed=4002):
    np.random.seed(seed)
    r0 = 0
    for i in range(trial):
        r, km = kmstat(X, K)
        if r > r0:            # update r0 if it is less than r
            r0, km0 = r, km
    print(f"K = {K}; stat = {r0}")
    return r0, km0