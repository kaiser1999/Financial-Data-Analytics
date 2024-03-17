import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange
import matplotlib.pyplot as plt

class MCMC_logistic:
    def __init__(self, num_it=5e5, burn_in=1e5, nu=0.5, seed=4002):
        self.num_it = int(num_it)
        self.burn_in = int(burn_in)
        self.nu = nu
        self.seed = seed
        self.proposal_rvs = multivariate_normal.rvs
    
    def sigmoid(self, beta, X, y=None):
        exp_eta = np.exp(X @ beta)
        return (exp_eta / (1 + exp_eta)).reshape(y.shape)

    def log_prior(self, beta):              # ln P(beta)
        # large cov to be a non-informative prior
        return np.sum(multivariate_normal.logpdf(beta, mean=0, 
                                                 cov=100**2))
    
    def get_log(self, x, val=1e-300):       # avoid log(0) = nan
        return np.log(np.clip(x, a_min=val, a_max=np.infty))
    
    def log_likelihood(self, beta, X, y):   # ln L(beta) 
        pi = self.sigmoid(beta, X, y)
        return np.sum(y*self.get_log(pi) + (1-y)*self.get_log(1-pi))
    
    def log_target(self, beta, X, y):       # approx. ln P(beta| X, y)
        return self.log_likelihood(beta, X, y) + self.log_prior(beta)
    
    def fit(self, X, y):
        np.random.seed(self.seed)
        X_ = np.c_[np.ones(len(y)), X]      # add intercept
        self.n_accept = 0
        self.n_col = X_.shape[1]
        
        Cov_nu = self.nu * np.linalg.inv(X_.T @ X_)
        beta_hist = np.empty(shape=(self.num_it, self.n_col))
        beta_hist[0] = self.proposal_rvs(mean=np.repeat(0,self.n_col), 
                                         cov=Cov_nu, size=1)
        for i in trange(1, self.num_it):
            beta_old = beta_hist[i-1]
            beta_new = self.proposal_rvs(mean=beta_old, cov=Cov_nu, 
                                         size=1).reshape(-1, 1)
            
            # Since the proposal distribution, normal, is symmetric, 
            # we have g(beta_old| beta_new) = g(beta_new| beta_old)
            log_target_old = self.log_target(beta_old, X_, y)
            log_target_new = self.log_target(beta_new, X_, y)
            alpha = np.exp(log_target_new - log_target_old)
            if np.random.rand() < min(1, alpha):
                beta_old = beta_new
                self.n_accept += 1
            
            beta_hist[i] = beta_old.T
        
        self.beta_dist = beta_hist
    
    def get_coefficient(self, method="median"):
        assert method in ["mean", "median"]
        func = getattr(np, method)
        return func(self.beta_dist[self.burn_in:], axis=0)
    
    def predict(self, X, method="median"):
        X_ = np.c_[np.ones(len(y)), X]
        beta = self.get_coefficient(method)
        pi = self.sigmoid(beta, X_, np.arange(len(X_)))
        return (pi > 0.5).astype(np.int32)
    
    def plot(self, save_name, method="median"):
        beta_MCMC = self.get_coefficient(method)
        for col in range(self.n_col):
            fig, (ax1, ax2) = plt.subplots(ncols=2, 
                                           figsize=(20, 10))
            ax1.plot(self.beta_dist[:,col], color="black", 
                     linewidth=2)
            ax1.axvline(self.burn_in, color="b", linestyle="--", 
                        linewidth=2)
            ax1.axhline(beta_MCMC[col], color="r", linestyle="--", 
                        linewidth=2)
            ax1.set_xlabel("iterations", fontsize=15)
            ax1.set_ylabel(f"beta {col}", fontsize=15)
            ax1.set_title("MCMC path", fontsize=20)
            
            ax2.hist(self.beta_dist[self.burn_in:,col], alpha=0.5,
                     color="gray", ec='black', bins="sturges")
            ax2.axvline(beta_MCMC[col], color="r", linestyle="--")
            ax2.set_xlabel(f"beta {col}", fontsize=15)
            ax2.set_ylabel("frequencies", fontsize=15)
            ax2.set_title("MCMC histogram", fontsize=20)
            
            plt.tight_layout()
            fig.savefig(save_name + f"_beta {col}.png", dpi=200)

#%%
import pandas as pd

df = pd.read_csv("../Datasets/fin-ratio.csv")
X, y = df.drop(columns=["HSI"]).values, df.HSI.values

MCMC_model = MCMC_logistic(num_it=5e5, burn_in=1e5)
MCMC_model.fit(X, y)
y_hat_MCMC = MCMC_model.predict(X, method="median")
print(MCMC_model.n_accept / MCMC_model.num_it)
print("MCMC_median", MCMC_model.get_coefficient("median"))
print("MCMC_mean", MCMC_model.get_coefficient("mean"))

#%%
from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression(penalty=None)
sklearn_model.fit(X, y)
y_hat_MLE = sklearn_model.predict(X)
print("MLE", np.append(sklearn_model.intercept_, 
                       sklearn_model.coef_[0]))

print(f"MCMC_median: {np.mean(y == y_hat_MCMC)}")
print(np.where(y_hat_MCMC != y)[0])

print(f"MLE: {np.mean(y == y_hat_MLE)}")
print(np.where(y_hat_MLE != y)[0])

#%%
MCMC_model.plot("../Picture/MCMC", method="median")
print(np.round(np.corrcoef(X, rowvar=False), 5))