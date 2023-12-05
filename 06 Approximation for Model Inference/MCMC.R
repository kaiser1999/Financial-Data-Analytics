# same as Risk_Measure.R
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(mvtnorm)

sigmoid <- function(beta, X, y=NA){
  exp_eta <- exp(X %*% beta)
  return(exp_eta / (1 + exp_eta))
}

MCMC_logistic <- function(X, y, num_it=1e5, nu=0.5, seed=4002){
  # ln P(beta): large cov to be a non-informative prior
  log_prior <- function(beta) sum(dnorm(beta, 0, 100, log=TRUE))
  
  log_likelihood <- function(beta, X, y){   # ln L(beta)
    pi <- sigmoid(beta, X, y)
    return(sum(y * log(pi) + (1 - y) * log(1 - pi)))
  }
  
  log_target <- function(beta, X, y){       # approx. ln P(beta| X, y)
    return(log_likelihood(beta, X, y) + log_prior(beta))
  }
  
  set.seed(seed)
  X_ <- cbind(1, as.matrix(X))
  n_accept <- 0
  
  Cov_nu <- nu * solve(t(X_) %*% X_)
  beta_hist <- matrix(NA, nrow=num_it, ncol=ncol(X_))
  beta_hist[1,] <- rmvnorm(1, rep(0, ncol(X_)), Cov_nu)
  for (i in 2:num_it){
    beta_old <- beta_hist[i-1,]
    beta_new <- rmvnorm(1, beta_old, Cov_nu)
    
    log_target_old <- log_target(matrix(beta_old, ncol=1), X_, y)
    log_target_new <- log_target(t(beta_new), X_, y)
    
    alpha <- exp(log_target_new - log_target_old)
    if (!is.na(alpha)){
      if (runif(1) < min(1, alpha)){
        beta_old <- beta_new
        n_accept <- n_accept + 1
      }
    }
    
    beta_hist[i,] <- beta_old
  }
  return(list(beta_hist=beta_hist, n_accept=n_accept, num_it=num_it))
}

get_coefficient <- function(MCMC_lst, method="median", burn_in=0){
  beta_hist <- MCMC_lst$beta_hist
  func <- get(method)
  return(apply(beta_hist[-c(1:burn_in),], MARGIN=2, FUN=func))
}

MCMC_predict <- function(X, MCMC_lst, method="median", burn_in=0){
  X_ <- cbind(1, as.matrix(X))
  beta <- get_coefficient(MCMC_lst, method, burn_in)
  pi <- sigmoid(beta, X_)
  return(as.numeric(pi > 0.5))
}

################################################################################
df <- read.csv("fin-ratio.csv")   # read in data file
X <- subset(df, select=-c(HSI)); y <- df$HSI

MCMC_lst <- MCMC_logistic(X, y, num_it=5e5)
burn_in <- 1e5
y_hat_MCMC <- MCMC_predict(X, MCMC_lst, method="median", 
                           burn_in=burn_in)
MCMC_lst$n_accept / MCMC_lst$num_it
cat("MCMC_median", get_coefficient(MCMC_lst, "median", 
                                   burn_in=burn_in), "\n")
cat("MCMC_median", get_coefficient(MCMC_lst, "mean", 
                                   burn_in=burn_in), "\n")

################################################################################
glm_model <- glm(HSI ~ ., data=df, family="binomial")
y_hat_MLE <- as.numeric(glm_model$fitted.values > 0.5)
cat("MLE", coef(glm_model), "\n")

################################################################################
which(y == 1)
cat("MCMC_median:", mean(y == y_hat_MCMC), "\n")
which(y_hat_MCMC == 1)
cat("MLE:", mean(y == y_hat_MLE), "\n")
which(y_hat_MLE == 1)

################################################################################
MCMC_median <- get_coefficient(MCMC_lst, "median", burn_in=burn_in)
beta_median <- MCMC_lst$beta_hist
par(mfrow=c(1, 2))
for (col in 1:(ncol(X)+1)){
  plot(beta_median[,col], type="l", lwd=2, main="MCMC path", 
       xlab="iterations", ylab=paste("beta", col-1))
  abline(v=burn_in, lty=2, lwd=2, col="blue")
  abline(h=MCMC_median[col], lty=2, lwd=2, col="red")
  
  hist(beta_median[-c(1:burn_in), col], main="MCMC histogram",
       xlab=paste("beta", col-1))
  abline(v=MCMC_median[col], lty=2, lwd=2, col="red")
}