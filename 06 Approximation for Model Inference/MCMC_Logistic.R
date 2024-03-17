# same as Risk_Measure.R
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(mvtnorm)

sigmoid <- function(beta, X, y=NA){
  exp_eta <- exp(X %*% beta)
  return(exp_eta / (1 + exp_eta))
}

MCMC_logistic <- function(X, y, num_it=5e5, burn_in=1e5, nu=0.5, 
                          seed=4002){
  # ln P(beta): large cov to be a non-informative prior
  log_prior <- function(beta) sum(dnorm(beta, 0, 100, log=TRUE))
  get_log <- function(x, val=1e-300) log(pmax(x, val))  # avoid log(0)=NA
  
  log_likelihood <- function(beta, X, y){   # ln L(beta)
    pi <- sigmoid(beta, X, y)
    return(sum(y * get_log(pi) + (1 - y) * get_log(1 - pi)))
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
  prog_bar <- txtProgressBar(min=0, max=num_it, width=50, style=3)
  for (i in 2:num_it){
    beta_old <- beta_hist[i-1,]
    beta_new <- rmvnorm(1, beta_old, Cov_nu)

    # Since the proposal distribution, normal, is symmetric, 
    # we have g(beta_old| beta_new) = g(beta_new| beta_old)
    log_target_old <- log_target(matrix(beta_old, ncol=1), X_, y)
    log_target_new <- log_target(t(beta_new), X_, y)
    alpha <- exp(log_target_new - log_target_old)
    if (runif(1) < min(1, alpha)){
      beta_old <- beta_new
      n_accept <- n_accept + 1
    }
    
    beta_hist[i,] <- beta_old
    setTxtProgressBar(prog_bar, i)
  }
  cat("\n")
  return(list(beta_hist=beta_hist, n_accept=n_accept, num_it=num_it,
              burn_in=burn_in))
}

get_coefficient <- function(MCMC_lst, method="median"){
  beta_hist <- MCMC_lst$beta_hist
  burn_in <- MCMC_lst$burn_in
  func <- get(method)
  return(apply(beta_hist[-c(1:burn_in),], MARGIN=2, FUN=func))
}

MCMC_predict <- function(X, MCMC_lst, method="median"){
  X_ <- cbind(1, as.matrix(X))
  beta <- get_coefficient(MCMC_lst, method)
  pi <- sigmoid(beta, X_)
  return(as.numeric(pi > 0.5))
}

################################################################################
df <- read.csv("../Datasets/fin-ratio.csv")   # read in data file
X <- subset(df, select=-c(HSI)); y <- df$HSI

burn_in <- 1e5
MCMC_lst <- MCMC_logistic(X, y, num_it=5e5, burn_in=burn_in)
y_hat_MCMC <- MCMC_predict(X, MCMC_lst, method="median")
MCMC_lst$n_accept / MCMC_lst$num_it
(MCMC_median <- get_coefficient(MCMC_lst, "median"))
(MCMC_mean <- get_coefficient(MCMC_lst, "mean"))

################################################################################
# glm_model <- glm(HSI ~ ., data=df, family="binomial")
# y_hat_MLE <- as.numeric(glm_model$fitted.values > 0.5)
# cat("MLE", coef(glm_model), "\n")

library(glmnet)
# 0 for penalty such that no regularization effect
glm_model <- glmnet(X, y, family="binomial", lambda=c(1e-4, 0),
                    standardize=FALSE)
y_hat_MLE <- predict(glm_model, s=0, newx=as.matrix(X), 
                     type="response") > 0.5
cat("MLE", coef(glm_model, s=0)@x, "\n")

################################################################################
cat("MCMC_median:", mean(y == y_hat_MCMC), "\n")
which(y_hat_MCMC != y)

cat("MLE:", mean(y == y_hat_MLE), "\n")
which(y_hat_MLE != y)

################################################################################
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