#library(tensorflow)
#install_tensorflow()

################################################################################
### Bernoulli Mixture Model ###
################################################################################
library(keras)        # MNIST dataset

# Load MNIST dataset
mnist <- dataset_mnist()
X_train <- mnist$train$x
y_train <- mnist$train$y
X_test <- mnist$test$x
y_test <- mnist$test$y

# Limit the pixel value between 0 and 1 to avoid computational explode
X_train[X_train!=0] <- 1
X_test[X_test!=0] <- 1

# Set index
M <- length(unique(y_train))  # 10: numbers from 0 to 9
img_dim <- dim(X_train)[2:3]  # 28x28 pixels
N_train <- length(y_train)    # 60000 training samples
N_test <- length(y_test)      # 10000 test samples

################################################################################
# i:i-th sample
# m:m-th digit
# Compute posterior probabilities
loglike_func <- function(post, prior, mu){
  loglike <- 0
  mu[mu < 1e-323] <- 1e-323
  mu[mu > 1 - 1e-323] <- 1 - 1e-323
  for (m in 1:M){
    for (n in 1:N_train){
      log_sums <- sum(X_train[n,,]*log(mu[m,,]) 
                      + (1-X_train[n,,])*log(1-mu[m,,]))
      loglike <- loglike + post[n,m]*(log(prior[m]) + log_sums)
    }
  }
  return(loglike)
}

################################################################################
# Initialize the algorithm with prior0 and mu0
prior <- runif(M)
prior <- prior/sum(prior)
mu <- array(data=0, dim=c(M, img_dim))
for (m in 1:M){
  digit <- X_train[which(y_train==(m-1)),,]
  mu[m,,] <- colMeans(digit)
}

################################################################################
# Record prior and likelihood for each iterations
iter <- 10
prior_record <- array(data=0, dim=c(M, iter))
loglike_record <- array(data=0, dim=c(iter, 1))

for (it in 1:iter){
  # Around 70secs for each iteration
  print(paste("no. iteration:", it))
  # Compute posterior probabilities
  posterior <- array(data=0, dim=c(N_train, M))
  nominator <- array(data=0, dim=c(M, 1))
  for (n in 1:N_train){
    for (m in 1:M){
      binm <- (mu[m,,]^X_train[n,,])*(1-mu[m,,])^(1-X_train[n,,])
      nominator[m] <- prior[m]*prod(binm)
    }
    posterior[n,] <- nominator/sum(nominator)
  }
  loglike_record[it] <- loglike_func(posterior, prior, mu)
  
  # Compute new mu and prior
  for (m in 1:M){
    px <- array(data=0, dim=img_dim)
    for (n in 1:N_train){
      px <- px + posterior[n,m]*X_train[n,,]
    }
    mu[m,,] <- px/sum(posterior[,m])
    prior[m] <- sum(posterior[,m])/N_train
  }
  prior <- prior/sum(prior)
  prior_record[,it] <- prior
} # Iterate for iter =10 times with the following output

################################################################################
# Test
y_hat <- array(data=0, dim=c(N_test, 1))
for (n in 1:N_test){
  y_prob <- array(data=0, dim=c(M, 1))
  for (m in 1:M){
    binm <- (mu[m,,]^X_test[n,,]) * (1-mu[m,,])^(1-X_test[n,,])
    y_prob[m] <- prior[m] * prod(binm)
  }
  y_hat[n] <- which.max(y_prob) - 1
}

################################################################################
# Table of predictions and prediction accuracy
table(y_test, y_hat)			              # Confusion matrix
sum(diag(table(y_test, y_hat)))/N_test	# Accuracy

# Plot for the log-likelihood for all 10 iterations
plot(loglike_record, ylab="Log-likelihood", xlab="iterations")

par(mfrow=c(3,4), mar=c(2,2,2,2))
for (m in 1:M) image(t(apply(mu[m,,], 2, rev)))