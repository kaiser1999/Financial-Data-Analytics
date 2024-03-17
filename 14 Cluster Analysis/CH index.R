#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
# Try several values of K, choose K so that stat. is maximized
kmstat <- function(X, K){
  km <- kmeans(X, K)              # K-means clustering
  n <- nrow(X)                    # sample size
  wcss <- sum(km$withinss)        # within group ss
  bcss <- km$betweenss            # between group ss
  # km$cluster: the cluster to which each point is allocated
  list(stat=(n-K)*bcss/((K-1)*wcss), km=km)
}

# Try kmeans(X, K) several times and output the best trial
best_km <- function(X, K, trial=5, seed=4002) {
  set.seed(seed)
  r0 <- 0
  for (i in 1:trial) {
    res <- kmstat(X, K)           # new trial
    if (res$stat > r0) {          # update r0 if it is less than r
      r0 <- res$stat; km0 <- res$km
    }
  }
  print(paste0("K=", K, "; stat=", r0))
  km0                             # best cluster
}

################################################################################
set.seed(5)
n0 <- 1000000
K <- 5

# Simulate n0 data points from each distribution
data1 <- rnorm(n0, mean=0, sd=0.1)
data2 <- rnorm(n0, mean=4, sd=0.1)
data3 <- rnorm(n0, mean=-6, sd=0.1)
data4 <- rnorm(n0, mean=10, sd=0.1)
data5 <- rnorm(n0, mean=35, sd=0.1)

data <- matrix(c(data1, data2, data3, data4, data5), ncol=1)

for (k in 2:10){
  best_km(data, k)
}

################################################################################
set.seed(5)
n0 <- 1000000
K <- 5

data <- matrix(runif(K*n0), ncol=1)
for (k in 2:10){
  best_km(data, k)
}

################################################################################
# Case 1: still normal, but with variances larger so that they have overlaps
set.seed(5)
n0 <- 1000000
K <- 5

data1 <- rnorm(n0, mean=0, sd=2)
data2 <- rnorm(n0, mean=4, sd=3)
data3 <- rnorm(n0, mean=-6, sd=4)
data4 <- rnorm(n0, mean=10, sd=6)
data5 <- rnorm(n0, mean=35, sd=6)

data <- matrix(c(data1, data2, data3, data4, data5), ncol=1)

for (k in 2:10){
  best_km(data, k)
}

################################################################################
library(smoothmest)

# Case 2: try double Laplace with large lambda=10
set.seed(5)
n0 <- 1000000
K <- 5

data1 <- rdoublex(n0, mu=0, lambda=10)
data2 <- rdoublex(n0, mu=15, lambda=10)
data3 <- rdoublex(n0, mu=30, lambda=10)
data4 <- rdoublex(n0, mu=-15, lambda=10)
data5 <- rdoublex(n0, mu=-30, lambda=10)

data <- matrix(c(data1, data2, data3, data4, data5), ncol=1)

for (k in 2:10){
  best_km(data, k)
}

################################################################################
library(smoothmest)

# Case 3: try double Laplace with small lambda=0.5
set.seed(5)
n0 <- 1000000
K <- 5

data1 <- rdoublex(n0, mu=0, lambda=0.5)
data2 <- rdoublex(n0, mu=15, lambda=0.5)
data3 <- rdoublex(n0, mu=30, lambda=0.5)
data4 <- rdoublex(n0, mu=-15, lambda=0.5)
data5 <- rdoublex(n0, mu=-30, lambda=0.5)

data <- matrix(c(data1, data2, data3, data4, data5), ncol=1)

for (k in 2:10){
  best_km(data, k)
}
