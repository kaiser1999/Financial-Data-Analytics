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
