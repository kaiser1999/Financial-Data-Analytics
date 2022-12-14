#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
### iris flower ###
################################################################################
X <- iris[,-5]                    # remove species label from iris
set.seed(4002)
(km_iris <- kmeans(X, 3))         # K-means clustering with K=3
km_iris$betweenss                 # between group sum of squares
km_iris$withinss                  # within group sum of squares
plot(X, col=km_iris$cluster)      # plot observations with color

################################################################################
table(km_iris$cluster, iris[,5])  # Classification table

################################################################################
### HSI ###
################################################################################
d <- read.csv("fin-ratio.csv")    # read in fin-ratio dataset
X <- d[,-7]                       # remove HSI label
set.seed(4002)
km_HSI <- kmeans(X, 2)            # K-means clustering with K=2
plot(X, col=km_HSI$cluster)       # plot observations with color
table(km_HSI$cluster, d[,7])      # Classification table

################################################################################
### iris flower ###
################################################################################
X <- iris[,-5]                    # remove species label from iris
X1 <- X[km_iris$cluster==1,]      # select group by cluster label
X2 <- X[km_iris$cluster==2,]
X3 <- X[km_iris$cluster==3,]
(n1 <- nrow(X1)); (n2 <- nrow(X2)); (n3 <- nrow(X3))  # cluster size

apply(X1, 2, mean)                # cluster mean
apply(X2, 2, mean)
apply(X3, 2, mean)
sum(diag((n1-1)*var(X1)))         # tr(SSCP)
sum(diag((n2-1)*var(X2)))         # within group sum of squares
sum(diag((n3-1)*var(X3)))

m <- apply(X, 2, mean)            # overall mean  
m <- matrix(rep(m, 3), nrow=3, byrow=T)   # matrix with row m
dm <- (km_iris$centers - m)^2     # (group mean - overall mean)^2
colsum <- apply(dm, 1, sum)       # column sum
sum(km_iris$size*colsum)          # between group sum of squares

################################################################################
# try several values of K, choose K so that stat. is maximized
kmstat <- function(X, K){
  km <- kmeans(X, K)              # K-means clustering
  ng <- km$size                   # size of each cluster
  n <- nrow(X)                    # sample size
  wcss <- sum(km$withinss)        # within group ss
  bcss <- km$betweenss            # between group ss
  # km$cluster: the cluster to which each point is allocated
  out <- list((n-K)*bcss/((K-1)*wcss), ng, km$cluster)
  names(out) <- c("stat", "size", "cluster")
  return(out)
}

################################################################################
# Try kmeans(X, K) several times and output the best trial
km <- function(X, K, trial=5) {
  res0 <- kmstat(X, K)            # result of the first trial
  r0 <- res0$stat                 # stat from the first trial
  for (i in 2:trial) {
    res <- kmstat(X, K)           # new trial 
    if (res$stat > r0) {          # update r0 & res if it is better 
      r0 <- res$stat
      res0 <- res                 
    }
  }
  cat("cluster size=", res0$size, "\n") # cluster size
  cat("stat=", res0$stat, "\n")         # display stat
  return(res0$cluster)                  # cluster label
}

################################################################################
### iris flower ###
################################################################################
X <- iris[,-5]
set.seed(4002)
km2 <- km(X, 2)                   # try K=2
km3 <- km(X, 3)                   # try K=3
km4 <- km(X, 4)                   # try K=4
km5 <- km(X, 5)                   # try K=5

################################################################################
table(km3, iris[,5])              # classification table

################################################################################
### Cleaned HSI ###
################################################################################
d <- read.csv("fin-ratio_cleaned.csv")   # read in cleaned HSI dataset
X <- d[,-7]                       # remove HSI label
set.seed(4002)
km2 <- km(X, 2)                   # try K=2
km3 <- km(X, 3)                   # try K=3
km4 <- km(X, 4)                   # try K=4
km5 <- km(X, 5)                   # try K=5

################################################################################
par(mfrow=c(2,3))                 # 2x3 multi-frame graphic
# boxplots for each variable
boxplot(X[,1]~km2, main=names(X)[1])		        
boxplot(X[,2]~km2, main=names(X)[2])
boxplot(X[,3]~km2, main=names(X)[3])
boxplot(X[,4]~km2, main=names(X)[4])
boxplot(X[,5]~km2, main=names(X)[5])
boxplot(X[,6]~km2, main=names(X)[6])
