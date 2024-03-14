#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("KMeansCluster.R")

################################################################################
### iris flower ###
################################################################################
X_iris <- iris[,-5]               # remove species label from iris
plot(X_iris, col=iris[,5])        # plot observations with color

set.seed(4002)
(km_iris <- kmeans(X_iris, 3))    # K-means clustering with K=3
km_iris$betweenss                 # between group sum of squares
km_iris$withinss                  # within group sum of squares

################################################################################
table(km_iris$cluster, iris[,5])  # Classification table

################################################################################
### HSI ###
################################################################################
df_HSI <- read.csv("../Datasets/fin-ratio.csv")    # read in HSI dataset
X_HSI <- df_HSI[,-7]              # remove HSI label
plot(X_HSI, col=df_HSI[,7]+1)     # plot observations with color

set.seed(4002)
km_HSI <- kmeans(X_HSI, 2)        # K-means clustering with K=2
table(km_HSI$cluster, df_HSI[,7]) # Classification table

################################################################################
### iris flower ###
################################################################################
X1 <- X_iris[km_iris$cluster==1,] # select group by cluster label
X2 <- X_iris[km_iris$cluster==2,]
X3 <- X_iris[km_iris$cluster==3,]
(n1 <- nrow(X1)); (n2 <- nrow(X2)); (n3 <- nrow(X3))  # cluster size

apply(X1, 2, mean)                # cluster mean
apply(X2, 2, mean)
apply(X3, 2, mean)
sum(diag((n1-1)*var(X1)))         # tr(SSCP)
sum(diag((n2-1)*var(X2)))         # within group sum of squares
sum(diag((n3-1)*var(X3)))

m <- apply(X_iris, 2, mean)       # overall mean  
# (group mean - overall mean)^2
dm <- sweep(km_iris$centers, 2, m, FUN="-")^2
sum(km_iris$size*rowSums(dm))     # between group sum of squares

################################################################################
### iris flower ###
################################################################################
km_iris2 <- best_km(X_iris, 2)        # try K=2
km_iris3 <- best_km(X_iris, 3)        # try K=3
km_iris4 <- best_km(X_iris, 4)        # try K=4
km_iris5 <- best_km(X_iris, 5)        # try K=5

################################################################################
table(km_iris3$cluster, iris[,5])     # classification table

################################################################################
### Cleaned HSI ###
################################################################################
df_cHSI <- read.csv("../Datasets/fin-ratio_cleansed.csv") # read in cleansed HSI dataset
X_cHSI <- df_cHSI[,-7]                # remove HSI label
km_cHSI2 <- best_km(X_cHSI, 2)        # try K=2
km_cHSI3 <- best_km(X_cHSI, 3)        # try K=3
km_cHSI4 <- best_km(X_cHSI, 4)        # try K=4
km_cHSI5 <- best_km(X_cHSI, 5)        # try K=5

################################################################################
table(km_cHSI2$cluster, df_cHSI[,7])  # classification table

par(mfrow=c(2,3))                     # 2x3 multi-frame graphic
# boxplots for each variable
for (i in 1:ncol(X_cHSI)){
  boxplot(X_cHSI[,i]~km_cHSI2$cluster, 
          xlab="HSI", ylab=names(X_cHSI)[i])
}
