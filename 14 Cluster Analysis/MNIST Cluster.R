#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(keras)        # MNIST dataset
library(nnet)
source("KMeansCluster.R")

# Load MNIST dataset
mnist <- dataset_mnist()
X_train <- mnist$train$x / 255
y_train <- mnist$train$y
X_test <- mnist$test$x / 255
y_test <- mnist$test$y

X_train <- matrix(X_train, nrow=length(y_train))
X_test <- matrix(X_test, nrow=length(y_test))
K <- 10

predict.kmeans <- function(object, newdata) {
  # Squared Euclidean distance of each sample to each cluster center
  sd_by_center <- apply(object$centers, 1, function(x) {
    colSums((t(newdata) - x)^2)
  })
  apply(sd_by_center, 1, which.min)
}

################################################################################
y_test[y_test == 0] <- K

start <- Sys.time()
km <- best_km(X_train, K)             # K-means with K=10
y_hat_kmeans <- predict.kmeans(km, newdata=X_test)
(tab <- table(y_hat_kmeans, y_test))  # Confusion matrix
(pos_dict <- apply(tab, 1, which.max))
y_pred <- pos_dict[as.character(y_hat_kmeans)]
(tab <- table(y_pred, y_test))        # Confusion matrix
mean(y_pred == y_test)                # Accuracy
Sys.time() - start                    # K-means

################################################################################
start <- Sys.time()
pca <- prcomp(X_train, center=TRUE, scale=FALSE)
s <- pca$sdev
t <- sum(s^2)           # Compute total variance
cumvar <- cumsum(s^2/t) # Cumulative sum of proportion of variance
(idx <- which(cumvar >= 0.95)[1])

X_train_pca <- predict(pca, newdata=X_train)[,1:idx]
X_test_pca <- predict(pca, newdata=X_test)[,1:idx]
km_pca <- best_km(X_train_pca, K)     # K-means with K=10
y_hat_pca <- predict.kmeans(km_pca, newdata=X_test_pca)
(tab <- table(y_hat_pca, y_test))     # Confusion matrix
(pos_dict <- apply(tab, 1, which.max))
y_pred <- pos_dict[as.character(y_hat_pca)]
(tab <- table(y_pred, y_test))        # Confusion matrix
mean(y_pred == y_test)                # Accuracy
Sys.time() - start                    # PCA + K-means

################################################################################
library(class)
y_test[y_test == K] <- 0

start <- Sys.time()
y_hat_knn <- knn(train=X_train, test=X_test, 
                 cl=factor(y_train), k=3) # KNN with K=3
(tab <- table(y_hat_knn, y_test))     # Confusion matrix
mean(y_hat_knn == y_test)             # Accuracy
Sys.time() - start                    # KNN

################################################################################
start <- Sys.time()
y_hat_pca <- knn(train=X_train_pca, test=X_test_pca, 
                 cl=factor(y_train), k=3) # KNN with K=3
(tab <- table(y_hat_pca, y_test))     # Confusion matrix
mean(y_hat_pca == y_test)             # Accuracy
Sys.time() - start                    # PCA + KNN