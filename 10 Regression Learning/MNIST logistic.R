library(keras)        # MNIST dataset
library(nnet)

# Load MNIST dataset
mnist <- dataset_mnist()
X_train <- mnist$train$x / 255
y_train <- mnist$train$y
X_test <- mnist$test$x / 255
y_test <- mnist$test$y

X_train <- matrix(X_train, nrow=length(y_train))
X_test <- matrix(X_test, nrow=length(y_test))

df_train <- data.frame(cbind(X_train, y_train))
names(df_train) <- c(paste0('x_', 1:ncol(X_train)), 'y')
df_test <- data.frame(X_test)
names(df_test) <- paste0('x_', 1:ncol(X_train))

start <- Sys.time()
set.seed(4002)
mnl <- multinom(y~., data=df_train, MaxNWts=1e6, maxit=200, trace=F)
y_hat <- predict(mnl, newdata=df_test, type="class")
table(y_hat, y_test)      # Confusion matrix
mean(y_hat == y_test)     # Accuracy
Sys.time() - start        # Logistic regression with full training set

################################################################################
start <- Sys.time()
pca <- prcomp(X_train, center=TRUE, scale=FALSE)
s <- pca$sdev
t <- sum(s^2)           # Compute total variance
cumvar <- cumsum(s^2/t) # Cumulative sum of proportion of variance
(idx <- which(cumvar >= 0.95)[1])

X_train_pca <- predict(pca, newdata=X_train)[,1:idx]
X_test_pca <- predict(pca, newdata=X_test)[,1:idx]
df_train_pca <- data.frame(cbind(X_train_pca, y_train))
names(df_train_pca) <- c(paste0('x_', 1:idx), 'y')
df_test_pca <- data.frame(X_test_pca)
names(df_test_pca) <- paste0('x_', 1:idx)

set.seed(4002)
mnl_pca <- multinom(y~., data=df_train_pca, MaxNWts=1e6, maxit=200, 
                    trace=F)
y_hat_pca <- predict(mnl_pca, newdata=df_test_pca, type="class")
table(y_hat_pca, y_test)    # Confusion matrix
mean(y_hat_pca == y_test)   # Accuracy
Sys.time() - start          # PCA + Logistic regression