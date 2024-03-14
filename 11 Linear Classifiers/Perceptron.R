#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
perceptron <- function(X, y, eta=1e-3, n_iter=1e5){
  X <- as.matrix(X); y <- as.numeric(y)
  X_1 <- cbind(1, X)
  n <- nrow(X_1); p <- ncol(X_1)
  omega <- rep(1/p, p)                  # initialize weight vector
  for (it in 0:(n_iter-1)){
    i <- it %% n + 1                    # R index starts from 1
    if (i == 1) new_idx <- sample(1:n, n, replace=FALSE)
    j <- new_idx[i]
    if(y[j] * omega %*% X_1[j,] < 0) {  # misclassified
      omega <- omega + eta * y[j] * X_1[j,]
    } 
  }
  return (list(omega=omega, y_pred=sign(X_1 %*% omega)))
}

################################################################################
df <- read.csv("../Datasets/fin-ratio.csv")
X <- df[-ncol(df)]
y <- df$HSI
y[y==0] <- -1                         # transform the output to {-1, 1}

set.seed(4002)
model <- perceptron(X, y, eta=0.01, n_iter=1e6)
(conf <- table(model$y_pred, y))      # confusion matrix
sum(diag(conf))/length(y)
model$omega