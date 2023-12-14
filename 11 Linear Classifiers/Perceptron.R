#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
perceptron <- function(X_train, y_train, eta=1e-3, epochs=1e4){
  X_1 <- cbind(1, X_train)
  omega <- rep(1/ncol(X_1), ncol(X_1))   # initialize weight vector
  for (ep in 1:epochs){
    for (i in sample(1:length(y_train), length(y_train))){
      if (y_train[i] * omega %*% X_1[i,] < 0) {   # misclassified
        omega <- omega + eta * y_train[i] * X_1[i,]
      }
    }
  }
  return(list(omega=omega, y_pred=sign(X_1 %*% omega)))
}

################################################################################
df <- read.csv("../Datasets/fin-ratio.csv")
X <- df[-ncol(df)]
y <- df[ncol(df)]
y[y==0] <- -1                   # transform the output to {-1, 1}

set.seed(4012)
# training:testing = 7:3 and shuffle the training dataset
train_idx <- sample(1:nrow(df), size=round(0.7*nrow(df)))
X_train <- as.matrix(X[train_idx,]); y_train <- y[train_idx,]
X_test <- as.matrix(X[-train_idx,]); y_test <- y[-train_idx,]

# prediction on testing dataset
model <- perceptron(X_train, y_train, eta=0.05, epochs=1e4)
y_pred <- sign(cbind(1, X_test) %*% model$omega)
(conf <- table(y_test, y_pred)) 	      # confusion matrix
sum(diag(conf))/length(y_test)

################################################################################
### XOR ###
################################################################################
X <- matrix(c(0,0, 1,1, 0,1, 0,1), ncol=2)
y <- c(1, 0, 0, 1)
y[y==0] = -1                  # transform the output to {-1, 1}

set.seed(4012)
model <- perceptron(X, y, eta=0.05, epochs=1e3)
y_pred <- factor(model$y_pred, levels=c(-1, 1))
(conf <- table(y, y_pred)) 	      # confusion matrix
sum(diag(conf))/length(y)
model$omega

par(mar=c(4.3,4.3,2,2))
plot(X, col=y+2, cex=2, pch=16, xlim=c(-0.2, 1.2), ylim=c(-0.2,1.2),
     main="XOR", xlab=expression(x^{(1)}), ylab=expression(x^{(2)}))
w0 <- model$omega[1]
w1 <- model$omega[2]
w2 <- model$omega[3]
abline(-w0/w2, -w1/w2)
text(0.1, -0.1, expression(omega[0]+omega[1]*x^{(1)}+omega[2]*x^{(2)} == 0))