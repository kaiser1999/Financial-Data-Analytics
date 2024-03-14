#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(e1071)
library(caret)

df <- read.csv("../Datasets/credit_scoring_sample.csv")
df <- na.omit(df)

# split df into training and testing sets with a ratio of 8:2
set.seed(4002)
train_idx <- sample(seq_len(nrow(df)), size=floor(0.8*nrow(df)))
df_train <- df[train_idx,]
df_test <- df[-train_idx,]

################################################################################
# Linear SVM
svm_lnr <- svm(SeriousDlqin2yrs~., data=df_train, cost=1,
               type="C-classification", scale=FALSE,
               kernel="linear")
ypred <- predict(svm_lnr, newdata=df_test)
mean(ypred == df_test$SeriousDlqin2yrs)
table(ypred, df_test$SeriousDlqin2yrs)

################################################################################
# RBF SVM
gamma <- function(sig) 1 / (2 * sig^2)
sigma <- c(0.5, 1, 5, 10, 50, 100)

train_acc <- numeric(length(sigma))
test_acc <- numeric(length(sigma))
for (i in seq_along(sigma)) {
  svm_rbf <- svm(SeriousDlqin2yrs~., data=df_train, cost=1, 
                 type="C-classification", scale=FALSE,
                 kernel="radial", gamma=gamma(sigma[i]))
  ypred_train <- predict(svm_rbf, newdata=df_train)
  ypred_test <- predict(svm_rbf, newdata=df_test)
  train_acc[i] <- mean(ypred_train == df_train$SeriousDlqin2yrs)
  test_acc[i] <- mean(ypred_test == df_test$SeriousDlqin2yrs)
}

data.frame(sigma=sigma, gamma=gamma(sigma), 
           train_accuracy=train_acc, test_accuracy=test_acc)

################################################################################
plot(sigma, test_acc, col="orange", type="l", lwd=2, ylab="accuracy", 
     ylim=c(0.75, 1))
lines(sigma, train_acc, col="blue", type="l", lwd=2)

plot(sigma, test_acc, col="orange", type="l", lwd=2, ylab="accuracy")