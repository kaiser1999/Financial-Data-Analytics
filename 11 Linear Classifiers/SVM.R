#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("../Datasets/fin-ratio.csv")      # read in dataset
# define HSI as a factor instead of a number for svm
d$HSI <- factor(d$HSI, levels=c(0,1))

# split data into training and testing sets with a ratio of 7:3
set.seed(4002)
train_ind <- sample(seq_len(nrow(d)), size=floor(0.7*nrow(d)))
d_train <- d[train_ind,]
d_test <- d[-train_ind,]

library(e1071)  # load the e1071 library for svm
svm_clf <- svm(formula=HSI~., data=d_train, kernel='linear', 
               scale=TRUE, cost=1)
y_pred <- predict(svm_clf, newdata=d_test)
table(y_pred, d_test$HSI)             # tabulate results

plot(svm_clf, d_test, CFTP~ln_MV, color.palette=cm.colors)