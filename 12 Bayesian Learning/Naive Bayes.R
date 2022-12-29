#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("fin-ratio.csv")            # read in dataset

# define HSI as a factor instead of a number
d$HSI <- factor(d$HSI, levels=c(0,1))

# split data into training and testing sets with a ratio 8:2
set.seed(4012)
train_idx <- sample(1:nrow(d), size=floor(0.8*nrow(d)))
train <- d[train_idx,]
test <- d[-train_idx,]

library(e1071)
# Build a Gaussian Naive Bayes classifier with training sets
gnb_clf <- naiveBayes(formula=HSI~., data=train)
y_pred <- predict(gnb_clf, newdata=test[,-7])
table(y_pred, test[,7])

# mean and standard deviation of each feature variable
(mu_0 <- sapply(gnb_clf$tables, "[[", 1))
(mu_1 <- sapply(gnb_clf$tables, "[[", 2))
(sig_0 <- sapply(gnb_clf$tables, "[[", 3))
(sig_1 <- sapply(gnb_clf$tables, "[[", 4))
