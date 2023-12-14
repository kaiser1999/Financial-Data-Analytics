#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(nnet)
library(devtools)
source_url('https://gist.githubusercontent.com/Peque/41a9e20d6687f2f3108d/raw/85e14f3a292e126f1454864427e3a189c2fe33f3/nnet_plot_update.r')
set.seed(4012)

df <- iris
classes <- factor(df$Species)
levels(classes) <- 1:length(levels(classes))
df$Species <- classes
df <- apply(df, 2, as.numeric)
# One hidden layer with 2 neurons and linear output unit; ncol(df)=5
iris.nn <- nnet(x=df[,-ncol(df)], y=df[,ncol(df)], size=2, linout=TRUE)
summary(iris.nn)

plot.nnet(iris.nn, wts.only = F)

df[c(1, 51, 101),]

sum((df[,ncol(df)] - iris.nn$fitted.values)^2)    # MSE
pred <- round(iris.nn$fit)	# round the fitted values
table(df[,ncol(df)], pred)		# classification table

################################################################################
df <- read.csv("../Datasets/fin-ratio.csv")
fin.nn <- nnet(x=df[,-ncol(df)], y=df$HSI, size=2, linout=TRUE)
summary(fin.nn)

pred <- round(fin.nn$fit)
table(df$HSI, pred)

################################################################################
### ANNet with linout=T ###
################################################################################
set.seed(4012)
source("ANNet.R")

df <- read.csv("../Datasets/fin-ratio.csv")
fin.nn <- ANNet(X=df[,-ncol(df)], y=df$HSI, size=2, linout=T, try=10)
fin.nn$value
summary(fin.nn)

pred <- round(fin.nn$fit)
(conf <- table(df$HSI, pred)) 				# confusion matrix
sum(diag(conf)) / length(df$HSI)      # accuracy

################################################################################
### ANNet with linout=F ###
################################################################################
set.seed(4012)
library(mltools)
library(data.table)
source("ANNet.R")

df <- iris
classes <- factor(df$Species)
levels(classes) <- 1:length(levels(classes))
df$Species <- classes
df <- apply(df, 2, as.numeric)

train_idx <- sample(1:nrow(df), size=round(0.7*nrow(df)))
df_train <- df[train_idx,]
X_train <- df_train[,-ncol(df)]
y_train <- as.factor(df_train[,ncol(df)])
df_test <- df[-train_idx,]
X_test <- df_test[,-ncol(df_test)]
y_test <- df_test[,ncol(df_test)]

iris.nn <- ANNet(X_train, y_train, size=2, linout=F, maxit=200, try=10)
-sum(one_hot(as.data.table(y_train))*log(iris.nn$fitted.values))
iris.nn$value
summary(iris.nn)		             # display weights

pred <- max.col(iris.nn$fitted.values)
table(y_train, pred)						# confusion matrix

################################################################################
# weights matrix from i to h
h1 <- matrix(iris.nn$wts[1:10], nrow=2, byrow=T)
(b1 <- h1[,1])
(W1 <- h1[,-1])

# weights matrix from h to o
h2 <- matrix(iris.nn$wts[11:length(iris.nn$wts)], nrow=3, byrow=T)
(b2 <- h2[,1])
(W2 <- h2[,-1])

logistic <- function(x) {1/(1+exp(-x))}             # logistic function
softmax <- function(x) {t(exp(x))/colSums(exp(x))}  # softmax function

a1 <- logistic(W1 %*% t(X_test) + b1)		            # the output a^{(1)}
a2 <- softmax(W2 %*% a1 + b2)		                    # compute fitted values

y_pred <- max.col(a2)		                      # logistic / round(a2) linear
table(y_test, y_pred)

# find the column number of the max. fitted values
prob <- predict(iris.nn, newdata=df_test)
pred <- max.col(prob)
table(y_test, pred)								                  # confusion matrix

################################################################################
set.seed(4012)
source("ANNet.R")

df <- read.csv("../Datasets/fin-ratio.csv")		      
train_idx <- sample(1:nrow(df), size=round(0.7*nrow(df)))
df_train <- df[train_idx,]
X_train <- df_train[,-ncol(df)]
y_train <- as.factor(df_train$HSI)	
df_test <- df[-train_idx,]
X_test <- df_test[,-ncol(df)]
y_test <- as.factor(df_test$HSI)	

write.csv(df_train, "../Datasets/fin-ratio_train.csv")
write.csv(df_test, "../Datasets/fin-ratio_test.csv")

fin.nn <- ANNet(X_train, y_train, size=2, linout=F, maxit=200, try=20)
fin.nn$value				# best value
summary(fin.nn)

pred <- (fin.nn$fit > 1/2)*1
# classification table for training data df_train
table(y_train, pred)