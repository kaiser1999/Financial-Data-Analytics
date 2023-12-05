#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(nnet)               # load library nnet
X <- iris[,1:4]; y <- as.numeric(iris[,5])
# ANN with single hidden layer and linear output unit
set.seed(1999)
iris.nn <- nnet(X,y,size=2,linout=T)  # linear output
summary(iris.nn)            # summary of output

################################################################################
M <- iris.nn$wts
h1 <- matrix(M[1:10], ncol=ncol(X)+1, byrow=T)
b1 <- h1[,1]; W1 <- h1[,2:(ncol(X)+1)]

b2 <- M[11]
W2 <- matrix(M[12:13], ncol=2, byrow=T)

x <- t(X[c(1, 51, 101),])
logistic <- function(x) return (exp(x)/(1+exp(x)))

x
(z1 <- W1 %*% x + b1)
(a1 <- logistic(z1))
(z2 <- W2 %*% a1 + b2)
iris.nn$fitted.values[c(1, 51, 101)]

################################################################################
pred <- round(iris.nn$fit)  # round the fitted values
table(pred, y)              # classification table

################################################################################
d <- read.csv("fin-ratio.csv")
names(d)
X <- subset(d, select=-c(HSI)); y <- d$HSI
set.seed(4002)
fin.nn <- nnet(X,y,size=2,linout=T,maxit=200)
# set max. number of iterations to 200
pred <- round(fin.nn$fit)   # round the fitted values
table(pred, y)              # classification table

################################################################################
source("ann.R")
d <- read.csv("fin-ratio.csv")
X <- subset(d, select=-c(HSI)); y <- d$HSI
set.seed(4002)
fin.nn <- ann(X,y,size=2,linout=T,try=10) # 10 trials
fin.nn$value                # the best result
summary(fin.nn)             # the best weights 
pred <- round(fin.nn$fit)   # round the fitted values
table(pred, y)              # classification table

################################################################################
X <- iris[,1:4]; y <- as.factor(iris[,5])
iris.nn <- ann(X,y,size=2,maxit=200,try=10)    # 10 trials, logistic
iris.nn$value               # best value
summary(iris.nn)            # display weights
pred <- max.col(iris.nn$fit)# find column name with max. fitted values
table(pred, y)              # classification table

################################################################################
d <- read.csv("fin-ratio.csv")
X <- subset(d, select=-c(HSI)); y <- d$HSI
y <- as.factor(d$HSI)       # y as factor
set.seed(4002)
fin.nn <- ann(X,y,size=2,maxit=200,try=10)
fin.nn$value                # best value
summary(fin.nn)
pred <- fin.nn$fit > 0.5    # check if it belongs to group 1
table(pred, y)              # classification table

################################################################################
logistic <- function(x) 1/(1+exp(-x))

X <- matrix(c(0.4,0.7,0.8,0.9,1.3,1.8,-1.3,-0.9),ncol=2,byrow=T)
y <- c(0,0,1,0) 				# target value
# hidden layer bias and weights
W1 <- matrix(c(0.1,-0.2,0.1,0.4,0.2,0.9),nrow=2,byrow=T)
# output layer bias and weights
W2 <- matrix(c(0.2,-0.5,0.1),nrow=1)

X1 <- cbind(1, X)
h <- logistic(W1 %*% t(X1))       # logistic hidden h'
h <- rbind(1, h)
o <- W2 %*% h                     # linear output o'
(err <- y - o)                    # output error
(mean_sse <- mean(err^2))         # mean SSE

################################################################################
lr <- 0.5                         # learning rate: $\eta$
n <- length(y)
del2 <- -2*err                    # output layer $\delta_2$
Delta_W2 <- -lr*del2 %*% t(h)     # $\Delta W2 = -\eta \delta_2 (h')^T$
new_W2 <- W2 + Delta_W2 / n       # new output weights: $W2 = W2 + \Delta W2$

del1 <- (t(W2) %*% del2)*h*(1-h)  # hidden layer $\delta_1$
del1 <- del1[-1,]                 # remove from cbind(1, X)
Delta_W1 <- -lr*del1 %*% X1       # $\Delta W1 = -\eta \delta_1 x^T$
new_W1 <- W1 + Delta_W1 / n       # new hidden weights: $W1 = W1 + \Delta W1$

new_h <- logistic(new_W1 %*% t(X1))
new_h <- rbind(1, new_h)
new_o <- new_W2 %*% new_h
new_err <- y - new_o
(new_mean_sse <- mean(new_err^2)) # new mean SSE