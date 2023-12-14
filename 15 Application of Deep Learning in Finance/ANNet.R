# Try nnet(x,y) K times and output the best trial
# y is the dependent value; which must be factor if linout=F
library(nnet)

ANNet <- function(X, y, size, maxit=100, linout=F, try=5){
  Best_ANN <- nnet(y~., data=X, size=size, maxit=maxit, linout=linout)
  for (i in 2:try) {
    ANN <- nnet(y~., data=X, size=size, maxit=maxit, linout=linout)
    if (ANN$value < Best_ANN$value) Best_ANN <- ANN	 # save the best model
  }
  return (Best_ANN)
}