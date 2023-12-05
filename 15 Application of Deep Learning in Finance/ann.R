# x is the matrix of input variable
# y is the dependent value; which must be factor if linout=F
library(nnet)
ann <- function(x,y,size,maxit=100,linout=F,try=5) {
  best <- nnet(y~.,data=x,size=size,maxit=maxit,linout=linout)
  for (i in 2:try) {
    ann <- nnet(y~.,data=x,size=size,maxit=maxit,linout=linout)
    if (ann$value < best$value) best <- ann # save best ann
  }
  return (best)		                  # return the results
}