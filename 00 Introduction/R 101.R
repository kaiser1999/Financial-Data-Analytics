install.packages("e1071")
library(e1071)

################################################################################
#Set directory: Run this on source instead of Console!!
install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read csv in R
data <- read.csv("fin-ratio.csv")
# Write csv in R
write.csv(data, "fin-ratio_new.csv")

################################################################################
# Assign data to x except the label y = HSI stock or not
x <- data[-ncol(data)]
# Compute sample means and sample variances for each columns
apply(x, MARGIN=2, FUN=mean)
apply(x, MARGIN=2, FUN=var)

# Compute sample covariance matrix
cov(x)

################################################################################
source("My_Function.R")

MyFun(2)

################################################################################
# For loop in R
for (i in 1:5){
  print(i)
}

################################################################################
(A <- matrix(c(1, 3, 5, 2, 6, 4, 2, 3, 1), nrow=3, byrow=T))
(B <- matrix(c(3, 1, 2, 4, 2, 8, 1, 3, 1), nrow=3, byrow=T))

A %*% B                           # Matrix Multiplication
A * B                             # Hadamard Product
solve(A)                          # Inverse