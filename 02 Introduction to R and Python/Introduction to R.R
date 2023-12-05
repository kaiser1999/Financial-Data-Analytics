#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read csv in R
data <- read.csv("../Datasets/fin-ratio.csv")
# Write csv in R
write.csv(data, "../Datasets/fin-ratio_new.csv")

# Function in R
source("My_Function.R")

MyFun(2)

# For loop in R
for (i in 1:10){
  print(i)
}

################################################################################
# Assign data to x except the label y = HSI stock or not
x <- data[-ncol(data)]
# Compute sample means and sample variances for each columns
apply(x, MARGIN=2, FUN=mean)
apply(x, MARGIN=2, FUN=var)

# Compute sample covariance matrix
cov(x)

################################################################################
(A <- matrix(c(1, 3, 5, 2, 6, 4, 2, 3, 1), nrow=3, ncol=3, byrow=T))
(B <- matrix(c(3, 1, 2, 4, 2, 8, 1, 3, 1), nrow=3, ncol=3, byrow=T))

A %*% B                           # Matrix Multiplication
A * B                             # Hadamard Product
solve(A)                          # Inverse

################################################################################
HSBC <- read.csv("../Datasets/0005.HK.csv")
X <- HSBC$Adj.Close
log_return <- diff(log(X))

hist(log_return, breaks="sturges")
qqnorm(log_return)
qqline(log_return)

################################################################################
ks.test(log_return, y=pnorm, mean=mean(log_return), sd=sd(log_return))

KS_test <- function(u, func=pnorm, ...){
  n <- length(u)
  S_n <- (0:n)/n
  z <- sort(u)
  Phi_z <- func(z, ...)
  Term1 <- S_n[2:(n+1)] - Phi_z
  Term2 <- Phi_z - S_n[1:n]
  KS <- max(Term1, Term2)
  crit <- sqrt(-0.5*log(0.05/2)/n)
  list(KS.stat=KS, KS.crit=crit)
}

KS_test(log_return, func=pnorm, mean=mean(log_return), sd=sd(log_return))

################################################################################
library("normtest")
jb.norm.test(log_return)

JB_test <- function(x) {            # function for JB-test
  u <- x - mean(x)
  n <- length(u)                  # sample size
  s <- sd(u)*sqrt((n-1)/n)        # compute population sd
  sk <- sum(u^3)/(n*s^3)          # compute skewness
  ku <- sum(u^4)/(n*s^4)-3        # excess kurtosis
  JB <- n*(sk^2/6+ku^2/24)        # JB test stat
  p <- 1-pchisq(JB,2)             # p-value
  list(JB.stat=JB, p.value=p)     # output
}

JB_test(log_return)

################################################################################
data <- read.csv("../Datasets/Pearson.txt", sep="\t")
model <- lm(Son~., data=data)

plot(Son~., data=data, 
     main="Heights of fathers and their full grown son (in inches)")
abline(model)