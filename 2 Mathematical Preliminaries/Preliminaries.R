setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

options(digits=4) # Control display into 4 digits

HSI_2002 <- read.csv("Financial ratio_2002.csv")
names(HSI_2002)

X_2002 <- HSI_2002[,1:6] # A 680x6 data matrix
(mu_2002 <- apply(X_2002, 2, mean)) # Mean vector

(S_2002 <- var(X_2002)) # Covariance matrix

(R_2002 <- cor(X_2002)) # Correlation matrix

############################################################

options(digits=4)

det(solve(S_2002))
1 / det(S_2002)

eig_2002 <- eigen(S_2002)
eig_2002$values
(H_2002 <- eig_2002$vector)

round(t(H_2002) %*% H_2002, 3)
t(H_2002[,1]) %*% H_2002[,2]
round(t(H_2002) %*% S_2002 %*% H_2002, 3)
(D_2002 <- diag(eig_2002$values))

sqrt_S_2002 <- H_2002 %*% sqrt(D_2002) %*% t(H_2002)
sqrt_S_2002 %*% sqrt_S_2002
S_2002

############################################################

set.seed(4002)

pse_uni_gen <- function(a=16807, c=1, m=(2^31)-1, seed=123456789, 
                        size=1, burn_in=1000){
  x <- (a*seed + c) %% m
  for (i in 1:(size+burn_in-1)){
    x <- c(x, (a*x[length(x)] + c) %% m)
  }
  return (x[(burn_in+1):length(x)] / m)
}

pse_uniform_gen <- function(lower=0, upper=1, seed=123456789,
                            size=1, burn_in=1000){
  U <- pse_uni_gen(seed=seed, size=size, burn_in=burn_in)
  return (lower+(upper - lower)*U)
}

pse_sample <- pse_uniform_gen(size=10000)
built_in_sample <- runif(10000)
hist(pse_sample, col=rgb(red=0,green=0,blue=1.0,alpha=0.5), 
     main="Histogram")
hist(built_in_sample, col=rgb(red=1.0,green=0,blue=0,alpha=0.5),
     add=TRUE)
legend('topright',legend=c("Pseudo Generator","R Generator"),
       fill=c(rgb(red=0,green=0,blue=1.0,alpha=0.5),
              rgb(red=1.0,green=0,blue=0,alpha=0.5)))

############################################################

set.seed(4012)

pse_exp_gen <- function(lamb, seed=123456789, 
                        size=1, burn_in=1000){
  U <- pse_uni_gen(seed=seed, size=size, burn_in=burn_in)
  X <- -(1/lamb)*log(1-U)
  return (X)
}

pse_sample <- pse_exp_gen(lamb=1, size=10000)
built_in_sample <- rexp(10000, 1)
hist(pse_sample, col=rgb(red=0,green=0,blue=1.0,alpha=0.5),
     main="Histogram")
hist(built_in_sample, col=rgb(red=1.0,green=0,blue=0,alpha=0.5),
     add=TRUE)
curve(10000*dexp(x, rate=1, log=FALSE), add=TRUE, col='black', 
      lwd=2.5)
legend('topright',legend=c("Pseudo Generator","R Generator"),
       fill=c(rgb(red=0,green=0,blue=1.0,alpha=0.5),
              rgb(red=1.0,green=0,blue=0,alpha=0.5)))

############################################################

set.seed(4002)

pse_norm_gen <- function(mu=0.0, sigma=1.0, seed=123456789, 
                         size=1, burn_in=1000){
  U <-  pse_uni_gen(seed=seed, size=2*size, burn_in=burn_in)
  U1 <- U[1:size]
  U2 <- U[(size+1):(2*size)]
  Z0 <- sqrt(-2*log(U1))*cos(2*pi * U2)
  Z1 <- sqrt(-2*log(U1))*sin(2*pi * U2)
  return (Z0*sigma + mu)
}
pse_sample <- pse_norm_gen(mu=0, sigma=1, size=10000)
built_in_sample <- rnorm(10000)
hist(pse_sample, col=rgb(red=0,green=0,blue=1.0,alpha=0.5),
     main="Histogram")
hist(built_in_sample, col=rgb(red=1.0,green=0,blue=0,alpha=0.5),
     add=TRUE)
legend('topright',legend=c("Pseudo Generator","R Generator"),
       fill=c(rgb(red=0,green=0,blue=1.0,alpha=0.5),
              rgb(red=1.0,green=0,blue=0,alpha=0.5)))