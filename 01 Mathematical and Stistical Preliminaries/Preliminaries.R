#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
options(digits=4) # Control display into 4 digits

HSI_2002 <- read.csv("../Datasets/fin-ratio.csv")
names(HSI_2002)

X_2002 <- HSI_2002[,1:6] # A 680x6 data matrix
(mu_2002 <- apply(X_2002, 2, mean)) # Mean vector

(S_2002 <- var(X_2002)) # Covariance matrix

(R_2002 <- cor(X_2002)) # Correlation matrix

################################################################################
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