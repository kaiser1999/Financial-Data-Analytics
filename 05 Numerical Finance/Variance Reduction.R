set.seed(4002)

n <- 10000
psi <- c()
for (i in 1:1000){
  u <- runif(n)
  psi <- c(psi, sum(u^3*(1-u)^2.5)/n)
}
mean(psi)
beta(4, 3.5)
var(psi)

################################################################################
set.seed(4002)

n <- 10000
psi_A <- c()
for (i in 1:1000){
  u <- runif(n/2)
  v <- 1 - u
  psi_A <- c(psi_A, sum(u^3*(1-u)^2.5 + v^3*(1-v)^2.5)/n)
}
mean(psi_A)
var(psi_A)

################################################################################
set.seed(4002)

n <- 10000
psi_C <- c()
mu_y <- 2/7
for (i in 1:1000){
  u <- runif(n)
  x <- u^3*(1-u)^2.5
  y <- (1-u)^2.5
  psi_C <- c(psi_C, mean(x) - cov(x, y)/var(y)*(mean(y) - mu_y))
}
mean(psi_C)
var(psi_C)

################################################################################
set.seed(4002)

n <- 10000
psi_MC <- c()
mu_y <- c(2/7, 1/4)
for (i in 1:1000){
  u <- runif(n)
  x <- u^3*(1-u)^2.5
  y <- cbind((1-u)^2.5, u^3)
  inv_S <- solve(var(y))
  diff_y <- apply(y, 2, mean) - mu_y
  psi_MC <- c(psi_MC, mean(x) - cov(x, y) %*% inv_S %*% diff_y)
}
mean(psi_MC)
var(psi_MC)

################################################################################
set.seed(4002)

n <- 10000
psi_S <- c()
J <- 10
for (i in 1:1000){
  u_j <- matrix(runif(n), ncol=J)
  u_j <- sweep(u_j, 2, 0:(J-1), "+")/J
  psi_S <- c(psi_S, sum(u_j^3*(1-u_j)^2.5)/n)
}
mean(psi_S)
var(psi_S)