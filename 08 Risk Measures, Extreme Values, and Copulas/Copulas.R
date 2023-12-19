setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("../Datasets/stock_1999_2002.csv", row.names=1) # read in data file
d <- as.ts(d)
returns <- (lag(d) - d)/d
colnames(returns) <- paste0(colnames(d), "_Return")
u1 <- returns[,"HSBC_Return"]
u2 <- returns[,"CLP_Return"]
u3 <- returns[,"CK_Return"]
n_sim <- 1e5

par(mfrow=c(1,3))
# Q-Q plot for empirical marginals
col <- c("blue", "orange", "green")
n_days <- nrow(returns)
i <- ((1:n_days) - 0.5) / n_days
for (k in 1:ncol(returns)){
  q <- quantile(returns[,k], probs=i, type=4, names=FALSE)
  qqplot(q, sort(returns[,k]), col=col[k], 
         xlab="Empirical quantiles", ylab="Returns quantiles",
         main=paste0(colnames(returns)[k], "'s return Q-Q Plot"))
  abline(lsfit(q, sort(returns[,k])), lwd=2)
}
par(mfrow=c(1, 1))

library(copula)  # Package for copula computation
empirical_marginals <- function(x) pobs(x)
empirical_quantile <- function(p, samples){
  q <- matrix(NA, nrow=nrow(p), ncol=ncol(p))
  for (k in 1:ncol(p)){
    q[,k] <- quantile(samples[,k], probs=p[,k], type=4, names=FALSE)
  }
  return (q)
}

# using empirical as the marginal distribution
emp_u <- empirical_marginals(returns)

################################################################################
# Assume a normal-copula with ncol(d)=3
# P2p: array of elements of upper triangular matrix
N.cop <- normalCopula(dim=ncol(d), dispstr="un")
fit <- fitCopula(N.cop, emp_u, "ml")
(rho <- coef(fit))
N.cop_fit <- normalCopula(rho, dim=ncol(d), dispstr="un")
set.seed(4002)
# Generate random samples u~U(0, 1) from the fitted gaussian copula
u_sim_N <- rCopula(n_sim, N.cop_fit)
colnames(u_sim_N) <- colnames(d)
pairs(u_sim_N[1:1e3,], col="blue")        # only show the first 1000
cor(u_sim_N)
cor(returns)

# Obtain returns based on empirical marginals
return_sim_N <- empirical_quantile(u_sim_N, returns)
colnames(return_sim_N) <- colnames(d)
pairs(return_sim_N[1:1e3,], col="green")  # only show the first 1000

################################################################################
Mahalanobis2 <- function(X){
  mu <- apply(X, 2, mean)
  inv_Sig <- solve(cov(X))
  X_minus_mu <- sweep(X, 2, mu, FUN="-")
  return (rowSums((X_minus_mu %*% inv_Sig) * X_minus_mu))
  
}

QQ_Plot <- function(sim_data, raw_data, col="blue"){
  n_days <- length(raw_data)
  i <- ((1:n_days) - 0.5) / n_days
  q <- quantile(sim_data, probs=i, type=4, names=FALSE)
  
  qqplot(q, sort(raw_data), col=col,
         xlab="Empirical quantiles", ylab="Returns quantiles", 
         main="Squared Mahalanobis Q-Q Plot with empeirical marginals")
  abline(lsfit(q, sort(raw_data)), lwd=2)
}

returns_md2 <- Mahalanobis2(returns)

################################################################################
sim_N_md2 <- Mahalanobis2(return_sim_N)
QQ_Plot(sim_N_md2, returns_md2, col="blue")

i <- ((1:n_days) - 0.5) / n_days
q <- qchisq(i, 3)
qqplot(q, sort(returns_md2), main="Chi2 Q-Q Plot")
abline(lsfit(q, sort(returns_md2)))

################################################################################
# Assume a t-copula  with ncol(d)=3
t.cop <- tCopula(dim=ncol(d), dispstr='un')
m <- pobs(returns)         # pseudo-observations
fit <- fitCopula(t.cop, m, "ml")
(rho <- coef(fit)[1:ncol(d)])
(df <- coef(fit)[length(coef(fit))])
t.cop_fit <- tCopula(dim=ncol(d), rho, df=df, dispstr="un")

# Generate random samples u~U(0, 1) from the fitted t copula
u_sim_t <- rCopula(n_sim, t.cop_fit)
colnames(u_sim_t) <- colnames(d)
pairs(u_sim_t[1:1e3,], col="blue")        # only show the first 1000

# Obtain returns based on empirical marginals
return_sim_t <- empirical_quantile(u_sim_t, returns)
colnames(return_sim_t) <- colnames(d)
pairs(return_sim_t[1:1e3,], col="green")  # only show the first 1000

################################################################################
sim_t_md2 <- Mahalanobis2(return_sim_t)
QQ_Plot(sim_t_md2, returns_md2, col="orange")

################################################################################
n_days <- nrow(returns)
i <- ((1:n_days) - 0.5) / n_days
q_N <- quantile(sim_N_md2, probs=i, type=4, names=FALSE)
q_t <- quantile(sim_t_md2, probs=i, type=4, names=FALSE)
linear_N <- lsfit(q_N, sort(returns_md2))
linear_t <- lsfit(q_t, sort(returns_md2))

par(mfrow=c(1, 1))
# plot theoretical quantiles starting from 10
(idx_start <- min(which(q_N > 10), which(q_t > 10)))
plot(sort(linear_N$residuals^2)[idx_start:n_days], ylim=c(0, 25), 
     ylab="squared residuals", pch=1, cex=1.5)
points(sort(linear_t$residuals^2)[idx_start:n_days], 
       pch=4, cex=1.5, lwd=2)
legend("topleft", pch=c(1, 4), cex=1.5, lwd=c(1, 2),
       legend=c("Gaussian copula", "t-copula"), lty=0)