setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("../Datasets/stock_1999_2002.csv", row.names=1) # read in data file
d <- as.ts(d)
returns <- (lag(d) - d)/d
colnames(returns) <- paste0(colnames(d), "_Return")
n_sim <- 1e5

# compute pseudo observations
library(copula)  # Package for copula computation
emp_u <- pobs(returns, ties.method="average")

par(mfrow=c(1,3))
# Q-Q plot for pseudo observations
col <- c("blue", "orange", "green")
n_days <- nrow(returns)
q <- ((1:n_days) - 0.5) / n_days
for (k in 1:ncol(returns)){
  qqplot(q, sort(emp_u[,k]), col=col[k], 
         xlab="Theoretical quantiles", ylab="Sample quantiles",
         main=paste0("Q-Q Plot of ", colnames(returns)[k]))
  abline(lsfit(q, sort(emp_u[,k])), lwd=2)
}
par(mfrow=c(1, 1))

pseudo_quantile <- function(p, samples){
  q <- matrix(NA, nrow=nrow(p), ncol=ncol(p))
  for (k in 1:ncol(p)){
    q[,k] <- quantile(samples[,k], probs=p[,k], type=4, names=FALSE)
  }
  return (q)
}

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

# Get back returns based on the random samples
return_sim_N <- pseudo_quantile(u_sim_N, returns)
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
         xlab="Bootstrapped quantiles", ylab="Sample quantiles", 
         main="Copula Q-Q Plot")
  abline(0, 1, lwd=2)
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
fit <- fitCopula(t.cop, emp_u, "ml")
(rho <- coef(fit)[1:ncol(d)])
(df <- coef(fit)[length(coef(fit))])
t.cop_fit <- tCopula(dim=ncol(d), rho, df=df, dispstr="un")

# Generate random samples u~U(0, 1) from the fitted t copula
u_sim_t <- rCopula(n_sim, t.cop_fit)
colnames(u_sim_t) <- colnames(d)
pairs(u_sim_t[1:1e3,], col="blue")        # only show the first 1000

# Get back returns based on the random samples
return_sim_t <- pseudo_quantile(u_sim_t, returns)
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

a <- 15
leg <- c("Gaussian copula", "t-copula")
# find the common index where both coordinates > a
sort_returns_md2 <- sort(returns_md2)
idx <- min(Reduce(intersect, list(which(q_N > a), which(q_t > a), 
                                  which(sort_returns_md2 > a))))
(idx_start <- min(idx))

plot(q_N[idx_start:n_days], sort_returns_md2[idx_start:n_days],
     xlab="Bootstrapped quantiles", ylab="Sample quantiles", 
     main="Copula Q-Q Plot",
     pch=1, cex=1.5, col="blue", xlim=c(a, max(q_N, q_t)))
points(q_t[idx_start:n_days], sort_returns_md2[idx_start:n_days],
       pch=4, cex=1.5, lwd=2, col="orange")
legend("topleft", pch=c(1, 4), cex=1.5, lwd=c(1, 2), 
       col=c("blue", "orange"), lty=0, legend=leg)
abline(0, 1, lwd=2)

################################################################################
# residuals with respect to the 45-degree line
sort_resid2_N <- sort((sort(returns_md2) - q_N)^2)
sort_resid2_t <- sort((sort(returns_md2) - q_t)^2)
# plot the largest 50 and skip the largest (last) entry
idx_plots <- (n_days-50):(n_days-1)
plot(sort_resid2_N[idx_plots], ylab="squared residuals", 
     ylim=c(0, max(sort_resid2_N[idx_plots], sort_resid2_t[idx_plots])),
     pch=1, cex=1.5, col="blue")
points(sort_resid2_t[idx_plots], pch=4, cex=1.5, lwd=2, col="orange")
legend("topleft", pch=c(1, 4), cex=1.5, lwd=c(1, 2),
       col=c("blue", "orange"), lty=0, legend=leg)

sort_resid2_N <- sort((sort(returns_md2) - q_N)^2)
sort_resid2_t <- sort((sort(returns_md2) - q_t)^2)
# plot the largest 50
idx_plots <- (n_days-50):(n_days)
plot(sort_resid2_N[idx_plots], ylab="squared residuals", 
     ylim=c(0, max(sort_resid2_N[idx_plots], sort_resid2_t[idx_plots])),
     pch=1, cex=1.5, col="blue")
points(sort_resid2_t[idx_plots], pch=4, cex=1.5, lwd=2, col="orange")
legend("topleft", pch=c(1, 4), cex=1.5, lwd=c(1, 2),
       col=c("blue", "orange"), lty=0, legend=leg)

################################################################################
plot(q_N[idx_start:n_days], sort_returns_md2[idx_start:n_days],
     xlab="Bootstrapped quantiles", ylab="Sample quantiles", 
     main="Copula Q-Q Plot",
     pch=1, cex=1.5, xlim=c(a, max(q_N, q_t)), col="blue")
points(q_t[idx_start:n_days], sort_returns_md2[idx_start:n_days],
       pch=4, cex=1.5, lwd=2, col="orange")
legend("topleft", pch=c(1, 4), cex=1.5, lwd=c(1, 2),
       col=c("blue", "orange"), lty=0, legend=leg)
abline(0, 1, lwd=2)
abline(lsfit(q_N, sort_returns_md2), lwd=2, col="blue")
abline(lsfit(q_t, sort_returns_md2), lwd=2, col="orange")