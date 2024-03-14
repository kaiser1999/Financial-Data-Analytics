MM_quantile <- function(x, q, tol=1e-10, maxit=1e4){
  n <- length(x)
  mu_old <- mean(x)
  for (i in 1:maxit){
    w <- 1/abs(x - mu_old)
    mu_new <- (sum(w * x) + (2*q - 1)*n)/sum(w)
    if (is.na(mu_new)) return (mu_old)
    if (abs(mu_new - mu_old) < tol) return (mu_new)
    mu_old <- mu_new
  }
  return (mu_new)
}

################################################################################
# same as Risk_Measure.R
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d <- read.csv("../Datasets/stock_1999_2002.csv", row.names=1) # read in data file
d <- as.ts(d)

x_n <- as.vector(d[nrow(d),]) # select the last obs
w <- c(40000, 30000, 30000)   # investment amount on each stock
p_0 <- sum(w)	                # total investment amount
w_s <- w/x_n                  # no. of shares bought at day n

h_sim <- t(t(lag(d)/d) * x_n)
p_n <- h_sim %*% w_s          # portfolio value at day n

loss <- p_0 - p_n	            # loss

################################################################################
(VaR_sim <- MM_quantile(loss, 0.99)) # 1-day 99% V@R