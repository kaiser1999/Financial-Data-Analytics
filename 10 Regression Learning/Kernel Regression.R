NW_estimator <- function(x0, x, y, h, kernel=dnorm) {
  f_x <- sapply(x, function(xi) kernel((x0 - xi)/h) / h)
  f_x %*% y / rowSums(f_x)
}

set.seed(4012)
n <- 500
eps <- rnorm(n, sd=2)
m <- function(x) x^2 * cos(x)
x <- rnorm(n, mean=2, sd=4)
y <- m(x) + eps
h <- 0.5    # Bandwidth

xGrid <- seq(-15, 15, l=100)

# Plot data
par(mfrow=c(1,1))
plot(x, y)
#rug(x, side=1); rug(y, side=2)
lines(xGrid, m(xGrid), col = 1)
lines(xGrid, NW_estimator(x0=xGrid, x=x, y=y, h=h), col=2)
legend("top", legend=c("True regression", "Nadaraya-Watson"),
       lwd=2, col=1:2)

################################################################################

