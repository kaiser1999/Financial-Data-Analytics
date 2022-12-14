setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d <- read.csv("stock_1999_2002.csv", row.names=1) # read in data file
d <- as.ts(d)

x_n <- as.vector(d[nrow(d),]) # select the last obs
w <- c(40000, 30000, 30000)   # investment amount on each stock
p_0 <- sum(w)	                # total investment amount
w_s <- w/x_n                  # no. of shares bought at day n

h_sim <- t(t(lag(d)/d) * x_n)
p_n <- h_sim %*% w_s          # portfolio value at day n

loss <- p_0 - p_n	            # loss
(VaR_sim <- quantile(loss, 0.99)) # 1-day 99% V@R

######################################################################

library(tseries)

d <- read.csv("stock_1999_2002.csv", row.names=1) # read in data file
t <- as.ts(d$HSBC)          # select HSBC
n <- nrow(d)                # no. of obs
x_n <- t[n]                 # select the last obs
ns <- n-1                   # number of scenarios
u <- (lag(t)-t)/t           # stock returns

vol <- garch(u)             # fit the GARCH(1,1) model
p_0 <- 100000               # initial portfolio value
w_s <- p_0/x_n              # shares owned on day n

# Fitted variance on day n 
var.n <- vol$coef[1] + vol$coef[2]*u[length(u)]^2 + 
  vol$coef[3]*vol$fitted.values[,1][length(u)]^2

t_i <- t[2:ns]
t_i_1 <- t[1:(ns-1)]
var.i <- vol$fitted.values[,1][2:ns]^2
h_sim <- x_n*(t_i_1+(t_i-t_i_1)*sqrt(var.n/var.i))/t_i_1

p_n <- h_sim * w_s          # portfolio value
loss_GARCH <- p_0 - p_n     # loss
(VaR_GARCH <- quantile(loss_GARCH, 0.99))   # 1-day 99% VaR

######################################################################

library(fGarch) # load library "fGarch"

d <- read.csv("stock_1999_2002.csv", row.names=1) # read in data file
t <- as.ts(d$HSBC)          # select HSBC
n <- nrow(d)                # no. of obs
x_n <- t[n]                 # select the last obs
ns <- n-1                   # number of scenarios
u <- (lag(t)-t)/t           # stock returns

var <- garchFit(~garch(1, 1), data=u, include.mean=FALSE)
p_0 <- 100000               # initial portfolio value
w_s <- p_0/x_n              # shares owned on day n

# Fitted variance on day n 
var.n <- coef(var)[1] + coef(var)[2]*u[length(u)]^2 + 
  coef(var)[3]*var@h.t[length(u)]

t_i <- t[2:ns]
t_i_1 <- t[1:(ns-1)]
var.i <- var@h.t[2:ns]
h_sim <- x_n*(t_i_1+(t_i-t_i_1)*sqrt(var.n/var.i))/t_i_1

p_n <- h_sim * w_s          # portfolio value
loss_GARCH <- p_0 - p_n     # loss
(VaR_GARCH <- quantile(loss_GARCH, 0.99))   # 1-day 99% VaR

######################################################################

d <- read.csv("stock_1999_2002.csv", row.names=1) # read in data file
t <- as.ts(d)
u <- (lag(t)-t)/t
S <- var(u)                   # sample cov. matrix
w <- c(40000, 30000, 30000)   # investment amount on each stock
delta_p <- u %*% w            # Delta P
sd_p <- sd(delta_p)           # sample sd of portfolio (empirical)
(VaR_N <- qnorm(0.99)*sd_p)   # 1-day 99% V@R with normal
qnorm(0.99)*sqrt(t(w) %*% S %*% w) # z x sqrt(w.T S w)

######################################################################

ku <- sum((delta_p/sd_p)^4)/length(delta_p)-3
nu <- round(6/ku+4)
(VaR_t <- qt(0.99, nu)*sd_p*sqrt((nu-2)/nu)) # 1-day 99% V@R with t

######################################################################

u <- 3.2                        # threshold value
m <- mean(loss)                 # mean loss
s <- sd(loss)                   # sd loss
z <- (loss-m)/s                 # standardize loss
z_u <- z[z>u]                   # select z>u
(n_u <- length(z_u))            # no. of z>u

n_log_lik <- function(p, y){    # p=(xi, beta)
  length(y)*log(p[2])+(1/p[1]+1)*sum(log(1+p[1]*y/p[2]))
}

p0 <- c(0.2, 0.01)              # initial p0
# min -ve log_likelihood
res <- optim(p0, n_log_lik, y=(z_u-u))

(p <- res$par)                  # MLE p=(xi, beta)
-res$value                      # max value
q <- 0.99
(VaR <- u+(p[2]/p[1])*(((1-q)*length(z)/n_u)^(-p[1])-1))
(VaR_EVT <- m+VaR*s)            # 1day 99% V@R by EVT

######################################################################

m <- 0:10
round(1-pbinom(m, 250, 0.01), 4)

######################################################################

x_n <- as.vector(d[nrow(d),])     # select the last obs
w <- c(40000, 30000, 30000)       # investment amount on each stock
p_0 <- sum(w)                     # total investment amount
w_s <- w/x_n                      # no. of shares bought at day n

ns <- 250                         # 250 days
x_250 <- as.matrix(tail(d, ns))   # recent 250 days
ps_250 <- x_250 %*% t(w_s)        # portfolio value
ps_250 <- c(ps_250, p_0)          # add total amount 
loss_250 <- ps_250[1:ns]-ps_250[2:(ns+1)] # compute daily loss

sum(loss_250 > VaR_sim)           # no. of exceptions
sum(loss_250 > VaR_GARCH)
sum(loss_250 > VaR_N)
sum(loss_250 > VaR_t)
sum(loss_250 > VaR_EVT)

par(mfrow=c(1,1))
hist(-loss_250, main="Profit/Loss", breaks=50, xlim=c(-4500, 3200))
abline(v=-VaR_sim, col="blue")
abline(v=-VaR_GARCH, col="red")
abline(v=-VaR_N, col="green")
abline(v=-VaR_t, col="gray")
abline(v=-VaR_EVT, col="orange")

text(-VaR_sim, -0.46, "-VaR_sim", col="blue")
text(-VaR_GARCH, -0.46, "-VaR_GARCH", col="red")
text(-VaR_N, 15, "-VaR_N", col="green")
text(-VaR_t, 8, "-VaR_t", col="gray")
text(-VaR_EVT, 15, "-VaR_EVT", col="orange")

######################################################################
k <- c(3.4, 3.5, 3.65, 3.75, 3.85, 4)
m <- 5:10
model <- lm(k~m)
theta <- round(coef(model), 5)

par(mar=c(4,4,1,1))
plot(m, k, xlim=c(0, 10), ylim=c(2.6, 4), col="blue", 
     xlab="Number of days of exceptions " ~italic("m"), 
     ylab="Regulatary multiplier " ~italic(" k"), pch=16, cex=1.4)
abline(lm(k~m), col="blue", lty=3, lwd=2)
points(0:4, rep(3, 5), col="orange", pch=16, cex=1.4)
abline(3, 0, col="orange", lty=3, lwd=2)

eq <- substitute(italic(k) == a + b ~italic(m)*","~~italic(R)^2~"="~r2, 
                 list(a = theta[1], 
                      b = theta[2], 
                      r2 = format(summary(model)$r.squared, digits=5)))

mtext(eq, 3, line=-3)

######################################################################
# expected shortfall
mean(loss[loss > VaR_sim])
mean(loss[loss > VaR_GARCH])
mean(loss_GARCH[loss_GARCH > VaR_GARCH])

mu <- mean(loss)
sig <- sd(loss)
# normal
mu + sig/0.01*dnorm(qnorm(0.01))

# student-t
s <- sig*sqrt((nu-2)/nu)
mu + s*(nu + qt(0.01, nu)^2)/(nu-1)*(dt(qt(0.01, nu), nu)/0.01)

# extreme value theorem
EVT <- VaR + (p[2] + p[1]*(VaR - u))/(1 - p[1])
mu + sig*EVT

# distribution free
K <- floor(n*0.01)
sort_loss <- sort(loss, decreasing=TRUE)
1/(0.01*n)*sum(sort_loss[1:(K-1)]) + (1-(K-1)/(0.01*n))*sort_loss[K]

######################################################################
EVT_gradient <- function(para, y){
  xi <- para[1]
  beta <- para[2]
  n_u <- length(y)
  partial_xi <- 1/xi^2*sum(log(1 + xi*y/beta)) - (1/xi+1)*sum(y/(beta+xi*y))
  partial_beta <- -n_u/beta + (1/xi+1)*sum(xi*y/(beta^2 + beta*xi*y))
  return (c(partial_xi, partial_beta))
}

n_log_lik <- function(p, y){    # p=(xi, beta)
  length(y)*log(p[2])+(1/p[1]+1)*sum(log(1+p[1]*y/p[2]))
}

# min -ve log_likelihood
res <- optim(p, fn=n_log_lik, gr=EVT_gradient, y=(z_u-u), method="BFGS")

(p2 <- res$par)                  # MLE p=(xi, beta)