setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#############################################################

library(tseries)

d <- read.csv("stock_1999_2002.csv", row.names=1)
date <- as.Date(rownames(d), format="%d/%m/%Y")
d <- as.ts(d)
u <- (lag(d) - d)/d
colnames(u) <- paste0(colnames(d), "_Return")

#############################################################
library(zoo)

msd <- function(t, w){    # function to compute moving s.d.
  n <- length(t)-w+1
  out <- c()
  for (i in 1:n) {
    s <- sd(t[i:(i+w-1)]) # compute the sd of t(i) to t(i+w-1)
    out <- c(out, s)      # append the result to out
  }
  zoo(out)                # convert to time series
}

u1 <- u[,"HSBC_Return"]
u2 <- u[,"CLP_Return"]
u3 <- u[,"CK_Return"]
s_HSBC_90 <- msd(u1, 90)	    # compute 90-day moving sd
index(s_HSBC_90) <- date[(90+1):length(date)]
s_HSBC_180 <- msd(u1, 180)	  # compute 180-day moving sd
index(s_HSBC_180) <- date[(180+1):length(date)]

par(mfrow=c(2, 1), mar=c(4,4,1,1))
plot(s_HSBC_90, col="blue", xaxt="n")
plot_date <- index(s_HSBC_90)[seq(1, length(s_HSBC_90), 100)]
axis(1, at=plot_date, format(plot_date, "%d-%m-%Y"))

plot(s_HSBC_180, col="red", xaxt="n")
plot_date <- index(s_HSBC_180)[seq(1, length(s_HSBC_180), 100)]
axis(1, at=plot_date, format(plot_date, "%d-%m-%Y"))

############################################################

library(fGarch) # load library "fGarch"

# GARCH(1,1) on HSBC return
res_HSBC <- garchFit(~garch(1, 1), data=u1, include.mean=FALSE)
round(coef(res_HSBC), 6)  # display coefficient in 6 digits
res_HSBC@fit$llh          # compute log-likelihood value

res_HSBC@fit$matcoef

############################################################

GARCH_11 <- function(para, u){
  u <- as.numeric(u)
  omega0 <- para[1]
  alpha <- para[2]
  beta <- para[3]
  nu <- var(u)
  #nu <- omega0/(1-alpha-beta)
  loglik <- dnorm(u[1], 0, sqrt(nu), log=TRUE)
  
  for (i in 2:length(u)){
    nu <- omega0 + alpha*u[i-1]^2 + beta*nu
    loglik <- loglik + dnorm(u[i], 0, sqrt(nu), log=TRUE)
    
  }
  return(-loglik)
}

para <- c(0.3,0.1,0.4)
self_model <- constrOptim(para, GARCH_11, grad=NULL, u=u1,
                          method=c("Nelder-Mead"), 
                          ui=rbind(c(-1,-1,-1), diag(3), -diag(3)), 
                          ci=c(-1, rep(0, 3), rep(-1, 3)))

self_model$par

############################################################

omega <- coef(res_HSBC)[1]
alpha <- coef(res_HSBC)[2]
beta <- coef(res_HSBC)[3]
# initialize u_0^2 and nu_0 being mean of u_i^2
nu <- omega + alpha*mean(u1**2) + beta*mean(u1**2) # nu_1
for (i in 2:length(u1)){
  nu <- c(nu, omega + alpha*u1[i-1]^2 + beta*nu[i-1])
}
all.equal(as.vector(nu), res_HSBC@h.t)

resid_HSBC <- as.vector(u1/sqrt(nu))  # u1/sqrt(res_HSBC@h.t)
all.equal(resid_HSBC, residuals(res_HSBC, standardize=TRUE))

Box.test(u1^2, lag=15, type="Ljung")
Box.test(resid_HSBC^2, lag=15, type="Ljung")

par(mfrow=c(2,2), mar=c(4,4,3,3))
plot(res_HSBC, which=c(2, 5, 11, 13))

############################################################

par(mfrow=c(1,1))
vol_HSBC <- zoo(sqrt(res_HSBC@h.t))
index(vol_HSBC) <- date[-1]
plot(vol_HSBC, col="blue", lwd=2, ylab="", type="l", 
     ylim=c(0.01, 0.04), xaxt="n")
lines(s_HSBC_90, col="red", lwd=2)
lines(s_HSBC_180, col="green", lwd=2)
legend(x="topleft", legend=c("nu", "s_90", "s_180"), 
       col=c("blue", "red", "green"), lwd=2)
plot_date <- date[seq(2, length(date), 200)]
axis(1, at=plot_date, format(plot_date, "%d-%m-%Y"))

############################################################

u[dim(u)[1],]
cor(u[(dim(u)[1]-89):dim(u)[1],])
var(u[(dim(u)[1]-89):dim(u)[1],])

############################################################

res_HSBC <- garchFit(~garch(1, 1), data=u1, include.mean=FALSE)
res_CLP <- garchFit(~garch(1, 1), data=u2, include.mean=FALSE)
res_CK <- garchFit(~garch(1, 1), data=u3, include.mean=FALSE)
(coef <- rbind(coef(res_HSBC), coef(res_CLP), coef(res_CK)))

round(colMeans(coef), 6) # compute the column mean

############################################################
test <- garchFit(~garch(1, 1), data=u1, include.mean=FALSE, cond.dist="std")
plot(test, which=13)
coef(test)

############################################################
library(rugarch)

gjr_mean_model <- list(armaOrder=c(0,0), include.mean=FALSE)
gjr_var_model <- list(model="gjrGARCH", garchOrder=c(1,1))
gjr_spec <- ugarchspec(mean.model=gjr_mean_model, 
                       variance.model=gjr_var_model,
                       distribution.model="norm")
gjr_HSBC <- ugarchfit(data=u1, spec=gjr_spec)
gjr_CLP <- ugarchfit(data=u2, spec=gjr_spec)
gjr_CK <- ugarchfit(data=u3, spec=gjr_spec)

gjr_param <- rbind(coef(gjr_HSBC), coef(gjr_CLP), coef(gjr_CK))
colnames(gjr_param) <- c("omega", "alpha", "beta", "theta")
rownames(gjr_param) <- c("HSBC", "CLP", "CK")
gjr_param

###########################################################

omega <- coef(gjr_HSBC)[1]
alpha <- coef(gjr_HSBC)[2]
beta <- coef(gjr_HSBC)[3]
theta <- coef(gjr_HSBC)[4]
# initialize nu_0 being mean of u_i^2
nu <- mean(u1^2) # nu_1
for (i in 2:length(u1)){
  nu <- c(nu, omega + alpha*u1[i-1]^2 + beta*nu[i-1] + 
            theta*u1[i-1]^2*(u1[i-1] < 0))
}
all.equal(sqrt(as.vector(nu)), gjr_HSBC@fit$sigma)

resid_HSBC <- as.vector(u1/sqrt(nu))  # u1/gjr_HSBC@fit$sigma
gjr_resid_HSBC <- as.numeric(residuals(gjr_HSBC, standardize=TRUE))
all.equal(resid_HSBC, gjr_resid_HSBC)

Box.test(gjr_resid_HSBC^2, lag=15, type="Ljung")

par(mfrow=c(2,2), mar=c(4,4,3,3))
plot(gjr_HSBC@fit$sigma, type="l")
acf(gjr_resid_HSBC)
acf(gjr_resid_HSBC^2)
plot(gjr_HSBC, which=9)

###########################################################
e_mean_model <- list(armaOrder=c(0,0), include.mean=FALSE)
e_var_model <- list(model="eGARCH", garchOrder=c(1,1))
e_spec <- ugarchspec(mean.model=e_mean_model,
                     variance.model=e_var_model,
                     distribution.model="norm")
e_HSBC <- ugarchfit(data=u1, spec=e_spec)
e_CLP <- ugarchfit(data=u2, spec=e_spec)
e_CK <- ugarchfit(data=u3, spec=e_spec)

e_param <- rbind(coef(e_HSBC), coef(e_CLP), coef(e_CK))
colnames(e_param) <- c("omega", "alpha", "beta", "gamma")
rownames(e_param) <- c("HSBC", "CLP", "CK")
e_param

###########################################################

omega <- coef(e_HSBC)[1]
alpha <- coef(e_HSBC)[2]
beta <- coef(e_HSBC)[3]
gamma <- coef(e_HSBC)[4]
# initialize nu_0 being mean of u_i^2
nu <- mean(u1^2) # nu_1
for (i in 2:length(u1)){
  ln_nu <- (omega + gamma*(abs(u1[i-1])/sqrt(nu[i-1]) - sqrt(2/pi)) 
            + alpha*u1[i-1]/sqrt(nu[i-1]) + beta*log(nu[i-1]))
  nu <- c(nu, exp(ln_nu))
}
all.equal(sqrt(as.vector(nu)), e_HSBC@fit$sigma)

resid_HSBC <- as.vector(u1/sqrt(nu))  # u1/e_HSBC@fit$sigma
e_resid_HSBC <- as.numeric(residuals(e_HSBC, standardize=TRUE))
all.equal(resid_HSBC, e_resid_HSBC)

Box.test(e_resid_HSBC^2, lag=15, type="Ljung")

par(mfrow=c(2,2), mar=c(4,4,3,3))
plot(e_HSBC@fit$sigma, type="l")
acf(e_resid_HSBC)
acf(e_resid_HSBC^2)
plot(e_HSBC, which=9)

###########################################################

#residuals(gjr_HSBC, standardize=TRUE)
gjr_vol_HSBC <- zoo(gjr_HSBC@fit$sigma)
index(gjr_vol_HSBC) <- date[-1]
e_vol_HSBC <- zoo(e_HSBC@fit$sigma)
index(e_vol_HSBC) <- date[-1]

par(mfrow=c(2,1))
y_max <- max(vol_HSBC, s_HSBC_90, s_HSBC_180, 
             gjr_vol_HSBC, e_vol_HSBC, na.rm=TRUE)
y_min <- min(vol_HSBC, s_HSBC_90, s_HSBC_180, 
             gjr_vol_HSBC, e_vol_HSBC, na.rm=TRUE)
plot(vol_HSBC, col="blue", lwd=2, ylab="", type="l", 
     ylim=c(y_min, y_max), xaxt="n")
lines(s_HSBC_90, col="red", lwd=2)
lines(s_HSBC_180, col="green", lwd=2)
lines(gjr_vol_HSBC, col="black", lwd=2)
lines(e_vol_HSBC, col="gray", lwd=2)
legend(x="topleft", col=c("blue", "red", "green", "black", "gray"), lwd=2,
       legend=c("GARCH", "sma_90", "sma_180", "L-GARCH", "E-GARCH"))
plot_date <- date[seq(2, length(date), 200)]
axis(1, at=plot_date, format(plot_date, "%d-%m-%Y"))

d_HSBC <- zoo(d[,"HSBC"][-1])
index(d_HSBC) <- date[-1]
plot(d_HSBC, col="blue", lwd=2, type="l", xaxt="n")
axis(1, at=plot_date, format(plot_date, "%d-%m-%Y"))

###########################################################
library(rmgarch)

garch_mean_model <- list(armaOrder=c(0,0), include.mean=FALSE)
garch_var_model <- list(model="sGARCH", garchOrder=c(1,1))
garch_spec <- ugarchspec(mean.model=garch_mean_model,
                         variance.model=garch_var_model,
                         distribution.model="norm")

dcc_mult_spec <- multispec(replicate(garch_spec, n=3))
dcc_spec <- dccspec(uspec=dcc_mult_spec, dccOrder=c(1,1), 
                    distribution="mvnorm", model="DCC")
dcc_GARCH <- dccfit(spec=dcc_spec, data=u)
coef(dcc_GARCH)

###########################################################
library(rugarch)

garch_mean_model <- list(armaOrder=c(0,0), include.mean=FALSE)
garch_var_model <- list(model="sGARCH", garchOrder=c(1,1))
garch_spec <- ugarchspec(mean.model=garch_mean_model,
                         variance.model=garch_var_model,
                         distribution.model="norm")
garch_HSBC <- ugarchfit(data=u1, spec=garch_spec)
garch_CLP <- ugarchfit(data=u2, spec=garch_spec)
garch_CK <- ugarchfit(data=u3, spec=garch_spec)
resid_HSBC <- as.numeric(residuals(garch_HSBC, standardize=TRUE))
resid_CLP <- as.numeric(residuals(garch_CLP, standardize=TRUE))
resid_CK <- as.numeric(residuals(garch_CK, standardize=TRUE))
eta <- cbind(resid_HSBC, resid_CLP, resid_CK)

sigma <- cbind(garch_HSBC@fit$sigma, garch_CLP@fit$sigma, garch_CK@fit$sigma)
bar_Sigma <- cov(eta)

theta_1 <- coef(dcc_GARCH)[10]
theta_2 <- coef(dcc_GARCH)[11]

Q <- dcc_GARCH@mfit$Q[[1]]
check_Q <- as.numeric()
check_R <- as.numeric()
for (i in 2:length(u1)){
  Q <- ((1-theta_1-theta_2)*bar_Sigma + theta_2*Q 
        + theta_1*eta[i-1,]%*%t(eta[i-1,]))
  R <- diag(diag(Q)^(-1/2)) %*% Q %*% diag(diag(Q)^(-1/2))
  check_Q <- c(check_Q, all.equal(Q, dcc_GARCH@mfit$Q[[i]], 
                                  check.attributes=FALSE))
  check_R <- c(check_R, all.equal(R, dcc_GARCH@mfit$R[[i]], 
                                  check.attributes=FALSE))
}
all(check_Q==1)
all(check_R==1)

dcc_GARCH@mfit$Q[[1]]
bar_Sigma

############################################################
DCC_GARCH_11 <- function(para, Q, eta){
  theta_1 <- para[1]
  theta_2 <- para[2]

  R <- diag(diag(Q)^(-1/2)) %*% Q %*% diag(diag(Q)^(-1/2))
  
  loglik <- -1/2*log(det(R)) - 1/2*eta[1,] %*% solve(R) %*% eta[1,]
  for (i in 2:nrow(eta)){
    Q <- ((1-theta_1-theta_2)*bar_Sigma + theta_2*Q 
          + theta_1*eta[i-1,] %*% t(eta[i-1,]))
    R <- diag(diag(Q)^(-1/2)) %*% Q %*% diag(diag(Q)^(-1/2))
    
    loglik <- (loglik - 1/2*log(det(R)) 
               - 1/2*eta[i,] %*% solve(R) %*% eta[i,])
  }
  return (-loglik)
}

para <- c(0.3, 0.3)
Q <- dcc_GARCH@mfit$Q[[1]]
self_model <- constrOptim(para, DCC_GARCH_11, grad=NULL, Q=Q, eta=eta,
                          method=c("Nelder-Mead"), 
                          ui=rbind(c(-1,-1), diag(2)), 
                          ci=c(-1, rep(0, 2)))

self_model$par

############################################################

n <- 180
u_180 <- tail(u, n)
mu_180 <- apply(u_180, 2, mean)
S_180 <- cov(u_180)

z_180 <- sweep(u_180, 2, mu_180)
d2_180 <- diag(z_180 %*% solve(S_180) %*% t(z_180))
sd2_180 <- sort(d2_180)		# sort d2 in ascendingly
i <- ((1:n)-0.5)/n		# create percentile vector
q <- qchisq(i,3)		# compute quantiles 

par(mfrow=c(1,1))
qqplot(q, sd2_180, main="Chi2 Q-Q Plot")		# QQ-chisquare plot
abline(lsfit(q, sd2_180))	

############################################################

d2 <- as.numeric()
n <- nrow(u)
ns <- 180
for (i in 1:n){
  D <- diag(sigma[i,])
  R <- dcc_GARCH@mfit$R[[i]]
  H <- D %*% R %*% D
  if (i > n-ns){d2 <- c(d2, u[i,] %*% solve(H) %*% u[i,])}
}

sort_d2 <- sort(d2)
i <- ((1:ns)-0.5)/ns		    # create percentile vector
q <- qchisq(i, ncol(eta))		# compute quantiles 

par(mfrow=c(1,1))
qqplot(q, sort_d2, main="Chi2 Q-Q Plot under DCC")	# QQ-chisquare plot
abline(lsfit(q, sort_d2))	