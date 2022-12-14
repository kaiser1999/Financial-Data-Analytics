setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d <- read.csv("stock_1999_2002.csv")
d <- as.ts(d)
u <- (lag(d) - d)/d
colnames(u) <- paste0(colnames(d), "_Return")

#############################################################

msd <- function(t, w){	 # function to compute moving s.d.
  n <- length(t)-w+1
  out <- c()
  for (i in 1:n) {
    s <- sd(t[i:(i+w-1)])   # compute the sd of t(i) to t(i+w-1)
    out <- c(out, s)     # append the result to out
  }
  as.ts(out)	   # convert to time series
}

u1 <- u[,"HSBC_Return"]
u2 <- u[,"CLP_Return"]
u3 <- u[,"CK_Return"]
s_HSBC_90 <- msd(u1, 90)	    # compute 90-day moving sd
s_HSBC_180 <- msd(u1, 180)	  # compute 180-day moving sd
par(mfrow=c(2, 1), mar=c(4,4,1,1))
plot(s_HSBC_90, col="blue")
plot(s_HSBC_180, col="red")

############################################################

library(tseries) # load library "tseries"

res_HSBC <- garch(u1, order=c(1,1)) # GARCH(1,1) on HSBC return
names(res_HSBC)
round(res_HSBC$coef, 6) # display coefficient in 6 digits
-2*res_HSBC$n.likeli # compute log-likelihood value
plot(res_HSBC)

summary(res_HSBC)

###########################################################

omega <- res_HSBC$coef[1]
alpha <- res_HSBC$coef[2]
beta <- res_HSBC$coef[3]
nu <- omega/(1 - alpha - beta)   #nu_1
for (i in 2:length(u1)){
  nu <- c(nu, omega + alpha*u1[i-1]^2 + beta*nu[i-1])
}
resid_HSBC <- u1/sqrt(nu)
all(resid_HSBC[-1] == res_HSBC$residuals[-1])

Box.test(u1^2, lag=15, type="Ljung")
Box.test(res_HSBC$resid^2, lag=15, type="Ljung")

###########################################################

par(mfrow=c(1,1))
# add 90 NA and 180 NA before s_HSBC_90 and s_HSBC_180
t_HSBC_90 <- as.ts(c(rep(NA,90), s_HSBC_90))
t_HSBC_180 <- as.ts(c(rep(NA,180), s_HSBC_180))
s <- cbind(res_HSBC$fitted.values[,1], t_HSBC_90, t_HSBC_180)
matplot(s, type="l")

###########################################################

u[dim(u)[1],]
cor(u[(dim(u)[1]-89):dim(u)[1],])
var(u[(dim(u)[1]-89):dim(u)[1],])

###########################################################

res_HSBC <- garch(u1)	# GARCH(1,1) with default order (1,1)
res_CLP <- garch(u2)
res_CK <- garch(u3)
(coef <- rbind(res_HSBC$coef, res_CLP$coef, res_CK$coef))

round(apply(coef, 2, mean), 6) # compute the column mean

###########################################################
omega <- res_HSBC$coef[1]
alpha <- res_HSBC$coef[2]
beta <- res_HSBC$coef[3]
nu <- omega/(1 - alpha - beta)
log_like <- -1/2*(log(2*pi) + log(nu[1]) + u1[1]^2/nu[1])
log_like2 <- -1/2*(log(nu[1]) + u1[1]^2/nu[1])
for (i in 2:length(u1)){
  nu <- c(nu, omega + alpha*u1[i-1]^2 + beta*nu[i-1])
  log_like <- log_like -1/2*(log(2*pi) + log(nu[i]) + u1[i]^2/nu[i])
  log_like2 <- log_like2 -1/2*(log(nu[i]) + u1[i]^2/nu[i])
}
