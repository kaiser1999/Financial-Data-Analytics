setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d <- read.csv("stock_1999_2002.csv")	  # read in data file
d <- as.ts(d)
u <- (lag(d) - d) / d
colnames(u) <- colnames(d)

plot(d, plot.type="multiple", col="blue")	# plot d
plot(u, plot.type="multiple", col="blue")	# plot u

par(mfrow=c(3,2), mar=c(4,4,4,4))
# if the dist is normal, the plot should close to this line
# histogram; qq-normal plot; add a line for reference
hist(u[,"HSBC"]); qqnorm(u[,"HSBC"]); qqline(u[,"HSBC"])
hist(u[,"CLP"]); qqnorm(u[,"CLP"]); qqline(u[,"CLP"])
hist(u[,"CK"]); qqnorm(u[,"CK"]); qqline(u[,"CK"])

############################################################

shapiro.test(u[,"HSBC"])
shapiro.test(u[,"CLP"])
shapiro.test(u[,"CK"])

############################################################

u1 <- u[,"HSBC"]; u2 <- u[,"CLP"]; u3 <- u[,"CK"]
ks.test(u1, pnorm, mean=mean(u1), sd=sd(u1))
ks.test(u2, pnorm, mean=mean(u2), sd=sd(u2))
ks.test(u3, pnorm, mean=mean(u3), sd=sd(u3))

############################################################

library("tseries")

JB.test <- function(u){
  z <- u - mean(u)            # Remove mean
  n <- length(z)              # Sample size
  s <- sd(z)*sqrt((n-1)/n)	# Population standard deviation
  sk <- sum(z^3)/(n*s^3)      # Skewness
  ku <- sum(z^4)/(n*s^4) - 3  # Excess Kurtosis
  JB <- n * (sk^2/6 + ku^2/24)	# JB test statistics
  p <- 1 - pchisq(JB, 2)      # chi-squared p-value
  list("JB_stat"=JB, "p_value"=p)
}

JB.test(u[,"HSBC"])
jarque.bera.test(u[,"HSBC"])

JB.test(u[,"CLP"])
jarque.bera.test(u[,"CLP"])

JB.test(u[,"CK"])
jarque.bera.test(u[,"CK"])

############################################################

library("car")

QQt.plot <- function(u, comp=""){
  z <- u - mean(u) # Remove mean
  sz <- sort(z)		   # sort z
  n <- length(z)		  # sample size
  s <- sd(z)*sqrt((n-1)/n)	# Population standard deviation
  ku <- sum(z^4)/(n*s^4) - 3	# Excess kurtosis
  nu <- 6/ku + 4 # Degrees of freedom
  i <- ((1:n)-0.5)/n  # create a vector of percentile
  q <- qt(i, nu)		  # percentile point from t(v)

  plot(q, sz, main=paste("Self-defined t Q-Q Plot of ", comp, " Return"))
  abline(lsfit(q, sz))
  qqPlot(z, distribution="t", df=nu, envelope=FALSE, line="none", cex=1.2,
         grid=FALSE, main=paste("t Q-Q Plot of ", comp, " Return"))
  abline(lsfit(q, sz))
  nu
}

par(mfrow=c(3,2), mar=c(4,4,4,4))
df_HSBC <- QQt.plot(u[,"HSBC"], comp="HSBC")
df_CLP <- QQt.plot(u[,"CLP"], comp="CLP")
df_CK <- QQt.plot(u[,"CK"], comp="CK")

############################################################

ks.test(u[,"HSBC"], pt, df_HSBC)
ks.test(u[,"CLP"], pt, df_CLP)
ks.test(u[,"CK"], pt, df_CK)

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

cor(u_180)

############################################################

pairs(u_180)

############################################################

par(mfrow=c(4,3), mar=c(4,4,4,4))

hist(d[,"HSBC"]); hist(d[,"CLP"]); hist(d[,"CK"])

qqnorm(d[,"HSBC"]); qqline(d[,"HSBC"])
qqnorm(d[,"CLP"]); qqline(d[,"CLP"])
qqnorm(d[,"CK"]); qqline(d[,"CK"])

plot(d[,"HSBC"], lag(d[,"HSBC"]))
plot(d[,"CLP"], lag(d[,"CLP"]))
plot(d[,"CK"], lag(d[,"CK"]))

plot(u[,"HSBC"], lag(u[,"HSBC"]))
plot(u[,"CLP"], lag(u[,"CLP"]))
plot(u[,"CK"], lag(u[,"CK"]))

############################################################

par(mfrow=c(3,3), mar=c(4,4,4,4))

acf(d[,"HSBC"]); acf(d[,"CLP"]); acf(d[,"CK"])

acf(u[,"HSBC"]); acf(u[,"CLP"]); acf(u[,"CK"])	

acf(u[,"HSBC"]^2); acf(u[,"CLP"]^2); acf(u[,"CK"]^2)	

############################################################

set.seed(4002)

mu_180 <- apply(u_180, 2, mean)
S_180 <- cov(u_180)
C_180 <- chol(S_180) # Cholesky decomposition of Sigma
# set s0 to the most recent price		 
s0 <- tail(d, 1)
s_pred <- c()
for (i in 1:90) {
  z <- rnorm(3)
  v <- mu_180 + t(C_180) %*% z
  s1 <- s0 * (1 + t(v))	# new stock price
  s_pred <- rbind(s_pred, s1)
  s0 <- s1	# update s0
}

s_pred <- ts(s_pred, start=nrow(d)+1)
data <- ts.union(d, s_pred)
par(mfrow=c(1,1))

col <- c("blue", "orange", "green", "pink", "brown", "red")
plot(data, plot.type="s", col=col)
legend("topright", col=col, lty=1,
       legend=c("HSBC", "CLP", "CK", "HSBC_pred", "CLP_pred", "CK_pred"))

############################################################

shapiro.test(u[,"HSBC"])

Shapiro_Test <- function(x, n_sim=100000){
  z <- matrix(rnorm(length(x)*n_sim), ncol=length(x))
  sz <- t(apply(z, 1, sort))
  m <- apply(sz, 2, mean)
  V <- cov(sz)
  inv_V <- solve(V)
  c <- (t(m) %*% inv_V %*% inv_V %*% m)^(1/2)
  a <- t(m) %*% inv_V / as.numeric(c)
  W <- sum(a * sort(x))^2/sum((x-mean(x))^2)
  return (W)
}

(Shapiro_Test(u[,"HSBC"]))