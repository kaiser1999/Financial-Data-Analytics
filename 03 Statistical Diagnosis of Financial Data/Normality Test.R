setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("../Datasets/stock_1999_2002.csv", row.names=1)	  # read in data file
d <- as.ts(d)
u <- (lag(d) - d) / d
colnames(u) <- colnames(d)

library(zoo)
plot(zoo(d), plot.type="multiple", col=c("blue", "orange", "green"))
plot(zoo(u), plot.type="multiple", col=c("blue", "orange", "green"))

par(mfrow=c(3,2), mar=c(4,4,4,4))
# if the dist is normal, the plot should close to this line
# histogram; qq-normal plot; add a line for reference
hist(u[,"HSBC"]); qqnorm(u[,"HSBC"]); qqline(u[,"HSBC"])
hist(u[,"CLP"]); qqnorm(u[,"CLP"]); qqline(u[,"CLP"])
hist(u[,"CK"]); qqnorm(u[,"CK"]); qqline(u[,"CK"])

################################################################################
shapiro.test(u[,"HSBC"])
shapiro.test(u[,"CLP"])
shapiro.test(u[,"CK"])

################################################################################
u1 <- u[,"HSBC"]; u2 <- u[,"CLP"]; u3 <- u[,"CK"]
ks.test(u1, pnorm, mean=mean(u1), sd=sd(u1))
ks.test(u2, pnorm, mean=mean(u2), sd=sd(u2))
ks.test(u3, pnorm, mean=mean(u3), sd=sd(u3))

################################################################################
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

################################################################################
library("car")

t.QQ.plot <- function(u, comp=""){
  z <- u - mean(u) # Remove mean
  sz <- sort(z)		   # sort z
  n <- length(z)		  # sample size
  s <- sd(z)*sqrt((n-1)/n)	# Population standard deviation
  ku <- sum(z^4)/(n*s^4) - 3	# Excess kurtosis
  nu <- 6/ku + 4 # Degrees of freedom
  i <- ((1:n)-0.5)/n  # create a vector of percentiles
  q <- qt(i, nu)		  # percentile points from t(v)

  plot(q, sz, main=paste("Self-defined t Q-Q Plot of ", comp, " Return"))
  qqline(sz, distribution=function(p) qt(p, df=nu), probs=c(0.25, 0.75))
  qqPlot(z, distribution="t", df=nu, envelope=FALSE, line="quartiles",
         col.lines="black", lwd=1, cex=1, grid=FALSE, id=FALSE,
         main=paste("t Q-Q Plot of ", comp, " Return"))
  nu
}

par(mfrow=c(3,2), mar=c(4,4,4,4))
df_HSBC <- t.QQ.plot(u[,"HSBC"], comp="HSBC")
df_CLP <- t.QQ.plot(u[,"CLP"], comp="CLP")
df_CK <- t.QQ.plot(u[,"CK"], comp="CK")

################################################################################
t_HSBC <- u[,"HSBC"]/sd(u[,"HSBC"])*sqrt(df_HSBC/(df_HSBC-2))
ks.test(t_HSBC, pt, df_HSBC)
t_CLP <- u[,"CLP"]/sd(u[,"CLP"])*sqrt(df_CLP/(df_CLP-2))
ks.test(t_CLP, pt, df_CLP)
t_CK <- u[,"CK"]/sd(u[,"CK"])*sqrt(df_CK/(df_CK-2))
ks.test(t_CK, pt, df_CK)

################################################################################
n <- 180
u_180 <- tail(u, n)
mu_180 <- apply(u_180, 2, mean)
S_180 <- cov(u_180)

z_180 <- sweep(u_180, 2, mu_180)
md2_180 <- rowSums((z_180 %*% solve(S_180)) * z_180)
smd2_180 <- sort(md2_180)		# sort md2 in ascendingly
i <- ((1:n)-0.5)/n		# create percentile vector
q <- qchisq(i,3)		# compute quantiles 

par(mfrow=c(1,1))
qqplot(q, smd2_180, main="Chi2 Q-Q Plot")		# QQ-chisquare plot
qqline(smd2_180, distribution=function(p) qchisq(p, df=3))

ks.test(smd2_180, pchisq, 3)

################################################################################
cor(u_180)

################################################################################
pairs(u_180)

################################################################################
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

################################################################################
par(mfrow=c(3,3), mar=c(4,4,4,4))

acf(d[,"HSBC"]); acf(d[,"CLP"]); acf(d[,"CK"])
acf(u[,"HSBC"]); acf(u[,"CLP"]); acf(u[,"CK"])
acf(u[,"HSBC"]^2); acf(u[,"CLP"]^2); acf(u[,"CK"]^2)

################################################################################
set.seed(4002)

mu_180 <- apply(u_180, 2, mean)
S_180 <- cov(u_180)
C_180 <- chol(S_180)  # Cholesky decomposition of Sigma
s0 <- tail(d, 1)      # set s0 to the most recent price
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
       legend=c(colnames(d), paste0(colnames(d), "_pred")))