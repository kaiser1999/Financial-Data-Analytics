setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d <- read.csv("stock_1999_2002.csv", row.names=1) # read in data file
d <- as.ts(d)
return <- (lag(d) - d)/d
colnames(return) <- paste0(colnames(d), "_Return")
u1 <- return[,"HSBC_Return"]
u2 <- return[,"CLP_Return"]
u3 <- return[,"CK_Return"]
n_sim <- 1000

library(copula)  # Package for copula computation
# Assume a normal-copula with ncol(d)=3
# P2p: array of elements of upper triangular matrix
N.cop <- normalCopula(P2p(cor(d)), dim=ncol(d), dispstr="un")
set.seed(4002)
# Generate random samples u~U(0, 1) based on gaussian copula
u_sim <- rCopula(n_sim, N.cop)
pairs(u_sim, col="blue")
cor(u_sim)
cor(d)

# mvdc:  Multi-Variate Distribution Copula
N_cop_dist <- mvdc(copula=N.cop, margins=rep("norm", ncol(d)),
                   paramMargins=list(list(mean=mean(u1), sd=sd(u1)),
                                     list(mean=mean(u2), sd=sd(u2)),
                                     list(mean=mean(u3), sd=sd(u3))))

# Generate random return samples based on multi-variate normal copula
set.seed(4002)
return_sim_N <- rMvdc(n_sim, N_cop_dist)
pairs(rbind(return, return_sim_N), col=c("orange","blue"))

######################################################################
Empirical_QQ_Plot <- function(sim_data, returns, col=c()){
  n_stocks <- ncol(returns); n_days <- nrow(returns)
  if (length(col) == 0) col <- rep("black", n_stocks)
  
  par(mfrow=c(1, n_stocks))
  for (k in 1:n_stocks){
    sim_data_k <- sim_data[,k]; returns_k <- returns[,k]
    i <- ((1:n_days) - 0.5) / n_days
    q <- quantile(ecdf(sim_data_k), probs=i)
    
    qqplot(q, sort(returns_k), xlab="Empirical quantiles",
           ylab="Returns quantiles", col=col[k],
           main=paste(colnames(returns)[k], "Empirical Q-Q Plot"))
    abline(lsfit(q, sort(returns_k)))
  }
}

total_sim <- 1e4; col <- c("blue", "orange", "green")

######################################################################
set.seed(4002)
sim_N <- rMvdc(total_sim, N_cop_dist)
Empirical_QQ_Plot(sim_N, return, col=col)

######################################################################

# Assume a t-copula  with ncol(d)=3
t.cop <- tCopula(dim=ncol(d), dispstr='un')
m <- pobs(return)
fit <- fitCopula(t.cop, m, "ml")
(rho <- coef(fit)[1:ncol(d)])
(df <- coef(fit)[length(coef(fit))])

# Generate random samples u~U(0, 1) based on the fitted t copula
u_sim <- rCopula(n_sim, tCopula(dim=ncol(d), rho, df=df, dispstr="un"))
pairs(u_sim, col="blue")
cor(u_sim)
cor(d)

t.nu <- function(u){
  z <- u - mean(u)              # Remove mean
  n <- length(z)                # sample size
  s <- sd(z)*sqrt((n-1)/n)      # Population standard deviation
  ku <- sum(z^4)/(n*s^4) - 3    # Excess kurtosis
  6/ku + 4                      # Degrees of freedom
}

t_cop_dist <- mvdc(copula=tCopula(rho, dim=ncol(d), df=df, dispstr="un"),
                   margins=rep("norm", ncol(d)),
                   paramMargins=list(list(mean=mean(u1), sd=sd(u1)),
                                     list(mean=mean(u2), sd=sd(u2)),
                                     list(mean=mean(u3), sd=sd(u3))))

# Generate random return samples based on multi-variate t copula
set.seed(4002)
return_sim_t <- rMvdc(n_sim, t_cop_dist)

pairs(rbind(return, return_sim_t), col=c("orange","blue"))

######################################################################
set.seed(4002)
sim_t <- rMvdc(total_sim, t_cop_dist)
Empirical_QQ_Plot(sim_t, return, col=col)
