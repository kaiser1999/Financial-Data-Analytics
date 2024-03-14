#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(stochvol)

df <- read.csv("../Datasets/stock_1999_2002.csv", row.names=1)
y_HSBC <- diff(log(df$HSBC))
y_CLP <- diff(log(df$CLP))
y_CK <- diff(log(df$CK))

set.seed(4002)
params <- list(draws=50000, burnin=10000, priornu=0, priorrho=NA,
               priormu=c(0, 100), priorphi=c(5, 1.5), priorsigma=1)
# y <- exp(h/2)*rt(1, df=nu)
# h <- mu + phi*(h - mu) + sigma*rt(1, df=nu)
sv_HSBC <- do.call(svsample, c(list(y=y_HSBC), params))
sv_CLP <- do.call(svsample, c(list(y=y_CLP), params))
sv_CK <- do.call(svsample, c(list(y=y_CK), params))

summary(sv_HSBC$para)$statistics[,"Mean"]
summary(sv_CLP$para)$statistics[,"Mean"]
summary(sv_CK$para)$statistics[,"Mean"]

################################################################################
get_h <- function(sv) summary(sv$latent)[1]$statistics[,"Mean"]
h_HSBC <- get_h(sv_HSBC); h_CLP <- get_h(sv_CLP);h_CK <- get_h(sv_CK)

resid_HSBC <- y_HSBC / exp(h_HSBC/2)
resid_CLP <- y_CLP / exp(h_CLP/2)
resid_CK <- y_CK / exp(h_CK/2)

Box.test(resid_HSBC, lag=15, type="Ljung-Box")
Box.test(resid_CLP, lag=15, type="Ljung-Box")
Box.test(resid_CK, lag=15, type="Ljung-Box")

shapiro.test(resid_HSBC)
shapiro.test(resid_CLP)
shapiro.test(resid_CK)

par(mfrow=c(1,3))
qqnorm(resid_HSBC, main="HSBC Normal Q-Q plot"); qqline(resid_HSBC)
qqnorm(resid_CLP, main="CLP Normal Q-Q plot"); qqline(resid_CLP)
qqnorm(resid_CK, main="CK Normal Q-Q plot"); qqline(resid_CK)

################################################################################
### Acceptance-rejection method and Gibbs Sampling ###
################################################################################
library(invgamma)
library(truncnorm)

SVol_h <- function(t, h_, y_t, mu, phi, nu, iter=1){
  if (t == 1){
    h_t <- h_[1]; h_2 <- h_[2]
    alpha_t <- mu + phi*(h_2 - mu)
    beta2_t <- nu
  }else if (t == T){
    h_1T <- h_[1]; h_t <- h_[2]
    alpha_t <- mu + phi*(h_1T - mu)
    beta2_t <- nu
  }else{
    h_1t <- h_[1]; h_t <- h_[2]; h_t1 <- h_[3]
    alpha_t <- mu + phi*(h_t1 - mu + h_1t - mu)/(1 + phi^2)
    beta2_t <- nu / (1 + phi^2)
  }
  
  tilde_alpha_t <- alpha_t + beta2_t/2 * (y_t^2*exp(-alpha_t) - 1)
  h_prop <- rnorm(1, tilde_alpha_t, sqrt(beta2_t))
  accept <- exp(-y_t^2/2 * (exp(-h_prop) - exp(-alpha_t)*(1 + alpha_t - h_prop)))
  
  if (!is.na(accept)){
    if (runif(1) < accept){
      return (h_prop)
    }
  }
  if (iter < 50){
    iter <- iter + 1
    return (SVol_h(t, h_, y_t, mu, phi, nu, iter=iter))
  }
  return (h_t)
}

# prior: mu ~ N(0, 100); phi ~ tN(0, 1); nu ~ IG(2.5, 0.025)
SVol_model <- function(y, num_it=1e5, priormu=c(0, 100), 
                       priorphi=c(0, 1), priorsigma2=c(2.5, 0.025), 
                       seed=4002){
  T <- length(y)
  alpha_mu <- priormu[1]; beta2_mu <- priormu[2]
  alpha_phi <- priorphi[1]; beta2_phi <- priorphi[2]
  alpha_nu <- priorsigma2[1]; beta_nu <- priorsigma2[2]
  
  set.seed(seed)
  mu <- rnorm(1, mean=alpha_mu, sd=sqrt(beta2_mu))
  phi <- rtruncnorm(1, a=-1, b=1, mean=alpha_phi, sd=sqrt(beta2_phi))
  nu <- rinvgamma(1, shape=alpha_nu, rate=beta_nu)
  
  h_hist <- matrix(0, nrow=num_it, ncol=T)
  mu_hist <- array(mu, dim=num_it)
  phi_hist <- array(phi, dim=num_it)
  nu_hist <- array(nu, dim=num_it)
  
  h_hist[1,] <- log(y^2 + 1e-10)
  prog_bar <- txtProgressBar(min=0, max=num_it, width=50, style=3)
  for (i in 2:num_it){
    # MH sampling for h
    h <- h_hist[i-1,]
    for (t in 1:T){
      if (t == 1) {
        h[t] <- SVol_h(t, h[c(t, t+1)], y[t], mu, phi, nu, 1)
      }else if (t == T) {
        h[t] <- SVol_h(t, h[c(t-1, t)], y[t], mu, phi, nu, 1)
      }else{
        h[t] <- SVol_h(t, h[c(t-1, t, t+1)], y[t], mu, phi, nu, 1)
      }
    }
    
    # Gibbs sampling for parameters
    hat_alpha_nu <- alpha_nu + T/2
    hat_beta_nu <- beta_nu + 1/2*(sum((h[-1] - mu - phi*(h[-T] - mu))^2) 
                                  + (h[1] - mu)^2*(1 - phi^2))
    nu <- rinvgamma(1, shape=hat_alpha_nu, rate=hat_beta_nu)
    
    hat_beta2_phi <- (sum((h[-c(1, T)] - mu)^2)/nu + 1/beta2_phi)^{-1}
    term_phi <- sum((h[-1] - mu)*(h[-T] - mu))
    hat_alpha_phi <- hat_beta2_phi*(term_phi/nu + alpha_phi/beta2_phi)
    phi <- rtruncnorm(1, a=-1, b=1, mean=hat_alpha_phi, 
                      sd=sqrt(hat_beta2_phi))
    
    hat_beta2_mu <- ((1 - phi^2 + (T-1)*(1 - phi)^2)/nu + 1/beta2_mu)^{-1}
    term_mu <- h[1]*(1 - phi^2) + (1 - phi)*sum(h[-1] - phi*h[-T])
    hat_alpha_mu <- hat_beta2_mu * (term_mu/nu + alpha_mu/beta2_mu)
    mu <- rnorm(1, mean=hat_alpha_mu, sd=sqrt(hat_beta2_mu))
    
    h_hist[i,] <- h
    mu_hist[i] <- mu; phi_hist[i] <- phi; nu_hist[i] <- nu
    setTxtProgressBar(prog_bar, i)
  }
  return(list(h_hist=h_hist, mu_hist=mu_hist, phi_hist=phi_hist,
              nu_hist=nu_hist, num_it=num_it))
}

get_h <- function(SV_lst, method="median", burn_in=0){
  h_hist <- SV_lst$h_hist
  func <- get(method)
  return(apply(h_hist[-c(1:burn_in),], MARGIN=2, FUN=func))
}

get_coefficient <- function(SV_lst, method="median", burn_in=0){
  coef_hist <- cbind(SV_lst$mu_hist, SV_lst$phi_hist, SV_lst$nu_hist)
  func <- get(method)
  return(apply(coef_hist[-c(1:burn_in),], MARGIN=2, FUN=func))
}

SV_HSBC <- SVol_model(y_HSBC, num_it=1e5)
h_HSBC <- get_h(SV_HSBC, method="mean", burn_in=1e4)
get_coefficient(SV_HSBC, method="mean", burn_in=1e4)

SV_CLP <- SVol_model(y_HSBC, num_it=1e5)
h_CLP <- get_h(SV_CLP, method="mean", burn_in=1e4)
get_coefficient(SV_CLP, method="mean", burn_in=1e4)

SV_CK <- SVol_model(y_HSBC, num_it=1e5)
h_CK <- get_h(SV_CK, method="mean", burn_in=1e4)
get_coefficient(SV_CK, method="mean", burn_in=1e4)

resid_HSBC <- y_HSBC / exp(h_HSBC/2)
resid_CLP <- y_CLP / exp(h_CLP/2)
resid_CK <- y_CK / exp(h_CK/2)
date <- as.Date(rownames(df), format="%d/%m/%Y")
for (year in 1999:2002){
  print(year)
  mask <- paste0(year, "-01-01") <= date & date < paste0(year+1, "-01-01")
  print(Box.test(resid_HSBC[mask], lag=15, type="Ljung-Box"))
  print(Box.test(resid_CLP[mask], lag=15, type="Ljung-Box"))
  print(Box.test(resid_CK[mask], lag=15, type="Ljung-Box"))
}
par(mfrow=c(1,3))
qqnorm(resid_HSBC, main="HSBC Normal Q-Q plot"); qqline(resid_HSBC)
qqnorm(resid_CLP, main="CLP Normal Q-Q plot"); qqline(resid_CLP)
qqnorm(resid_CK, main="CK Normal Q-Q plot"); qqline(resid_CK)

shapiro.test(resid_HSBC)
shapiro.test(resid_CLP)
shapiro.test(resid_CK)