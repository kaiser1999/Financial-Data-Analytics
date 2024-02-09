#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################

#df <- read.csv("../Datasets/stock_1999_2002.csv", row.names=1)
#y <- diff(log(df$HSBC)) * 100
################################################################################
set.seed(4002)

mu <- -0.5; phi <- 0.332; nu <- 0.12
T <- 1000
y <- array(0, T)
h <- array(0, T)

h[1] <- rnorm(1, mean=mu, sd=sqrt(nu/(1 - phi^2)))
y[1] <- exp(h[1]/2) * rnorm(1, 0, 1)
for (i in 2:T) {
  h[i] <- mu + phi*(h[i-1] - mu) + sqrt(nu)*rnorm(1, 0, 1)
  y[i] <- exp(h[i]/2) * rnorm(1, 0, 1)
}

library(invgamma)
library(truncnorm)

seed=4002; num_it=1e5


MH_h <- function(t, h_, y_t, mu, phi, nu){
  if (t == 1){
    h_1 <- h_[1]; h_2 <- h_[2]
    alpha_t <- mu + phi*(h_2 - mu)
    beta2_t <- nu
  }else if (t == T){
    h_1T <- h_[1]; h_T <- h_[2]
    alpha_t <- mu + phi*(h_1T - mu)
    beta2_t <- nu
  }else{
    h_1t <- h_[1]; h_t <- h_[2]; h_t1 <- h_[3]
    alpha_t <- (mu + phi*(h_t1 - mu + h_1t - mu)) / (1 + phi^2)
    beta2_t <- nu / (1 + phi^2)
  }
  
  tilde_alpha_t <- alpha_t + beta2_t/2 * (y_t^2*exp(-alpha_t) - 1)
  h_prop <- rnorm(1, tilde_alpha_t, sqrt(beta2_t))
  accept <- exp(-y_t^2/2 * (exp(-h_prop) 
                            - exp(-alpha_t)*(1 + alpha_t - h_prop)))

  if (!is.na(accept)){
    if (runif(1) < min(1, accept)){
      return (h_prop)
    }
  }
  return (MH_h(t, h_, y_t, mu, phi, nu))
}

# prior: mu ~ N(0, 1); phi ~ N(0, 1); nu ~ IG(2.5, 0.025)
alpha_mu <- 0; beta2_mu <- 10; alpha_phi <- 0; beta2_phi <- 1
alpha_nu <- 2.5; beta_nu <- 0.025

T <- length(y)

set.seed(seed)
mu <- rnorm(1, mean=alpha_mu, sd=sqrt(beta2_mu))
phi <- rnorm(1, mean=alpha_phi, sd=sqrt(beta2_phi))
nu <- rinvgamma(1, shape=alpha_nu, rate=beta_nu)
print(paste(mu, phi, nu))

#mu <- 0.6 ; phi <- -0.4; nu <- 0.2

h_hist <- matrix(0, nrow=num_it, ncol=T)
mu_hist <- array(mu, dim=num_it)
phi_hist <- array(phi, dim=num_it)
nu_hist <- array(nu, dim=num_it)

h_hist[1,1] <- rnorm(1, mean=mu, sd=sqrt(nu/(1 - phi^2)))
for (t in 2:T) {
  h_hist[1,t] <- mu + phi*(h[i-1] - mu) + sqrt(nu)*rnorm(1, 0, 1)
}

for (i in 2:num_it){
  # MH sampling for h
  h <- h_hist[i-1,]
  for (t in 1:T){
    if (t == 1) {
      h[t] <- MH_h(t, h[c(t, t+1)], y[t], mu, phi, nu)
    }else if (t == T) {
      h[t] <- MH_h(t, h[c(t-1, t)], y[t], mu, phi, nu)
    }else{
      h[t] <- MH_h(t, h[c(t-1, t, t+1)], y[t], mu, phi, nu)
    }
  }
  
  # Gibbs sampling for parameters
  hat_beta2_mu <- ((1 - phi^2 + (T-1)*(1 - phi)^2)/nu + 1/beta2_mu)^{-1}
  term_mu <- h[1]*(1 - phi^2) + (1 - phi)*sum(h[-1] - phi*h[-T])
  hat_alpha_mu <- hat_beta2_mu * (term_mu/nu + alpha_mu/beta2_mu)
  mu <- rnorm(1, mean=hat_alpha_mu, sd=sqrt(hat_beta2_mu))
  alpha_mu <- hat_alpha_mu; beta2_mu <- hat_beta2_mu
  
  hat_beta2_phi <- ((sum((h[-T] - mu)^2) - (h[1] - mu)^2)/nu + 1/beta2_phi)^{-1}
  term_phi <- sum((h[-1] - mu)*(h[-T] - mu))
  hat_alpha_phi <- hat_beta2_phi * (term_phi/nu + alpha_phi/beta2_phi)
  phi <- rtruncnorm(1, a=-1, b=1, mean=hat_alpha_phi, sd=sqrt(hat_beta2_phi))
  alpha_phi <- hat_alpha_phi; beta2_phi <- hat_beta2_phi
  
  alpha_nu <- alpha_nu + T/2
  beta_nu <- beta_nu + 1/2*(sum((h[-1] - mu - phi*(h[-T] - mu))^2) + (h[1] - mu)^2*(1 - phi^2))
  nu <- rinvgamma(1, shape=alpha_nu, rate=beta_nu)
  print(paste(mu, phi, nu, i))
  
  h_hist[i,] <- h
  mu_hist[i] <- mu
  phi_hist[i] <- phi
  nu_hist[i] <- nu
}
