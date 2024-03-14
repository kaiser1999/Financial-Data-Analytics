################################################################################
### Finite Difference ###
################################################################################
# Initialize variables
S_0 <- 10; K <- 8; r <- 0.05; sigma <- 0.3; T <- 1

Sim_greek <- function(h_d, h_v, n, M, S_0, K, r, sigma, T, 
                      theta, seed=4002){
  delta_t <- T / M
  Milstein <- function(z, S_t, r, sigma){
    (S_t*(1 + r*delta_t) + sigma*S_t^(theta/2)*sqrt(delta_t)*z + 
       sigma^2*theta/2*S_t^(theta-1)*delta_t/2*(z^2-1))
  }
  
  Euro_call <- function(S_t, K, r, sigma, T){
    set.seed(seed)
    for (m in 1:M){
      z <- rnorm(n)
      S_t <- Milstein(z, S_t, r, sigma)
    }
    mean(exp(-r*T) * pmax(S_t - K, 0))
  }
  
  Y <- Euro_call(S_0, K, r, sigma, T)
  
  # Estimate delta using forward and central difference method
  Y_S0_neg <- Euro_call(S_0-h_d, K, r, sigma, T)
  Y_S0_pos <- Euro_call(S_0+h_d, K, r, sigma, T)
  
  # Estimate vega using forward and central difference method
  Y_sig_neg <- Euro_call(S_0, K, r, sigma-h_v, T)
  Y_sig_pos <- Euro_call(S_0, K, r, sigma+h_v, T)
  
  list(delta=list(forward=(Y_S0_pos - Y)/h_d, 
                  central=(Y_S0_pos - Y_S0_neg)/(2*h_d)),
       vega=list(forward=(Y_sig_pos - Y)/h_v, 
                 central=(Y_sig_pos - Y_sig_neg)/(2*h_v)))
}

################################################################################
# Compute the exact delta and vega for theta=2 (BS)
d_plus <- (log(S_0/K) + (r+sigma^2/2)*T) / (sigma*sqrt(T))
(exact_delta <- pnorm(d_plus))
(exact_vega <- S_0 * sqrt(T) * dnorm(d_plus))

h_delta <- seq(0.5, 0.05, by=-0.05)
h_vega <- seq(0.05, 0.005, by=-0.005)
n <- 1e6; M <- 1e4
theta_lst <- c(2, 1.8, 1)

for (theta in theta_lst){
  delta_finite <- data.frame(h=h_delta, theta=theta, 
                             delta_forward=0, delta_central=0)
  vega_finite <- data.frame(h=h_vega, theta=theta, 
                            vega_forward=0, vega_central=0)
  prog_bar <- txtProgressBar(min=0, max=length(h_delta), width=50, 
                             style=3)
  for (i in 1:length(h_delta)) {
    h_d <- h_delta[i]; h_v <- h_vega[i]
    Euro_CEV <- Sim_greek(h_d, h_v, n, M, S_0, K, r, sigma, T, 
                          theta=theta)
    delta_finite$delta_forward[i] <- Euro_CEV$delta$forward
    delta_finite$delta_central[i] <- Euro_CEV$delta$central
    vega_finite$vega_forward[i] <- Euro_CEV$vega$forward
    vega_finite$vega_central[i] <- Euro_CEV$vega$central
    setTxtProgressBar(prog_bar, i)
  }
  cat("\n")
  print(delta_finite)
  print(vega_finite)
}

################################################################################
### Pathwise differentiation + Likelihood ratio ###
################################################################################
# Initialize variables
S_0 <- 10; K <- 8; r <- 0.05; sigma <- 0.3; T <- 1

d_plus <- (log(S_0/K) + (r+sigma^2/2)*T) / (sigma*sqrt(T))
(exact_delta <- pnorm(d_plus))
(exact_vega <- S_0 * sqrt(T) * dnorm(d_plus))

################################################################################
### CEV ###
################################################################################
greek_pathwise <- function(n, M, S_0, K, r, sigma, T, theta, seed=4002){
  delta_t <- T / M
  y_t <- 1; v_t <- 0; S_t <- S_0
  set.seed(seed)
  for (m in 1:M){
    dw_t <- sqrt(delta_t) * rnorm(n)
    y_t <- (y_t + r*y_t*delta_t 
            + sigma*theta/2*S_t^(theta/2-1)*y_t*dw_t)
    v_t <- (v_t + r*v_t*delta_t 
            + S_t^(theta/2-1)*(S_t + sigma*theta/2*v_t)*dw_t)
    S_t <- S_t + r*S_t*delta_t + sigma*S_t^(theta/2)*dw_t
  }
  
  list(delta=mean(exp(-r*T)*(S_t > K) * y_t), 
       vega=mean(exp(-r*T)*(S_t > K) * v_t))
}

################################################################################
n_size <- seq(500, 100000, by=500)
M <- 1e4; theta <- 2
delta_pathwise <- c(); vega_pathwise <- c()
# Estimate delta with different n
for (i in 1:length(n_size)){
  Euro_CEV <- greek_pathwise(n_size[i], M, S_0, K, r, sigma, T, theta)
  delta_pathwise[i] <- Euro_CEV$delta
  vega_pathwise[i] <- Euro_CEV$vega
}

plot(n_size, delta_pathwise, type="o", col="red",
     xlab="number of paths", ylab="delta", main="", lwd=1.5, pch=21)
abline(h=exact_delta, col="blue", lwd=1.8)
legend("topright", col=c("red", "blue"), lwd=1.5,
       legend=c("pathwise differentiation", "exact"), pch=c(21, NA))

plot(n_size, vega_pathwise, type="o", col="red",
     xlab="number of paths", ylab="vega", main="", lwd=1.5, pch=21)
abline(h=exact_vega, col="blue", lwd=1.8)
legend("topright", col=c("red", "blue"), lwd=1.5,
       legend=c("pathwise differentiation", "exact"), pch=c(21, NA))

################################################################################
theta <- 1.8
delta_pathwise <- c(); vega_pathwise <- c()
# Estimate delta with different n
for (i in 1:length(n_size)){
  Euro_CEV <- greek_pathwise(n_size[i], M, S_0, K, r, sigma, T, theta)
  delta_pathwise[i] <- Euro_CEV$delta
  vega_pathwise[i] <- Euro_CEV$vega
}

plot(n_size, delta_pathwise, type="o", col="red",
     xlab="number of paths", ylab="delta", main="", lwd=1.5, pch=21)
plot(n_size, vega_pathwise, type="o", col="red",
     xlab="number of paths", ylab="vega", main="", lwd=1.5, pch=21)

################################################################################
theta <- 1
delta_pathwise <- c(); vega_pathwise <- c()
# Estimate delta with different n
for (i in 1:length(n_size)){
  Euro_CEV <- greek_pathwise(n_size[i], M, S_0, K, r, sigma, T, theta)
  delta_pathwise[i] <- Euro_CEV$delta
  vega_pathwise[i] <- Euro_CEV$vega
}

plot(n_size, delta_pathwise, type="o", col="red",
     xlab="number of paths", ylab="delta", main="", lwd=1.5, pch=21)
plot(n_size, vega_pathwise, type="o", col="red",
     xlab="number of paths", ylab="vega", main="", lwd=1.5, pch=21)

################################################################################
### Black-scholes ###
################################################################################
set.seed(4002)
n_size <- seq(500, 100000, by=500)
delta_pathwise <- c(); vega_pathwise <- c()
delta_likelihood <- c(); vega_likelihood <- c()
mu <- (r-sigma^2/2)*T

# Estimate delta and vega with different n
for (i in 1:length(n_size)) {
  # Generate the Black-Scholes sample
  w_T <- sqrt(T) * rnorm(n_size[i])
  S_T <- S_0 * exp(mu + sigma * w_T)
  
  d_payoff <- exp(-r * T) * (S_T > K)
  # Estimate delta and vega using pathwise differentiation method
  delta_pathwise[i] <- mean(d_payoff * S_T/S_0)
  vega_pathwise[i] <- mean(d_payoff * S_T*(w_T - sigma*T))
  
  payoff <- exp(-r * T) * pmax(S_T - K, 0)
  # Estimate delta and vega using likelihood ratio method
  delta_likelihood[i] <- mean(payoff * w_T/(S_0*sigma*T))
  vega_likelihood[i] <- mean(payoff * ((w_T^2/T - 1)/sigma - w_T))
}

# Plot the graph of estimated delta and exact delta
plot(n_size, delta_pathwise, type="o", col="red", ylim=c(0.8, 0.93),
     xlab="number of paths", ylab="delta", main="", lwd=1.5, pch=21)
lines(n_size, delta_likelihood, type="o", col="black", lwd=1.5, pch=21)
abline(h=exact_delta, col="blue", lwd=1.8)
legend("topright", col=c("red", "black", "blue"), lwd=1.5,
       legend=c("pathwise differentiation", "likelihood ratio", 
                "exact"), pch=c(21, 21, NA))

# Plot the graph of estimated vega and exact vega
plot(n_size, vega_pathwise, type="o", col="red", ylim=c(0.8, 4.1),
     xlab="number of paths", ylab="vega", main="", lwd=1.5, pch=21)
lines(n_size, vega_likelihood, type="o", col="black", lwd=1.5, pch=21)
abline(h=exact_vega, col="blue", lwd=1.8)
legend("topright", col=c("red", "black", "blue"), lwd=1.5,
       legend=c("pathwise differentiation", "likelihood ratio", 
                "exact"), pch=c(21, 21, NA))