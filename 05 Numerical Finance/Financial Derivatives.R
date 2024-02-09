# Initialize variables
S_0 <- 10; K <- 8; sigma <- 0.3; r <- 0.03; T <- 1
h_delta <- seq(0.1, 0.5, by=0.1); h_vega <- seq(0.01, 0.05, by=0.01)
n <- 1e5; M <- 1e4
# Do not take large h on vega, since sigma < 1 !

# Compute the exact Delta and Vega
d_plus <- (log(S_0/K) + (r+sigma^2/2)*T) / (sigma*sqrt(T))
(exact_delta <- pnorm(d_plus))
(exact_vega <- S_0 * sqrt(T) * dnorm(d_plus))

################################################################################
### Finite Difference ###
################################################################################
Sim_greek <- function(h_d, h_v, n, S_0, K, r, sigma, T, theta, M=1){
  Euro_call_Ext <- function(z, S_0, K, r, sigma, theta=2){
    S_t <- S_0 * exp((r-sigma^2/2)*T + sigma*sqrt(T)*z)
    mean(exp(-r*T) * pmax(S_t - K, 0))
  }
  
  delta_t <- T / M
  Milstein <- function(z, S_t, theta, r, sigma){
    (S_t*(1 + r*delta_t) + sigma*S_t^(theta/2)*sqrt(delta_t)*z + 
       sigma^2*theta/2*S_t^(theta-1)*delta_t/2*(z^2-1))
  }
  
  Euro_call_Mil <- function(z, S_0, K, r, sigma, theta){
    S_t <- S_0
    for (m in 1:M){
      S_t <- Milstein(z[m,], S_t, theta, r, sigma)
    }
    mean(exp(-r*T) * pmax(S_t - K, 0))
  }
  
  if (theta == 2){
    z <- rnorm(n)
    call_type <- Euro_call_Ext
  } else{
    z <- matrix(rnorm(n * M), nrow=M)
    call_type <- Euro_call_Mil
  }
  
  Y <- call_type(z, S_0, K, r, sigma, theta)
  
  # Estimate Delta using forward and central difference method
  Y_S0_minus <- call_type(z, S_0-h_d, K, r, sigma, theta)
  Y_S0_plus <- call_type(z, S_0+h_d, K, r, sigma, theta)
  
  est_delta_forward <- (Y_S0_plus - Y) / h_d
  est_delta_central <- (Y_S0_plus - Y_S0_minus) / (2*h_d)
  
  # Estimate Vega using forward and central difference method
  Y_sigma_minus <- call_type(z, S_0, K, r, sigma-h_v, theta)
  Y_sigma_plus <- call_type(z, S_0, K, r, sigma+h_v, theta)
 
  est_vega_forward <- (Y_sigma_plus - Y)/h_v
  est_vega_central <- (Y_sigma_plus - Y_sigma_minus)/(2*h_v)
  
  list(est_delta=list(forward=est_delta_forward,
                      central=est_delta_central),
       est_vega=list(forward=est_vega_forward,
                     central=est_vega_central))
}

################################################################################
# Estimate Delta and Vega using forward and central difference 
# with different h
theta_lst <- c(2, 1, 1.8)
for (theta in theta_lst){
  results_delta <- data.frame(h=h_delta, n=n, theta=theta, 
                              est_delta_forward=0, est_delta_central=0)
  
  results_vega <- data.frame(h=h_vega, n=n, theta=theta, 
                             est_vega_forward=0, est_vega_central=0)
  
  set.seed(4002)
  for (i in 1:length(h_delta)) {
    h_d <- h_delta[i]; h_v <- h_vega[i]
    
    Euro_BS <- Sim_greek(h_d, h_v, n, S_0, K, r, sigma, T, 
                         theta=theta, M=M)
    results_delta$est_delta_forward[i] <- Euro_BS$est_delta$forward
    results_delta$est_delta_central[i] <- Euro_BS$est_delta$central
    results_vega$est_vega_forward[i] <- Euro_BS$est_vega$forward
    results_vega$est_vega_central[i] <- Euro_BS$est_vega$central
  }
  
  print(results_delta)
  print(results_vega)
}

################################################################################
### Pathwise + Likelihood ###
################################################################################
set.seed(4002)

n_size <- seq(400, 30000, by=400)
est_delta_pathwise <- c(); est_vega_pathwise <- c()
est_delta_lr <- c(); est_vega_lr <- c()
params <- list(mean=(r-q-sigma^2/2)*T, sd=sigma*sqrt(T))
discount <- exp(-r * T)

# Estimate Delta using pathwise method with different n
for (i in 1:length(n_size)) {
  # Generate the Black-Scholes sample
  S_T <- S_0 * exp(do.call(rnorm, c(n_size[i], params)))
  
  # Estimate Delta using pathwise method
  result <- (S_T > K) * S_T / S_0
  est_delta_pathwise[i] <- discount * mean(result)
  
  # Estimate Vega using pathwise method
  result <- (S_T > K)*S_T*(log(S_T/S_0) - (r - q + sigma^2/2)*T)
  est_vega_pathwise[i] <- discount / sigma * mean(result)
  
  # Estimate Delta using likelihood ratio method
  h <- (log(S_T/S_0) - ((r - sigma^2/2)*T)) / (sigma*sqrt(T))
  result <- pmax(S_T - K, 0) * h / (S_0 * sigma * sqrt(T))
  est_delta_lr[i] <- mean(discount * result)
  
  # Estimate Vega using likelihood ratio method
  result <- pmax(S_T - K, 0) * (-1/sigma + h^2/sigma -sqrt(T)*h)
  est_vega_lr[i] <- mean(discount * result)
}

# Plot the graph of estimated Delta and exact Delta
plot(n_size, est_delta_pathwise, type="o", col="red", ylim=c(0.5, 1),
     xlab="number of paths", ylab="Delta", main="", lwd=1.5, pch=21)
lines(n_size, est_delta_lr, type="o", col="black", lwd=1.5, pch=21)
abline(h=exact_delta, col="blue", lwd=1.8)
legend("topright", col=c("red", "black", "blue"), lwd=1.5,
       legend=c("pathwise estimate", "likelihood ratio estimate", 
                "exact"), pch=c(21, 21, NA))

# Plot the graph of estimated Vega and exact Vega
plot(n_size, est_vega_pathwise, type="o", col="red", ylim=c(2, 4.5),
     xlab="number of paths", ylab="Vega", main="", lwd=1.5, pch=21)
lines(n_size, est_vega_lr, type="o", col="black", lwd=1.5, pch=21)
abline(h=exact_vega, col="blue", lwd=1.8)
legend("topright", col=c("red", "black", "blue"), lwd=1.5,
       legend=c("pathwise estimate", "likelihood ratio estimate", 
                "exact"), pch=c(21, 21, NA))