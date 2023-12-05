# Initialize variables
S_0 <- 10; q <- 0; sigma <- 0.7; r <- 0; T <- 1; K <- 9; h <- 0.3
params <- list(mean=(r-q-sigma^2/2)*T, sd=sigma*sqrt(T))
discount <- exp(-r * T)

# Compute the exact Delta
d_plus <- (log(S_0/K) + (r-q+sigma^2/2)*T) / (sigma*sqrt(T))
(delta <- pnorm(d_plus))

################################################################################
set.seed(4002)
# Create a data frame to store results
results <- data.frame(h=numeric(0), n=numeric(0), delta=numeric(0),
                      est_forward_diff=numeric(0), 
                      est_central_diff=numeric(0))

n_size <- c(1e6, 1e8)
# Estimate Delta by forward and central difference with different n
for (n in n_size) {
  # Generate the Black-Scholes sample
  S_T <- S_0 * exp(do.call(rnorm, c(n, params)))
  Y_bar_S0 <- discount * mean(pmax(S_T - K, 0))
  
  S_T_minus_h <- (S_0 - h) * exp(do.call(rnorm, c(n, params)))
  Y_bar_S0_minus_h <- discount * mean(pmax(S_T_minus_h - K, 0))
  
  S_T_plus_h <- (S_0 + h) * exp(do.call(rnorm, c(n, params)))
  Y_bar_S0_plus_h <- discount * mean(pmax(S_T_plus_h - K, 0))
  
  # Estimate using forward difference method
  est_forward_diff <- (Y_bar_S0_plus_h - Y_bar_S0) / h
  
  # Estimate using central difference method
  est_central_diff <- (Y_bar_S0_plus_h - Y_bar_S0_minus_h) / (2*h)
  
  # Store the results in the data frame
  output <- data.frame(h=h, n=n, delta=delta, 
                       est_forward_diff=est_forward_diff, 
                       est_central_diff=est_central_diff)
  results <- rbind(results, output)
}

results

# Relative error for forward and central difference methods
abs(results$est_forward_diff - delta) / delta * 100
abs(results$est_central_diff - delta) / delta * 100

################################################################################
set.seed(4002)

n_size <- seq(400, 20000, by=400)
est_pathwise <- c()

# Estimate Delta by pathwise method with different n
for (i in 1:length(n_size)) {
  # Generate the Black-Scholes sample
  S_T <- S_0 * exp(do.call(rnorm, c(n_size[i], params)))

  # Estimate using pathwise method
  result <- (S_T > K) * S_T / S_0
  est_pathwise[i] <- discount * mean(result)
}

# Plot the graph of estimated Delta and exact Delta
plot(n_size, est_pathwise, type="o", col="red", ylim=c(0.4, 0.9), 
     xlab="number of paths", ylab="Delta", main="", lwd=1.5, pch=21)
abline(h=delta, col="blue", lwd=1.5)
legend("topright", legend=c("pathwise estimate", "exact"), 
       col=c("red","blue"), lwd=1.5, pch=c(21, NA))