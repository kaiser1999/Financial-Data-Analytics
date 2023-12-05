# Initialize variables
S_0 <- 10; q <- 0; sigma <- 0.7; r <- 0; T <- 1; K <- 8; h <- 0.3
params <- list(mean=(r-q-sigma^2/2)*T, sd=sigma*sqrt(T))
params_minus_h<-list(mean=(r-q-(sigma-h)^2/2)*T,sd=(sigma-h)*sqrt(T))
params_plus_h <-list(mean=(r-q-(sigma+h)^2/2)*T,sd=(sigma+h)*sqrt(T))
discount <- exp(-r * T)

# Compute the exact Delta and Vega
d_plus <- (log(S_0/K) + (r-q+sigma^2/2)*T) / (sigma*sqrt(T))
(exact_delta <- pnorm(d_plus))
(exact_vega <- S_0 * sqrt(T) * dnorm(d_plus))

################################################################################
set.seed(4002)
# Create a data frame to store results
results <- data.frame(n=numeric(0), exact_delta=numeric(0), 
                      exact_vega=numeric(0), 
                      est_delta_forward=numeric(0), 
                      est_vega_forward=numeric(0), 
                      est_delta_central=numeric(0), 
                      est_vega_central=numeric(0))

n_size <- c(1e6, 1e8)
# Estimate Delta and Vega using forward and central difference 
# with different n
for (n in n_size) {
  # Generate the Black-Scholes sample
  S_T <- S_0 * exp(do.call(rnorm, c(n, params)))
  
  # Estimate Delta using forward and central difference method
  Y_bar_S0 <- discount * mean(pmax(S_T - K, 0))
  S_T_minus_h <- (S_0 - h) * exp(do.call(rnorm, c(n, params)))
  Y_bar_S0_minus_h <- discount * mean(pmax(S_T_minus_h - K, 0))
  S_T_plus_h <- (S_0 + h) * exp(do.call(rnorm, c(n, params)))
  Y_bar_S0_plus_h <- discount * mean(pmax(S_T_plus_h - K, 0))
  
  est_delta_forward <- (Y_bar_S0_plus_h - Y_bar_S0) / h
  est_delta_central<- (Y_bar_S0_plus_h - Y_bar_S0_minus_h) / (2*h)
  
  # Estimate Vega using forward and central difference method
  Y_bar_sigma <- Y_bar_S0
  S_T_minus_h <- S_0  * exp(do.call(rnorm, c(n, params_minus_h)))
  Y_bar_sigma_minus_h <- discount * mean(pmax(S_T_minus_h - K, 0))
  S_T_plus_h <- S_0  * exp(do.call(rnorm, c(n, params_plus_h)))
  Y_bar_sigma_plus_h <- discount * mean(pmax(S_T_plus_h - K, 0))
  
  est_vega_forward <- (Y_bar_sigma_plus_h - Y_bar_sigma)/h
  est_vega_central<- (Y_bar_sigma_plus_h - Y_bar_sigma_minus_h)/(2*h)
  
  # Store the results in the data frame
  output <- data.frame(h=h, n=n, exact_delta=exact_delta, 
                       est_delta_forward=est_delta_forward, 
                       est_delta_central=est_delta_central,
                       exact_vega=exact_vega,
                       est_vega_forward=est_vega_forward, 
                       est_vega_central=est_vega_central)
  results <- rbind(results, output)
}

################################################################################
# Subset results for Delta 
(delta_results <- results[, c(1,2,3,4,5)])

# Relative error 
abs(results$est_delta_forward - exact_delta) / exact_delta * 100
abs(results$est_delta_central - exact_delta) / exact_delta * 100

# Subset results for Vega
(vega_results <- results[, c(1,2,6,7,8)])

# Relative error 
abs(results$est_vega_forward - exact_vega) / exact_vega * 100
abs(results$est_vega_central - exact_vega) / exact_vega * 100

################################################################################
set.seed(4002)

n_size <- seq(400, 30000, by=400)
est_delta_pathwise <- c(); est_vega_pathwise <- c()
est_delta_lr <- c(); est_vega_lr <- c()

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