# Initialize variables
S_0 <- 10; r <- 0; sigma <- 0.7; T <- 1; K <- 8
M_values <- c(50, 100, 150)
path_values <- c(1e4, 1e5, 1e6)

# Create a data frame to store results
results <- data.frame(M=rep(M_values, each=length(path_values)),
                      Path=rep(path_values, times=length(M_values)),
                      Estimate=0)

set.seed(4002)
for (i in 1:nrow(results)){
  M <- results$M[i]
  n <- results$Path[i]
  
  delta_t <- T / M
  prices <- S_0
  avg_price <- S_0 / (M+1)
  for (m in 1:M){
    z <- rnorm(n)
    prices <- prices * (1 + r*delta_t + sigma*sqrt(delta_t)*z)
    avg_price <- avg_price + prices / (M+1)
  }
  
  payoff_T <- exp(-r*T) * pmax(avg_price - K, 0)
  results$Estimate[i] <- mean(payoff_T)
}

results