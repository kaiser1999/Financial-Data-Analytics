# Initialize variables
S_0 <- 10; K <- 8; r <- 0.05; sigma <- 0.3; T <- 1
M_lst <- c(50, 100, 150); n <- 5e4

Sim_call <- function(n, M, S_0, K, r, sigma, T, theta){
  delta_t <- T / M
  
  Euler <- function(z, S_t, theta){
    S_t*(1 + r*delta_t) + sigma*S_t^(theta/2)*sqrt(delta_t)*z
  }
  
  Milstein <- function(z, S_t, theta){
    (Euler(z, S_t, theta) + 
       sigma^2*theta/2*S_t^(theta-1)*delta_t/2*(z^2-1))
  }
  
  Exact <- function(z, S_t, theta=2){
    S_t*exp((r - sigma^2/2)*delta_t + sigma*sqrt(delta_t)*z)
  }
  
  # Asian_call
  S_Eul <- S_0; avg_Eul <- S_0 / (M+1)
  S_Mil <- S_0; avg_Mil <- S_0 / (M+1)
  if (theta == 2) {
    S_Ext <- S_0; avg_Ext <- S_0 / (M+1)
  } else {
    S_Ext <- NA; avg_Ext <- NA
  }

  for (m in 1:M){
    z <- rnorm(n)
    S_Eul <- Euler(z, S_Eul, theta)
    avg_Eul <- avg_Eul + S_Eul/(M+1)
    
    S_Mil <- Milstein(z, S_Mil, theta)
    avg_Mil <- avg_Mil + S_Mil/(M+1)
    
    if (theta == 2){
      S_Ext <- Exact(z, S_Ext, theta)
      avg_Ext <- avg_Ext + S_Ext/(M+1)
    }
  }
  
  list(Eul=list(price=S_Eul, payoff=exp(-r*T)*mean(avg_Eul)), 
       Mil=list(price=S_Mil, payoff=exp(-r*T)*mean(avg_Mil)), 
       Ext=list(price=S_Ext, payoff=exp(-r*T)*mean(avg_Ext)))
}

Sim_Milstein <- function(n, M, S_0, K, r, sigma, T, theta){
  delta_t <- T / M; S_t <- S_0
  for (m in 1:M){
    z <- rnorm(n)
    S_t <- (S_t*(1 + r*delta_t) + sigma*S_t^(theta/2)*sqrt(delta_t)*z
            + sigma^2*theta/2*S_t^(theta-1)*delta_t/2*(z^2-1))
  }
  S_t
}

################################################################################
BS_results <- data.frame(M=M_lst, n=n, theta=2,
                         Asian_Eul=0, Asian_Mil=0, Asian_Ext=0)
par(mfrow=c(1,2))
set.seed(4002)
for (i in 1:length(M_lst)){
  M <- M_lst[i]
  Asian_BS <- Sim_call(n, M, S_0, K, r, sigma, T, 2)
  BS_results$Asian_Eul[i] <- Asian_BS$Eul$payoff
  BS_results$Asian_Mil[i] <- Asian_BS$Mil$payoff
  BS_results$Asian_Ext[i] <- Asian_BS$Ext$payoff
  
  Eul_diff <- Asian_BS$Ext$price - Asian_BS$Eul$price
  Mil_diff <- Asian_BS$Ext$price - Asian_BS$Mil$price
  hist(Eul_diff, xlim=c(-max(abs(Eul_diff)), max(abs(Eul_diff))),
       xlab="Error", main="Euler Scheme")
  hist(Mil_diff, xlim=c(-max(abs(Mil_diff)), max(abs(Mil_diff))),
       xlab="Error", main="Milstein Scheme")
}
BS_results

################################################################################
theta <- 1
CEV_results <- data.frame(M=M_lst, n=n, theta=theta,
                          Asian_Eul=0, Asian_Mil=0, Asian_Ext=0)
set.seed(4002)
bootstrapped <- Sim_Milstein(1e6, 1e4, S_0, K, r, sigma, T, theta)
i <- ((1:n) - 0.5)/n
q <- quantile(bootstrapped, probs=i, type=4, names=FALSE)
for (i in 1:length(M_lst)){
  M <- M_lst[i]
  Asian_theta <- Sim_call(n, M, S_0, K, r, sigma, T, theta)
  CEV_results$Asian_Eul[i] <- Asian_theta$Eul$payoff
  CEV_results$Asian_Mil[i] <- Asian_theta$Mil$payoff
  
  qqplot(q, sort(Asian_theta$Eul$price), 
         ylab="Sample quantiles", main="Euler scheme")
  abline(lsfit(q, sort(Asian_theta$Eul$price)))
  
  qqplot(q, sort(Asian_theta$Mil$price), 
         ylab="Sample quantiles", main="Milstein scheme")
  abline(lsfit(q, sort(Asian_theta$Mil$price)))
  
  Eul_diff <- q - sort(Asian_theta$Eul$price)
  Mil_diff <- q - sort(Asian_theta$Mil$price)
  hist(Eul_diff, xlab="Error", main="Euler Scheme", 
       breaks=1000, xlim=c(-0.03, 0.03))
  hist(Mil_diff, xlab="Error", main="Milstein Scheme", 
       breaks=1000, xlim=c(-0.03, 0.03))
  
}
CEV_results

################################################################################
theta <- 1.8
CEV_results <- data.frame(M=M_lst, n=n, theta=theta,
                          Asian_Eul=0, Asian_Mil=0)
set.seed(4002)
bootstrapped <- Sim_Milstein(1e6, 1e4, S_0, K, r, sigma, T, theta)
i <- ((1:n) - 0.5)/n
q <- quantile(bootstrapped, probs=i, type=4, names=FALSE)
for (i in 1:length(M_lst)){
  M <- M_lst[i]
  Asian_theta <- Sim_call(n, M, S_0, K, r, sigma, T, theta)
  CEV_results$Asian_Eul[i] <- Asian_theta$Eul$payoff
  CEV_results$Asian_Mil[i] <- Asian_theta$Mil$payoff
  
  qqplot(q, sort(Asian_theta$Eul$price), 
         ylab="Sample quantiles", main="Euler scheme")
  abline(lsfit(q, sort(Asian_theta$Eul$price)))
  
  qqplot(q, sort(Asian_theta$Mil$price), 
         ylab="Sample quantiles", main="Milstein scheme")
  abline(lsfit(q, sort(Asian_theta$Mil$price)))
}
CEV_results