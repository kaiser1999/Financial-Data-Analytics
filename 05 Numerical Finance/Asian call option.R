# Initialize variables
S_0 <- 10; K <- 8; r <- 0.05; sigma <- 0.3; T <- 1
M_lst <- c(1e1, 1e2, 1e3); n <- 1e5

Sim_Asian <- function(n, M, S_0, K, r, sigma, T, theta){
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
  
  list(Eul=list(price=S_Eul, payoff=exp(-r*T)*pmax(avg_Eul-K, 0)), 
       Mil=list(price=S_Mil, payoff=exp(-r*T)*pmax(avg_Mil-K, 0)), 
       Ext=list(price=S_Ext, payoff=exp(-r*T)*pmax(avg_Ext-K, 0)))
}

################################################################################
BS_results <- data.frame(M=M_lst, n=n, Asian_Eul=0, 
                         Asian_Mil=0, Asian_Ext=0)
par(mfrow=c(1,2))
for (i in 1:length(M_lst)){
  M <- M_lst[i]
  set.seed(4002)
  Asian_BS <- Sim_Asian(n, M, S_0, K, r, sigma, T, 2)
  BS_results$Asian_Eul[i] <- mean(Asian_BS$Eul$payoff)
  BS_results$Asian_Mil[i] <- mean(Asian_BS$Mil$payoff)
  BS_results$Asian_Ext[i] <- mean(Asian_BS$Ext$payoff)
  
  Eul_diff <- Asian_BS$Ext$price - Asian_BS$Eul$price
  Mil_diff <- Asian_BS$Ext$price - Asian_BS$Mil$price
  print(paste(diff(range(Eul_diff)), diff(range(Mil_diff)), 
              diff(range(Eul_diff))/diff(range(Mil_diff))))
  hist(Eul_diff, xlim=c(-max(abs(Eul_diff)), max(abs(Eul_diff))),
       xlab="Error", main="Euler Scheme")
  hist(Mil_diff, xlim=c(-max(abs(Mil_diff)), max(abs(Mil_diff))),
       xlab="Error", main="Milstein Scheme")
}
BS_results

################################################################################
M <- 1e3; n_lst <- c(1e3, 1e4, 1e5, 1e6, 1e7)
BS_results <- data.frame(M=M, n=n_lst, Asian_Eul=0, 
                         Asian_Mil=0, Asian_Ext=0)

for (i in 1:length(n_lst)){
  n <- n_lst[i]
  set.seed(4002)
  Asian_BS <- Sim_Asian(n, M, S_0, K, r, sigma, T, 2)
  BS_results$Asian_Eul[i] <- mean(Asian_BS$Eul$payoff)
  BS_results$Asian_Mil[i] <- mean(Asian_BS$Mil$payoff)
  BS_results$Asian_Ext[i] <- mean(Asian_BS$Ext$payoff)
  
  Eul_diff <- Asian_BS$Ext$price - Asian_BS$Eul$price
  Mil_diff <- Asian_BS$Ext$price - Asian_BS$Mil$price
  print(paste(diff(range(Eul_diff)), diff(range(Mil_diff)), 
              diff(range(Eul_diff))/diff(range(Mil_diff))))
}
BS_results

################################################################################
control_variate <- function(x, y, mu_y){
  mean(x) - cov(x, y)/var(y) * (mean(y) - mu_y)
}

theta <- 1; epochs <- 100
CEV_results <- data.frame(M=M_lst, n=n, theta=theta,
                          mu_Eul=0, mu_Mil=0, mu_Eul_cv=0, mu_mil_cv=0,
                          sd_Eul=0, sd_Mil=0, sd_Eul_cv=0, sd_mil_cv=0)
set.seed(4002)
benchmark <- Sim_Asian(1e6, 1e4, S_0, K, r, sigma, T, theta)
Asian_Eul <- array(NA, epochs); Asian_Mil <- array(NA, epochs)
Asian_Eul_cv <- array(NA, epochs); Asian_Mil_cv <- array(NA, epochs)
mu_y <- mean(benchmark$Eul$price)
for (i in 1:length(M_lst)){
  M <- M_lst[i]
  set.seed(4002)
  prog_bar <- txtProgressBar(min=0, max=epochs, width=50, style=3)
  for (j in 1:epochs){
    Asian_theta <- Sim_Asian(n, M, S_0, K, r, sigma, T, theta)
    
    Asian_Eul[j] <- mean(Asian_theta$Eul$payoff)
    Asian_Mil[j] <- mean(Asian_theta$Mil$payoff)
    Asian_Eul_cv[j] <- control_variate(Asian_theta$Eul$payoff, 
                                       Asian_theta$Eul$price, mu_y)
    Asian_Mil_cv[j] <- control_variate(Asian_theta$Mil$payoff, 
                                       Asian_theta$Mil$price, mu_y)
    setTxtProgressBar(prog_bar, j); cat(paste0(" M = ", M))
  }
  CEV_results[i,4:11] <- c(mean(Asian_Eul), mean(Asian_Mil), 
                           mean(Asian_Eul_cv), mean(Asian_Mil_cv),
                           sd(Asian_Eul), sd(Asian_Mil), 
                           sd(Asian_Eul_cv), sd(Asian_Mil_cv))
}

CEV_results
mean(benchmark$Eul$payoff); mean(benchmark$Mil$payoff)

################################################################################
theta <- 1.8; epochs <- 100
CEV_results <- data.frame(M=M_lst, n=n, theta=theta,
                          mu_Eul=0, mu_Mil=0, mu_Eul_cv=0, mu_mil_cv=0,
                          sd_Eul=0, sd_Mil=0, sd_Eul_cv=0, sd_mil_cv=0)
set.seed(4002)
benchmark <- Sim_Asian(1e6, 1e4, S_0, K, r, sigma, T, theta)
Asian_Eul <- array(NA, epochs); Asian_Mil <- array(NA, epochs)
Asian_Eul_cv <- array(NA, epochs); Asian_Mil_cv <- array(NA, epochs)

mu_y <- mean(benchmark$Eul$price)
for (i in 1:length(M_lst)){
  M <- M_lst[i]
  set.seed(4002)
  prog_bar <- txtProgressBar(min=0, max=epochs, style=3, label=M)
  for (j in 1:epochs){
    Asian_theta <- Sim_Asian(n, M, S_0, K, r, sigma, T, theta)
    
    Asian_Eul[j] <- mean(Asian_theta$Eul$payoff)
    Asian_Mil[j] <- mean(Asian_theta$Mil$payoff)
    
    Asian_Eul_cv[j] <- control_variate(Asian_theta$Eul$payoff, 
                                       Asian_theta$Eul$price, mu_y)
    Asian_Mil_cv[j] <- control_variate(Asian_theta$Mil$payoff, 
                                       Asian_theta$Mil$price, mu_y)
    setTxtProgressBar(prog_bar, j); cat(paste0(" M = ", M))
  }
  CEV_results[i,4:11] <- c(mean(Asian_Eul), mean(Asian_Mil), 
                           mean(Asian_Eul_cv), mean(Asian_Mil_cv),
                           sd(Asian_Eul), sd(Asian_Mil), 
                           sd(Asian_Eul_cv), sd(Asian_Mil_cv))
}

CEV_results
mean(benchmark$Eul$payoff); mean(benchmark$Mil$payoff)