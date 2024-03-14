seed <- 2021
set.seed(seed)
y <- rbinom(4000, 1, 0.352)
n1 <- table(y)[1]
n2 <- table(y)[2]

X1 <- c(rnorm(n1)*1.5, rnorm(n2)+1)
X2 <- X1 + c(rnorm(n1)*0.5, -rnorm(n2)/1.5)
X3 <- X1 + runif(n1+n2)
X4 <- c(rnorm(n1)*2.5, -1.5*rnorm(n2)-0.5)
X5 <- X4 + c(rexp(n1)/2, -rexp(n2)/4)
X6 <- rbinom(n1+n2, 2, 0.6)

X_cont <- cbind(X1, X2, X3, X4, X5)

maxi <- apply(X_cont, 2, max)
mini <- apply(X_cont, 2, min)

bin_cont1 <- c()
bin_cont2 <- c()
for (i in 1:ncol(X_cont)){
  width <- (maxi[i]-mini[i])/10
  bin <- cut(X_cont[1:n1,i], breaks=seq(mini[i], maxi[i], width))
  bin_cont1 <- rbind(bin_cont1, as.numeric(table(bin)))
  
  bin <- cut(X_cont[(n1+1):(n1+n2),i], breaks=seq(mini[i], maxi[i], width))
  bin_cont2 <- rbind(bin_cont2, as.numeric(table(bin)))
}

relative_freq1 <- t(apply(bin_cont1, 1, function(x) x/sum(x)))
round(relative_freq1, 4)

relative_freq2 <- t(apply(bin_cont2, 1, function(x) x/sum(x)))
round(relative_freq2, 4)

table(y)
table(X6[1:n1])
table(X6[(n1+1):(n1+n2)])

(cum_density1 <- t(apply(relative_freq1, 1, cumsum)))
(cum_density2 <- t(apply(relative_freq2, 1, cumsum)))

X611 <- table(X6[1:n1])[1]
X612 <- table(X6[1:n1])[2]
X613 <- table(X6[1:n1])[3]

X621 <- table(X6[(n1+1):(n1+n2)])[1]
X622 <- table(X6[(n1+1):(n1+n2)])[2]
X623 <- table(X6[(n1+1):(n1+n2)])[3]

################################################################################
#(a)
B1 <- c(5, 5, 6, 5, 5, 1)

Naive0 <- n1/(n1+n2)*X611/(X611+X612+X613)
Naive1 <- n2/(n1+n2)*X621/(X621+X622+X623)
for (i in 1:nrow(relative_freq1)){
  Naive0 <- Naive0*relative_freq1[i,B1[i]]
  Naive1 <- Naive1*relative_freq2[i,B1[i]]
}

Naive0/(Naive0+Naive1)
Naive1/(Naive0+Naive1)

#(b)
como10 <- max(min(cum_density1[1,B1[1]], cum_density1[2,B1[2]], cum_density1[3,B1[3]]) 
              - max(cum_density1[1,B1[1]-1], cum_density1[2,B1[2]-1], cum_density1[3,B1[3]-1]), 0)
como20 <- max(min(cum_density1[4,B1[4]], cum_density1[5,B1[5]]) 
              - max(cum_density1[4,B1[4]-1], cum_density1[5,B1[5]-1]), 0)
CIBer0 <- como10*como20*X611/(X611+X612+X613)*n1/(n1+n2)

como11 <- max(min(cum_density2[1,B1[1]], cum_density2[2,B1[2]], cum_density2[3,B1[3]]) 
              - max(cum_density2[1,B1[1]-1], cum_density2[2,B1[2]-1], cum_density2[3,B1[3]-1]), 0)
como21 <- max(min(cum_density2[4,B1[4]], cum_density2[5,B1[5]]) 
              - max(cum_density2[4,B1[4]-1], cum_density2[5,B1[5]-1]), 0)
CIBer1 <- como11*como21*X621/(X621+X622+X623)*n2/(n1+n2)

CIBer0/(CIBer0+CIBer1)
CIBer1/(CIBer0+CIBer1)

################################################################################

disc1 <- c()
disc2 <- c()
for (i in 1:ncol(X_cont)){
  width <- (maxi[i]-mini[i]+2e-10)/10
  bin <- cut(X_cont[1:n1,i], breaks=seq(mini[i]-1e-10, maxi[i]+1e-10, width), include.lowest=TRUE)
  levels(bin) <- 1:10
  disc1 <- cbind(disc1, as.numeric(bin))
  
  bin <- cut(X_cont[(n1+1):(n1+n2),i], breaks=seq(mini[i]-1e-10, maxi[i]+1e-10, width), include.lowest=TRUE)
  levels(bin) <- 1:10
  disc2 <- cbind(disc2, as.numeric(bin))
}


Kcor <- function(a, b){
  n <- length(a)
  s <- 0
  for(i in 2:n){
    total <- sign((a[i]-a[1:(i-1)])*(b[i]-b[1:(i-1)]))
    total[total == 0] <- 1
    s <- s+sum(total)
  }
  s/choose(n, 2)
}

b1_1 <- disc1[,1]
b1_2 <- disc1[,2]
b1_3 <- disc1[,3]
b1_4 <- disc1[,4]
b1_5 <- disc1[,5]
b1_6 <- X6[1:n1]

A <- c(Kcor(b1_1, b1_1), Kcor(b1_2, b1_1), Kcor(b1_3, b1_1), Kcor(b1_4, b1_1), Kcor(b1_5, b1_1), Kcor(b1_6, b1_1),
       Kcor(b1_1, b1_2), Kcor(b1_2, b1_2), Kcor(b1_3, b1_2), Kcor(b1_4, b1_2), Kcor(b1_5, b1_2), Kcor(b1_6, b1_2),
       Kcor(b1_1, b1_3), Kcor(b1_2, b1_3), Kcor(b1_3, b1_3), Kcor(b1_4, b1_3), Kcor(b1_5, b1_3), Kcor(b1_6, b1_3),
       Kcor(b1_1, b1_4), Kcor(b1_2, b1_4), Kcor(b1_3, b1_4), Kcor(b1_4, b1_4), Kcor(b1_5, b1_4), Kcor(b1_6, b1_4),
       Kcor(b1_1, b1_5), Kcor(b1_2, b1_5), Kcor(b1_3, b1_5), Kcor(b1_4, b1_5), Kcor(b1_5, b1_5), Kcor(b1_6, b1_5),
       Kcor(b1_1, b1_6), Kcor(b1_2, b1_6), Kcor(b1_3, b1_6), Kcor(b1_4, b1_6), Kcor(b1_5, b1_6), Kcor(b1_6, b1_6))

A1 <- matrix(A, ncol=6)
round(1 - A1, 4)

################################################################################
b2_1 <- disc2[,1]
b2_2 <- disc2[,2]
b2_3 <- disc2[,3]
b2_4 <- disc2[,4]
b2_5 <- disc2[,5]
b2_6 <- X6[(n1+1):(n1+n2)]

A <- c(Kcor(b2_1, b2_1), Kcor(b2_2, b2_1), Kcor(b2_3, b2_1), Kcor(b2_4, b2_1), Kcor(b2_5, b2_1), Kcor(b2_6, b2_1),
       Kcor(b2_1, b2_2), Kcor(b2_2, b2_2), Kcor(b2_3, b2_2), Kcor(b2_4, b2_2), Kcor(b2_5, b2_2), Kcor(b2_6, b2_2),
       Kcor(b2_1, b2_3), Kcor(b2_2, b2_3), Kcor(b2_3, b2_3), Kcor(b2_4, b2_3), Kcor(b2_5, b2_3), Kcor(b2_6, b2_3),
       Kcor(b2_1, b2_4), Kcor(b2_2, b2_4), Kcor(b2_3, b2_4), Kcor(b2_4, b2_4), Kcor(b2_5, b2_4), Kcor(b2_6, b2_4),
       Kcor(b2_1, b2_5), Kcor(b2_2, b2_5), Kcor(b2_3, b2_5), Kcor(b2_4, b2_5), Kcor(b2_5, b2_5), Kcor(b2_6, b2_5),
       Kcor(b2_1, b2_6), Kcor(b2_2, b2_6), Kcor(b2_3, b2_6), Kcor(b2_4, b2_6), Kcor(b2_5, b2_6), Kcor(b2_6, b2_6))

A2 <- matrix(A, ncol=6)
round(1 - A2, 4)