Default <- c(rep(0, 9), rep(1, 6))
Risk <- c(rep(0, 6), rep(1, 3), rep(0, 3), rep(1, 3))
Sex <- c(0, 0, rep(1, 4), rep(0, 4), 1, 1, rep(0, 3))

p <- table(Default)/length(Default)
(Entropy_S <- -sum(p*log2(p)))

Default_L <- Default[Risk==0]
p_L <- table(Default_L)/length(Default_L)
(Entropy_L <- -sum(p_L*log2(p_L)))

Default_H <- Default[Risk==1]
p_H <- table(Default_H)/length(Default_H)
(Entropy_H <- -sum(p_H*log2(p_H)))

Default_F <- Default[Sex==0]
p_F <- table(Default_F)/length(Default_F)
(Entropy_F <- -sum(p_F*log2(p_F)))

Default_M <- Default[Sex==1]
p_M <- table(Default_M)/length(Default_M)
(Entropy_M <- -sum(p_M*log2(p_M)))

(IG_R <- Entropy_S - length(Risk[Risk==0])/length(Default)*Entropy_L - length(Risk[Risk==1])/length(Default)*Entropy_H)
(IG_S <- Entropy_S - length(Sex[Sex==0])/length(Default)*Entropy_F - length(Sex[Sex==1])/length(Default)*Entropy_M)