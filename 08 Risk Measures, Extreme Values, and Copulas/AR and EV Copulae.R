library(copula)
library(evd)              # frechet
library(LaplacesDemon)    # pareto, laplace, halfnorm

################################################################################
# "copula" parameter in mvdc() can be any one of the following: 
#(01) galambosCopula(2)
#(02) huslerReissCopula(0.3)
#(03) tawnCopula(0.3)
#(04) tevCopula(0.3, df=2, df.fixed=TRUE)
#(05) amhCopula(1, dim=2)
#(06) gumbelCopula(1, dim=2)
#(07) claytonCopula(1, dim=2)
#(08) frankCopula(3, dim=2)
#(09) joeCopula(1, dim=2)
#(10) fgmCopula(c(0.2,-0.2,-0.4), dim=2)
#(11) plackettCopula(param=2)
#(12) normalCopula(0.5, dim=3)
#(13) tCopula(0.5, dim=2, dispstr="toep", df=2, df.fixed=TRUE)

# "margins" and "paramMargins" parameters in mdvc() can be any one of the following:
# e.g. norm as long as pnorm, dnorm, qnorm, rnorm are all available
#(01) frechet: list(loc=0, scale=1, shape=2)
#(02) pareto: list(alpha=2)
#(03) exp: list(rate=1)
#(04) gamma: list(shape=1)
#(05) laplace: list(location=0, scale=1)
#(06) halfnorm: list(scale=1)
#(07) beta: list(shape1=2, shape2=3)
#(08) norm: list(mean=0, sd=1)

margin_lst <- list(frechet=list(name="Frechet", params=c("shape")),
                   pareto=list(name="Pareto", params=c("alpha")),
                   exp=list(name="Exp", params=c("rate")),
                   gamma=list(name="Gamma", params=c("shape")),
                   laplace=list(name="laplace", params=c("location", "scale")),
                   halfnorm=list(name="HalfNorm", params=c("scale")),
                   beta=list(name="Beta", params=c("shape1", "shape2")),
                   norm=list(name="Norm", params=c("mean", "sd")))

# beware Norm(mu,sd) here NOT Norm(mu,sd^2)!
get_legend <- function(margins, paramMargins){
  text.legend <- c()
  for (i in 1:length(margins)){
    mar <- margins[i]; par <- paramMargins[[i]]
    dist <- margin_lst[[mar]]
    params <- paste0(par[dist$params], collapse=",")
    text.legend[i] <- paste0(dist$name, "(", params, ")")
  }
  paste0(text.legend, collapse="+")
}

################################################################################
size <- 10^7
seed <- 4002

p <- seq(0.999, 0.99999, 0.000001)
get_quantile <- function(Z, p) quantile(rowSums(Z), p)/rowSums(apply(Z, 2, quantile, p))

################################################################################
### Clayton Copula with theta=1 ###
################################################################################
margins1 <- c("frechet", "frechet")
paramMargins1 <- list(list(loc=0, scale=1, shape=2),
                      list(loc=0, scale=1, shape=2))

margins2 <- c("frechet", "frechet")
paramMargins2 <- list(list(loc=0, scale=1, shape=1),
                      list(loc=0, scale=1, shape=1))

margins3 <- c("frechet", "frechet")
paramMargins3 <- list(list(loc=0, scale=1, shape=0.5),
                      list(loc=0, scale=1, shape=0.5))

set.seed(seed)
Z1 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins1)), margins=margins1, paramMargins=paramMargins1))
Z2 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins2)), margins=margins2, paramMargins=paramMargins2))
Z3 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins3)), margins=margins3, paramMargins=paramMargins3))

plot(p, get_quantile(Z1, p), lty=1, type="l", col="blue", lwd=2, 
     ylim=c(0.5, 3), pch=16, xlab="p", ylab="V@R Ratio", cex.main=1)
lines(p, get_quantile(Z2, p), lty=2, type="l", col="black", lwd=2)
lines(p, get_quantile(Z3, p), lty=3, type="l", col="red", lwd=2)
abline(h=c(1), lwd=1.5, lty=4, col="orange")
text.legend <- c(get_legend(margins1, paramMargins1), get_legend(margins2, paramMargins2), get_legend(margins3, paramMargins3))
legend("topleft", legend=text.legend, lty=1:3, col=c("blue", "black", "red"), lwd=2, cex=1, 
       text.width=max(strwidth(text.legend))*1.2)

################################################################################
margins1 <- c("frechet", "frechet")
paramMargins1 <- list(list(loc=0, scale=1, shape=2),
                      list(loc=0, scale=1, shape=0.5))

margins2 <- c("frechet", "exp")
paramMargins2 <- list(list(loc=0, scale=1, shape=2),
                      list(rate=1))

margins3 <- c("frechet", "beta")
paramMargins3 <- list(list(loc=0, scale=1, shape=2),
                      list(shape1=2, shape2=3))

set.seed(seed)
Z1 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins1)), margins=margins1, paramMargins=paramMargins1))
Z2 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins2)), margins=margins2, paramMargins=paramMargins2))
Z3 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins3)), margins=margins3, paramMargins=paramMargins3))

plot(p, get_quantile(Z1, p), lty=1, type="l", col="blue", lwd=2, 
     ylim=c(0.8, 1.2), pch=16, xlab="p", ylab="V@R Ratio", cex.main=1)
lines(p, get_quantile(Z2, p), lty=2, type="l", col="black", lwd=2)
lines(p, get_quantile(Z3, p), lty=3, type="l", col="red", lwd=2)
abline(h=c(1), lwd=1.5, lty=4, col="orange")
text.legend <- c(get_legend(margins1, paramMargins1), get_legend(margins2, paramMargins2), get_legend(margins3, paramMargins3))
legend("topleft", legend=text.legend, lty=1:3, col=c("blue", "black", "red"), lwd=2, cex=1, 
       text.width=max(strwidth(text.legend))*1.2)

################################################################################
margins1 <- c("frechet", "pareto", "exp", "norm", "beta")
paramMargins1 <- list(list(loc=0, scale=1, shape=2),
                      list(alpha=2),
                      list(rate=1),
                      list(mean=1, sd=1),
                      list(shape1=2, shape2=3))

margins2 <- c("frechet", "pareto", "exp", "norm", "beta")
paramMargins2 <- list(list(loc=0, scale=1, shape=1),
                      list(alpha=1),
                      list(rate=1),
                      list(mean=1, sd=1),
                      list(shape1=2, shape2=3))

margins3 <- c("frechet", "pareto", "exp", "norm", "beta")
paramMargins3 <- list(list(loc=0, scale=1, shape=0.5),
                      list(alpha=0.5),
                      list(rate=1),
                      list(mean=1, sd=1),
                      list(shape1=2, shape2=3))
set.seed(seed)
Z1 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins1)), margins=margins1, paramMargins=paramMargins1))
Z2 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins2)), margins=margins2, paramMargins=paramMargins2))
Z3 <- rMvdc(size, mvdc(copula=claytonCopula(1, dim=length(margins3)), margins=margins3, paramMargins=paramMargins3))

plot(p, get_quantile(Z1, p), lty=1, type="l", col="blue", lwd=2, 
     ylim=c(0.5, 3), pch=16, xlab="p", ylab="V@R Ratio", cex.main=1)
lines(p, get_quantile(Z2, p), lty=2, type="l", col="black", lwd=2)
lines(p, get_quantile(Z3, p), lty=3, type="l", col="red", lwd=2)
abline(h=c(1), lwd=1.5, lty=4, col="orange")
text.legend <- c(get_legend(margins1, paramMargins1), get_legend(margins2, paramMargins2), get_legend(margins3, paramMargins3))
legend("topleft", legend=text.legend, lty=1:3, col=c("blue", "black", "red"), lwd=2, cex=1, 
       text.width=max(strwidth(text.legend))*1.2)

################################################################################
### Gumbel Copula with alpha=1.5 ###
################################################################################
margins1 <- c("frechet", "frechet")
paramMargins1 <- list(list(loc=0, scale=1, shape=2),
                      list(loc=0, scale=1, shape=2))

margins2 <- c("frechet", "frechet")
paramMargins2 <- list(list(loc=0, scale=1, shape=1),
                      list(loc=0, scale=1, shape=1))

margins3 <- c("frechet", "frechet")
paramMargins3 <- list(list(loc=0, scale=1, shape=0.5),
                      list(loc=0, scale=1, shape=0.5))

set.seed(seed)
Z1 <- rMvdc(size, mvdc(copula=gumbelCopula(1.5, dim=length(margins1)), margins=margins1, paramMargins=paramMargins1))
Z2 <- rMvdc(size, mvdc(copula=gumbelCopula(1.5, dim=length(margins2)), margins=margins2, paramMargins=paramMargins2))
Z3 <- rMvdc(size, mvdc(copula=gumbelCopula(1.5, dim=length(margins3)), margins=margins3, paramMargins=paramMargins3))

plot(p, get_quantile(Z1, p), lty=1, type="l", col="blue", lwd=2, 
     ylim=c(0.8, 2), pch=16, xlab="p", ylab="V@R Ratio", cex.main=1)
lines(p, get_quantile(Z2, p), lty=2, type="l", col="black", lwd=2)
lines(p, get_quantile(Z3, p), lty=3, type="l", col="red", lwd=2)
abline(h=c(1), lwd=1.5, lty=4, col="orange")
text.legend <- c(get_legend(margins1, paramMargins1), get_legend(margins2, paramMargins2), get_legend(margins3, paramMargins3))
legend("topleft", legend=text.legend, lty=1:3, col=c("blue", "black", "red"), lwd=2, cex=1, 
       text.width=max(strwidth(text.legend))*1.2)

################################################################################
margins1 <- c("exp", "exp")
paramMargins1 <- list(list(rate=1),
                      list(rate=1))

margins2 <- c("halfnorm", "halfnorm")
paramMargins2 <- list(list(scale=1),
                      list(scale=1))

margins3 <- c("beta", "beta")
paramMargins3 <- list(list(shape1=2, shape2=3),
                      list(shape1=2, shape2=3))

set.seed(seed)
Z1 <- rMvdc(size, mvdc(copula=gumbelCopula(1.5, dim=length(margins1)), margins=margins1, paramMargins=paramMargins1))
Z2 <- rMvdc(size, mvdc(copula=gumbelCopula(1.5, dim=length(margins2)), margins=margins2, paramMargins=paramMargins2))
Z3 <- rMvdc(size, mvdc(copula=gumbelCopula(1.5, dim=length(margins3)), margins=margins3, paramMargins=paramMargins3))

plot(p, get_quantile(Z1, p), lty=1, type="l", col="blue", lwd=2, 
     ylim=c(0.9, 1.05), pch=16, xlab="p", ylab="V@R Ratio", cex.main=1)
lines(p, get_quantile(Z2, p), lty=2, type="l", col="black", lwd=2)
abline(h=c(1), lwd=1.5, lty=4, col="orange")
text.legend <- c(get_legend(margins1, paramMargins1), get_legend(margins2, paramMargins2))
legend("bottomleft", legend=text.legend, lty=1:2, col=c("blue", "black"), lwd=2, cex=1, 
       text.width=max(strwidth(text.legend))*1.2)


plot(p, get_quantile(Z3, p), lty=3, type="l", col="red", lwd=2, 
     ylim=c(0.9, 1.05), pch=16, xlab="p", ylab="V@R Ratio", cex.main=1)
abline(h=c(1), lwd=1.5, lty=4, col="orange")
text.legend <- c(get_legend(margins3, paramMargins3))
legend("bottomleft", legend=text.legend, lty=3, col=c("red"), lwd=2, cex=1, 
       text.width=max(strwidth(text.legend))*1.2)

################################################################################
size <- 2000
seed <- 4002

################################################################################
### Archimedean Copulae ###
################################################################################
par.old <- par()
par(mfrow=c(1,2), mar=c(2, 2, 2, 2))

################################################################################
set.seed(seed)
plot(rCopula(size, amhCopula(0.2, dim=2)), col="blue", ylab="", xlab="")
plot(rCopula(size, amhCopula(0.9, dim=2)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, claytonCopula(-0.5, dim=2)), col="blue", ylab="", xlab="")
plot(rCopula(size, claytonCopula(8, dim=2)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, frankCopula(2, dim=2)), col="blue", ylab="", xlab="")
plot(rCopula(size, frankCopula(8, dim=2)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, gumbelCopula(2, dim=2)), col="blue", ylab="", xlab="")
plot(rCopula(size, gumbelCopula(6, dim=2)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, joeCopula(2, dim=2)), col="blue", ylab="", xlab="")
plot(rCopula(size, joeCopula(6, dim=2)), col="blue", ylab="", xlab="")

################################################################################
### Extreme Value Copulae ###
################################################################################
set.seed(seed)
plot(rCopula(size, galambosCopula(2)), col="blue", ylab="", xlab="")
plot(rCopula(size, galambosCopula(8)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, huslerReissCopula(2)), col="blue", ylab="", xlab="")
plot(rCopula(size, huslerReissCopula(8)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, tevCopula(-0.9, df=2, df.fixed=TRUE)), col="blue", ylab="", xlab="")
plot(rCopula(size, tevCopula(0.8, df=2, df.fixed=TRUE)), col="blue", ylab="", xlab="")

set.seed(seed)
plot(rCopula(size, tawnCopula(0.2)), col="blue", ylab="", xlab="")
plot(rCopula(size, tawnCopula(0.7)), col="blue", ylab="", xlab="")

################################################################################
par(par.old)