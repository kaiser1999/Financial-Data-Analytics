library(graphics)
library(quadprog)

x1 <- seq(-5, 1, 0.01)
x2 <- seq(-3, 3, 0.01)
P <- matrix(c(2, 1, 1, 6), ncol=2)
q <- c(7, 3)
G.T <- matrix(c(1, 0, -1, 0, 0, 1, 0, -1), ncol=2, byrow=T)
h <- c(1, 1, 1, 1)

QPex <- function(x_1, x_2){
  # x: A 2-D matrix of shape 2 x total number of grid
  x <- matrix(c(x_1, x_2), nrow=2, byrow=TRUE)
  as.numeric(rowSums(t(x) %*% P * t(x))/2 + t(q) %*% x)
}

z <- outer(x1, x2, FUN=QPex)
contour(x1, x2, z, nlevels=8, col="green", lty=2, lwd=3, labcex=2)
polygon(c(-1, -1, 1, 1, -1), c(-1, 1, 1, -1, -1), 
        col=adjustcolor("gray", alpha.f=0.5), border="black", lwd=2)

# minimize (1/2 x^T P x + q^T x) with constraints (G^T x <= h)
# solve.QP: (1/2 x^T P x - (-q^T) x) with constraints (-(G^T)^T x >= -h)
sol <- solve.QP(P, -q, -t(G.T), bvec=-h)
(real_ans <- sol$unconstrained.solution)
(const_ans <- sol$solution)

contour(x1, x2, z, levels=sol$value, col="purple", 
        lty=1, lwd=3, labcex=2, add=TRUE)
points(real_ans[1], real_ans[2], pch=19, col="red", cex=2)
points(const_ans[1], const_ans[2], pch="*", col="red", cex=3)
title(xlab=expression(italic("x")^"(1)"), 
      ylab=expression(italic("x")^"(2)"), cex.lab=1.5)