# Read in data (a CSV file) under Dataset
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("../Datasets/us-rate.csv")  # Read in data

label <- c("1m","3m","6m","9m","12m","18m","2y",
           "3y","4y","5y","7y","10y","15y")
names(d) <- label         # Apply labels
options(digits=2)         # Display the number of 2 digits
cor(d)                    # Compute correlation matrix

################################################################################
## Find PC"s using built in package
options(digits=4)         # Display the number of 4 digits
pca <- princomp(d, cor=T)
pca$loadings[,1:6]  # Display first six loadings of PCAs

################################################################################
pc1 <- pca$loadings[,1]   # Save the loading of 1st PC
pc2 <- pca$loadings[,2]   # Save the loading of 2nd PC

pc1 %*% pc1
pc2 %*% pc2
pc1 %*% pc2

################################################################################

(s <- pca$sdev)   # Save the s.d. of all PC's to s
round(s^2, 4)     # Display the variances of all PC"s
(t <- sum(s^2))   # Compute total variance (should equal 13)
round(s^2/t, 4)   # Proportion of variance explained by PC's
cumsum(s^2/t)     # Cumulative sum of proportion of variance

################################################################################
### Scree Plot ###
# screeplot: plot variances against number of principle components
par(mfrow=c(1,1))
screeplot(pca, type="lines")

################################################################################
pc1 <- pca$loadings[,1]   # Save the loading of 1st PC
pc2 <- -pca$loadings[,2]  # Save the loading of 2nd PC
pc3 <- pca$loadings[,3]   # Save the loading of 3rd PC

par(mfrow=c(3,1))         # Multi-frame for plotting
plot(pc1, ylim=c(-0.6, 0.6), type="o")
plot(pc2, ylim=c(-0.6, 0.6), type="o")
plot(pc3, ylim=c(-0.6, 0.6), type="o")

################################################################################
score <- pca$scores[,1:3]		# save scores of PC1-PC3
pairs(score)			          # scatterplot of scores

################################################################################

pr_comp <- function(X, center=TRUE, scale=TRUE, method="SD"){
  X <- as.matrix(X)
  N <- nrow(X)
  feature_name <- colnames(X)
  ones = rep(1, N)
  if (center){ X <- X - ones %*% t(colMeans(X))}
  # rescale with population standard deviation
  if (scale) { 
    X <- X %*% diag(1/sqrt(apply(X, 2, var)))
    X <- X * sqrt(N/(N-1))
  }
  matr <- cov(X)
  if (method=="SD"){           # spectral decomposition
    eig <- eigen(matr)        
    loadings <- eig$vectors
    std <- sqrt(eig$values)
  }else{                       # singular value decomposition
    svdcom <- svd(matr)
    loadings <- svdcom$v
    std <- sqrt(svdcom$d)
  }
  # computing sample standard deviation, different from Python
  std <- std * sqrt((N-1)/N)
  rownames(loadings) <- feature_name
  colnames(loadings) <- paste0("PC", 1:ncol(X))
  score <- X %*% loadings
  return (list(loadings=loadings, sdev=std, scores=score))
}

pca2 <- pr_comp(d)
all.equal(as.numeric(pca2$sdev), as.numeric(pca$sdev))

################################################################################

d <- read.csv("../Datasets/us stocks.csv", header=TRUE, row.names=1)
# Compute daily logarithmic return for each equity
r <- sapply(d, function(x) diff(log(x)))
rownames(r) <- rownames(d)[-1]

spx <- read.csv("../Datasets/spx_2020.csv", header=TRUE, row.names=1)

PCA <- prcomp(r)
pc1 <- PCA$rotation[,1]        # Save the loading of 1st PC

# Select 10 Equities to invest base on the loadings of PC1
n_stock <- 10
investment <- 10000
short_r <- 0.5
short_interest <- 0.08
base_fee <- 2.05

select_long <- order(pc1, decreasing=FALSE)[1:n_stock]
sort(pc1, decreasing=FALSE)[1:n_stock]
select_short <- order(pc1, decreasing=TRUE)[1:n_stock]
sort(pc1, decreasing=TRUE)[1:n_stock]

library(zoo)
date <- as.Date(rownames(d), format="%d/%m/%Y")
equity_price <- as.matrix(d)
index_price <- as.matrix(spx)

pca_long_n <- floor(investment/equity_price[1, select_long])
pca_long_remain <- as.numeric(investment*n_stock 
                              - equity_price[1, select_long] %*% pca_long_n)
pca_long_com <- sum(sapply(pca_long_n, function(x) max(base_fee, 0.013*x)))
pca_long_stock <- equity_price[-1, select_long] %*% pca_long_n
pca_long_val <- zoo(pca_long_stock - pca_long_com + pca_long_remain)
index(pca_long_val) <- date[-1]

pca_short_n <- floor(investment*short_r/equity_price[1, select_short])
pca_short_remain <- as.numeric(equity_price[1, select_short] %*% pca_short_n)
pca_short_com <- sum(sapply(pca_short_n, function(x) max(base_fee, 0.013*x)))
pca_short_stock <- equity_price[-1, select_short] %*% pca_short_n
short_days <- date[-1] - date[1]
pca_short_interest <- short_interest * short_days/365 * pca_short_remain
pca_short_val <- zoo(investment*n_stock + pca_short_remain 
                     - pca_short_stock - pca_short_com - pca_short_interest)
index(pca_short_val) <- date[-1]

market_n <- floor(investment*n_stock/index_price[1,])
market_remain <- investment*n_stock - index_price[1,]*market_n
market_com <- max(base_fee, 0.013*market_n)
market_stock <- market_n*index_price[-1,]
market_val <- zoo(market_stock + market_remain - market_com)
index(market_val) <- date[-1]

par(mfrow=c(1,1))
plot_date <- date[seq(2, nrow(r), 20)]
plot(pca_long_val, ylim=c(investment*7, investment*14), 
     col='blue', lwd=2, xaxt="n", ylab="")
lines(pca_short_val, col='red', lwd=2)
lines(market_val, col='black', lwd=2)
legend("topleft", c("PCA long", "PCA short", "S&P500"), 
       col=c("blue","red","black"), lwd=2)
axis(side=1, plot_date , format(plot_date , "%d-%m-%y"), cex.axis=1)

################################################################################
d <- read.csv("../Datasets/fin-ratio.csv")  # read in dataset
# standardize the data so the columns of large loadings 
# will not dominate the data
d_scale <- scale(d[1:5])
pca_svd <- prcomp(d_scale)      # run PCA with SVD
pca_svd$rotation

s_svd <- pca_svd$sdev
cumsum(s_svd^2/sum(s_svd^2))

pca_sd <- princomp(d_scale)     # run PCA with Spectral Decomposition        
pca_sd$loadings

s_sd <- pca_sd$sdev
cumsum(s_sd^2/sum(s_sd^2))

################################################################################
d_scale <- t(scale(d[1:5]))          
pca_svd <- prcomp(d_scale)      # run PCA with SVD
s_svd <- pca_svd$sdev
cumsum(s_svd^2/sum(s_svd^2))

pca_sd <- princomp(d_scale)     # run PCA with Spectral Decomposition
pca_sd$loadings

################################################################################
pca_sd <- pr_comp(d[1:5], method="SD")
pca_svd <- pr_comp(d[1:5], method="SVD")

pca_sd <- pr_comp(t(d[1:5]), method="SD")
pca_svd <- pr_comp(t(d[1:5]), method="SVD")