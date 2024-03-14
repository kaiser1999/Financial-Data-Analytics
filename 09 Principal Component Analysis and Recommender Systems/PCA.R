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
# compute the covariance matrix and extract the variance
var <- diag(cov(d))
sqrt(var)

################################################################################
df <- read.csv("../Datasets/us_Open_2020H.csv", header=TRUE, row.names=1)
# Compute daily logarithmic return for each equity
r <- sapply(df, function(x) diff(log(x)))
date <- as.Date(rownames(df), format="%d/%m/%Y")
r_idx_train <- which(date[-1] < as.Date("2020/04/01"))
r_train <- r[r_idx_train,]

PCA <- prcomp(r_train, center=TRUE, scale=FALSE)
pc1 <- PCA$rotation[,1]        # Save the loading of 1st PC

# Select 10 equities to invest base on the loadings of pc1
n_stock <- 10
smallest_n <- order(pc1, decreasing=FALSE)[1:n_stock]
largest_n <- order(pc1, decreasing=TRUE)[1:n_stock]

# check which group performs the best with training data
df_train_last <- tail(r_idx_train, 1) + 1     # add back 1 in r for d
df[df_train_last, smallest_n] - df[1, smallest_n]
df[df_train_last, largest_n] - df[1, largest_n]

################################################################################
# commission and short-selling fees formulas
# https://www.futuhk.com/en/support/topic2_417?
library(zoo)

investment <- 100000
buy_and_hold <- function(test_price, test_date, n_stock=1){
  n_units <- floor(investment/n_stock/test_price[1,])
  commission <- sum(sapply(n_units, function(x) max(2.05, 0.013*x)))
  stocks_amount <- as.numeric(test_price[1,] %*% n_units)
  price_path <- test_price %*% n_units
  remain <- investment - 2*commission
  values <- zoo(remain + (price_path - stocks_amount))
  index(values) <- test_date
  return (values)
}

spx <- read.csv("../Datasets/SPX_Open_2020H.csv", header=TRUE, row.names=1)
equity_price <- as.matrix(df[(df_train_last+1):nrow(df),])
index_price <- as.matrix(spx[(df_train_last+1):nrow(df),])
test_date <- date[(df_train_last+1):nrow(df)]

pca_bnh_smallest <- buy_and_hold(equity_price[,smallest_n], test_date, 
                                 n_stock)
pca_bnh_largest <- buy_and_hold(equity_price[,largest_n], test_date, 
                                n_stock)
market_bnh <- buy_and_hold(index_price, test_date, 1)

par(mfrow=c(1,1))
plot_date <- test_date[seq(1, length(test_date), 10)]
plot(pca_bnh_smallest, ylim=c(0.9*investment, 3*investment),
     col='red', lwd=2, xaxt="n", ylab="")
lines(pca_bnh_largest, col='blue', lwd=2)
lines(market_bnh, col='black', lwd=2)
legend("topleft", c("PCA smallest", "PCA largest", "S&P500"), 
       col=c("red","blue","black"), lwd=2)
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