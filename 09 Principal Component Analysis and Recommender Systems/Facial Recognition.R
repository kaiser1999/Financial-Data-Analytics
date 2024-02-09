#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(pixmap)

SUB_FOLDERS <- paste0("s", 1:40)
train_img <- c()
test_img <- c()

for (sub_folder in SUB_FOLDERS) {
  path <- file.path("../Datasets/ATT", sub_folder)
  files <- list.files(path)
  for (i in 1:length(files)) {
    img_array <- read.pnm(file=file.path(path, files[i]))
    # matching python arrays
    rescale_array <- as.vector(t(img_array@grey) * 255)
    
    if (i == 1) {
      test_img <- cbind(test_img, rescale_array)
    } else {
      train_img <- cbind(train_img, rescale_array)
    }
  }
}

n <- ncol(train_img)
mu_train <- apply(train_img, 1, mean)
std_train <- apply(train_img, 1, sd) * sqrt((n-1)/n)
rescale_train <- (train_img - mu_train) / std_train
rescale_test <- (test_img - mu_train) / std_train

################################################################################
# method 1: using eigen
eig <- eigen(t(rescale_train) %*% rescale_train / n)
lamb <- eig$values; H <- t(eig$vectors) %*% t(rescale_train)
H <- H / sqrt(rowSums(H^2))     # normalize each eigenvector
dim(H)                          # a n x (hw) matrix
sum(lamb)

CumSum_s2 <- cumsum(lamb) / sum(lamb)
idx <- which(CumSum_s2 > 0.9)[1]
tilde_H <- H[1:idx,]
dim(tilde_H)                    # m x (hw) matrix

################################################################################
# method 2: using prcomp with the already normalized training faces
pca <- prcomp(t(rescale_train), center=FALSE, scale=FALSE)
idx <- which(cumsum(pca$sdev^2)/sum(pca$sdev^2) > 0.9)[1]
tilde_H <- t(pca$rotation[,1:idx])
dim(tilde_H)                    # m x (hw) matrix

################################################################################
score_train <- tilde_H %*% rescale_train    # m x n matrix
score_test <- tilde_H %*% rescale_test      # m x n matrix
pred_dist <- c()
n_test <- ncol(score_test)

img_arr <- function(img){
  t(apply(matrix(img, nrow=112, ncol=92, byrow=TRUE), 2, rev))
}

nrow <- 8; ncol <- 10
opar <- par()
par(mfrow=c(nrow, ncol), plt=c(0.05,0.95,0,0.7), oma=c(1,1,1,1))
for (i in 1:n_test) {
  distance <- sqrt(colSums((score_train - score_test[,i])^2))
  idx_train <- which.min(distance)
  
  image(img_arr(test_img[,i]), 
        col=grey(seq(0, 1, length=256)), xaxt='n', yaxt='n')
  title("Test image", font.main=1 , line=1)
  image(img_arr(train_img[,idx_train]), 
        col=grey(seq(0, 1, length=256)), xaxt='n', yaxt='n')
  title(paste("Distance", round(distance[idx_train], 3)), 
        font.main=1, line=1)
  pred_dist <- c(pred_dist, distance[idx_train])
}

################################################################################
par(opar)
col <- rep("blue", n_test)
col[35] <- "red"
barplot(pred_dist, names=1:n_test, col=col, ylim=c(0, 80),
        xlab="Index", ylab="Distance")
abline(h=65, lty=2)