#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(rpart)                          # load rpart library
library(rpart.plot)                     # plot rpart object

################################################################################
### Financial ratio ###
################################################################################
df <- read.csv("../Datasets/fin-ratio.csv")         # read in data in csv format
ctree <- rpart(HSI~., data=df, method="class")
print(ctree)                            # print detailed information
rpart.rules(ctree, nn=TRUE)             # print classification rules
rpart.plot(ctree, extra=1, cex=1.5, digits=4, nn=TRUE) # plot ctree

################################################################################
plot(df$HSI, df$ln_MV, pch=21, bg=c("red","blue")[df$HSI+1])
legend(0, 14, legend=c("Non-HSI", "HSI"), pch=16, col=c("red","blue"))
abline(h=9.478)		# add a horizontal line at y=9.478

################################################################################
prob <- predict(ctree)      # 2 columns of probabilities for 0 or 1
y_hat <- colnames(prob)[max.col(prob)]
table(y_hat, df$HSI)		                # confusion matrix

################################################################################
### IRIS flower ###
################################################################################
data("iris")                # load the built-in iris flower dataset
df <- iris
ctree <- rpart(Species~., data=df, method="class")
print(ctree)                            # print detailed information
rpart.rules(ctree, nn=TRUE)             # print classification rules
rpart.plot(ctree, extra=1, cex=1.5, digits=4, nn=TRUE) # plot ctree

################################################################################
# plot Petal.Width versus Petal.Length with different color for each species
plot(df$Petal.Length, df$Petal.Width, pch=21, 
     bg=c("red","blue","green")[df$Species])
legend(1, 2.5, legend=unique(df$Species), pch=16, 
       col=c("red", "blue", "green"))
abline(h=1.75)		                      # add a horizontal line
abline(v=2.45)		                      # add a vertical line

################################################################################
prob <- predict(ctree)  # 3 columns of probabilities for 3 species
y_hat <- colnames(prob)[max.col(prob)]
table(y_hat, df$Species)                # confusion matrix

################################################################################
### Regression Tree ###
################################################################################
library(rpart)                          # load rpart library
library(rpart.plot)                     # plot rpart object

# Prepare the dataset
df <- read.csv("../Datasets/Medicalpremium.csv")

# Fit and plot the regression tree model
rtree <- rpart(PremiumPrice~Age+Weight, data=df, method="anova")
rpart.rules(rtree, nn=TRUE)
rpart.plot(rtree, extra=0, cex=1, digits=2, type=0, nn=TRUE)

# Plot dataset with segments and text annotations
layout(t(1:2), widths=c(6,1))
par(mar=c(4,4,1,1))
n_col <- 20
y <- df$PremiumPrice
y_scale <- round((y - min(y))/diff(range(y))*(n_col-1)) + 1
plot(df$Age, df$Weight, pch=20, xlab="Age", ylab="Weight", 
     col=rainbow(n_col)[y_scale])
abline(v=30, lwd=2)
abline(v=47, lwd=2)
segments(47, 70, 70, 70, lwd=2)
segments(47, 95, 70, 95, lwd=2)
text(22, 90, "R1", cex=1.5)
text(38, 90, "R2", cex=1.5)
text(58, 60, "R3", cex=1.5)
text(58, 82, "R4", cex=1.5)
text(58, 110, "R5", cex=1.5)
image(y=1:n_col, z=t(1:n_col), col=rainbow(n_col), axes=FALSE, 
      main="Premium", cex.main=.8, ylab="")

################################################################################
### Random Forest ###
################################################################################
library(randomForest)

set.seed(4002)
df <- read.csv("../Datasets/fin-ratio.csv")         # read in data in csv format
df$HSI <- as.factor(df$HSI) # change label into factor for classification
rf_clf <- randomForest(HSI~., data=df, ntree=10, mtry=2, importance=TRUE)
y_hat <- predict(rf_clf, newdata=df)
table(y_hat, df$HSI)

################################################################################
library("caret")                        # confusionMatrix

set.seed(4002)				                  # set random seed
df <- read.csv("../Datasets/credit default.csv")    # read in data in csv format
df$default.payment.next.month <- as.factor(
  df$default.payment.next.month
)                       # change label into factor for classification

train_idx <- sample(1:nrow(df), size=floor(nrow(df)*0.8))
df_train <- df[train_idx,]		          # training dataset
df_test <- df[-train_idx,]		          # testing dataset

ctree <- rpart(default.payment.next.month~., data=df_train, 
               method="class")
rpart.plot(ctree, extra=1, cex=1.5, digits=4, nn=TRUE) # plot ctree
prob <- predict(ctree, newdata=df_test)
y_hat_dt <- colnames(prob)[max.col(prob)]

# confusionMatrix(y_test, y_true, ...)
dt_result <- confusionMatrix(as.factor(y_hat_dt), 
                             df_test$default.payment.next.month, 
                             mode="prec_recall", positive="1")
dt_result$table         # Confusion matrix
dt_result$byClass[c("Precision", "Recall")]

rf_clf <- randomForest(default.payment.next.month~., data=df_train, 
                       ntree=10, importance=TRUE)
y_hat_rf <- predict(rf_clf, newdata=df_test)
rf_result <- confusionMatrix(as.factor(y_hat_rf), 
                             df_test$default.payment.next.month, 
                             mode="prec_recall", positive="1")
rf_result$table            # Confusion matrix
rf_result$byClass[c("Precision", "Recall")]