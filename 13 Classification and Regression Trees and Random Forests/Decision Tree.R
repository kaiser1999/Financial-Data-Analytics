setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(rpart)                          # load rpart library
library(rpart.plot)                     # plot rpart object

################################################################################
### Financial ratio ###
################################################################################
df <- read.csv("fin-ratio.csv")         # read in data in csv format
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
### Titanic survival ###
################################################################################
data("ptitanic")            # load the built-in titanic dataset in rpart.plot
df <- ptitanic
names(df)					          # display variables
dim(df)					            # display dimension
N <- nrow(df)		            # no. of obs
N_train <- floor(N * 0.8)

set.seed(4012)				                  # set random seed
train_idx <- sample(1:N, size=N_train)  # generate idx for train-test split
df_train <- df[train_idx,]		          # training dataset
df_test <- df[-train_idx,]		          # testing dataset

ctree <- rpart(survived~pclass+sex+age, data=df_train, method="class")
print(ctree)                            # print detailed information
rpart.rules(ctree, nn=TRUE)             # print classification rules
rpart.plot(ctree, extra=1, cex=1.5, digits=4, nn=TRUE) # plot ctree

################################################################################
prob <- predict(ctree, newdata=df_test) # 2 columns of prob. for testing dataset
y_hat <- colnames(prob)[max.col(prob)]
table(y_hat, df_test$survived)	        # confusion matrix

################################################################################
### Regression Tree ###
################################################################################
data("Boston", package="MASS")          # Load the data

set.seed(4002)
training.samples <- sample(1:nrow(Boston), round(0.8*nrow(Boston)))
train.data <- Boston[training.samples,]
test.data <- Boston[-training.samples,]

rtree <- rpart(medv~., data=train.data, method="anova")
print(rtree)
rpart.rules(rtree, nn=TRUE)             # print classification rules
rpart.plot(rtree, extra=1, cex=1.5, digits=4, nn=TRUE) # plot rtree

n_col <- 20
y <- train.data$medv
y_scale <- round((y - min(y)) / diff(range(y)) * (n_col-1)) + 1
plot(train.data$lstat, train.data$rm, pch=20, col=rainbow(n_col)[y_scale])
abline(h=7.437)		                      # add a horizontal line
abline(h=6.657)		                      # add a horizontal line
abline(v=9.645)		                      # add a vertical line
abline(v=16.04)		                      # add a vertical line

lgd_ <- rep(NA, n_col)
lgd_[c(1, n_col%/%2, n_col)] <- c(min(y), diff(range(y))/2, max(y))
legend(x=30, y=8.7, legend=lgd_, fill=rainbow(n_col), border="white",
       y.intersp=0.5, cex=1.5, bty="n")

y_hat <- predict(rtree, newdata=test.data)
mean((test.data$medv - y_hat)^2)

################################################################################
### Random Forest ###
################################################################################
library(randomForest)

set.seed(4002)
df <- read.csv("fin-ratio.csv")         # read in data in csv format
df$HSI <- as.factor(df$HSI) # change label into factor for classification
rf_clf <- randomForest(HSI~., data=df, ntree=10, mtry=2, importance=TRUE)
y_hat <- predict(rf_clf)
table(y_hat, df$HSI)