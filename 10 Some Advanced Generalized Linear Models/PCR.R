data("Boston", package="MASS")      # Load the data

# Split the data into training and testing dataset in ratio of 8:2
set.seed(4002)
training.samples <- sample(1:nrow(Boston), round(0.8*nrow(Boston)))
train.data <- Boston[training.samples,]
test.data <- Boston[-training.samples,]

# Principal Component Regression model
library(pls)
pcr_model <- pcr(medv~., data=train.data, scale=T, center=T, 
                 validation="none")

RMSEP(pcr_model)
MSEP(pcr_model)
R2(pcr_model)

par(mfrow=c(1,3), mar=c(4,4,2,2))
validationplot(pcr_model, val.type="RMSE")
validationplot(pcr_model, val.type="MSE")
validationplot(pcr_model, val.type="R2")

# Make prediction with 5 principal components
y_hat_pcr <- predict(pcr_model, test.data, ncomp=5)

# Linear regression model
lr_model <- lm(medv~., data=train.data)
y_hat_lr <- predict(lr_model, test.data)

# Model comparison with RMSE
sqrt(mean((test.data$medv - y_hat_pcr)^2))
sqrt(mean((test.data$medv - y_hat_lr)^2))

################################################################################
# threshold model for dummay variables chas
table(train.data$chas)

train.data_0 <- train.data[train.data$chas==0, -which(names(Boston) %in% c("chas"))]
train.data_1 <- train.data[train.data$chas==1, -which(names(Boston) %in% c("chas"))]

test.data_0 <- test.data[test.data$chas==0, -which(names(Boston) %in% c("chas"))]
test.data_1 <- test.data[test.data$chas==1, -which(names(Boston) %in% c("chas"))]

pcr_model_0 <- pcr(medv~., data=train.data_0, scale=T, center=T, validation="none")
pcr_model_1 <- pcr(medv~., data=train.data_1, scale=T, center=T, validation="none")

par(mfrow=c(1,2), mar=c(4,4,2,2))
validationplot(pcr_model_0, val.type="RMSE")
validationplot(pcr_model_1, val.type="RMSE")

y_hat_pcr_0 <- predict(pcr_model_0, test.data_0, ncomp=4)
y_hat_pcr_1 <- predict(pcr_model_1, test.data_1, ncomp=7)

sqrt(mean(c((test.data_0$medv - y_hat_pcr_0)^2, (test.data_1$medv - y_hat_pcr_1)^2)))

################################################################################

lr_model_0 <- lm(medv~., data=train.data_0)
y_hat_lr_0 <- predict(lr_model_0, test.data_0)

lr_model_1 <- lm(medv~., data=train.data_1)
y_hat_lr_1 <- predict(lr_model_1, test.data_1)

sqrt(mean(c((test.data_0$medv - y_hat_lr_0)^2, (test.data_1$medv - y_hat_lr_1)^2)))