library(caret)                      # Library for RMSE and R2

data("Boston", package="MASS")      # Load the data

# Split the data into training and testing dataset in ratio of 8:2
set.seed(123)
training.samples <- sample(1:nrow(Boston), round(0.8*nrow(Boston)))
train.data <- Boston[training.samples,]
test.data <- Boston[-training.samples,]

# Build the model
model <- lm(medv ~ poly(lstat, degree=5, raw=TRUE), data=train.data)
summary(model)

# Make predictions
predictions <- predict(model, test.data)

# Model performance
RMSE(predictions, test.data$medv)
R2(predictions, test.data$medv)

(ggplot(train.data, aes(lstat, medv)) + geom_point()
  + stat_smooth(method=lm, formula=y ~ poly(x, 5, raw=TRUE)))