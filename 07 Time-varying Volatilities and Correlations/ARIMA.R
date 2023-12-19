#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################

library("tseries")
library("xts")
library("forecast")

df <- read.csv("../Datasets/Bitstamp_BTCUSD_2018_minute.csv", skip=1)

# Split prices into train (last quarter) and test (last day)
train_idx <- "2018-10-01" < df$date & df$date < "2018-12-31"
train_close <- rev(df$close[train_idx])
train_date <- rev(df$date[train_idx])
test_idx <- df$date > "2018-12-31"
test_close <- rev(df$close[test_idx])
test_date <- rev(df$date[test_idx])
c(train_date[1], tail(train_date, 1), test_date[1])

acf(train_close, main=paste("ACF Plot of Bitcoin Price"))

par(mfrow=c(1, 2))
lag_price <- diff(train_close)
acf(lag_price, main=paste("ACF Plot of 1-lagged Bitcoin Price"))
pacf(lag_price, main=paste("PACF Plot of 1-lagged Bitcoin Price"))

################################################################################
auto.arima(train_close, max.p=5, max.q=5, max.d=2, 
           seasonal=FALSE, ic='aic')

################################################################################
# 1 lag difference; ACF and PACF: 2 significant lags and looks similar
p <- 2; d <- 1; q <- 0
arima_name <- paste0("ARIMA(", paste(p, d, q, sep=", "), ")")

model <- Arima(ts(train_close), order=c(p, d, q), seasonal=FALSE)
round(coef(model), 6)
resid <- xts(model$residuals, 
             order.by=strptime(train_date, format="%Y-%m-%d %H:%M:%S"))
par(mfrow=c(1, 1))
plot(resid/sqrt(model$sigma2), type="h", ylab="", 
     main=paste("Standardized Residuals of", arima_name))

Box.test(resid, lag=10, type="Ljung-Box", fitdf=p+q)

################################################################################
hist_data <- tail(train_close, 30)
arima_pred <- array(NA, dim=length(test_close))
for (i in 1:length(test_close)) {
  # update the model with old estimates of ARIMA parameters
  model <- Arima(hist_data, model=model)
  arima_pred[i] <- predict(model, n.ahead=1)$pred[1]
  hist_data <- c(hist_data, test_close[i])
}

ts_pred <- xts(cbind(test_close, arima_pred), 
               order.by=strptime(test_date, format="%Y-%m-%d %H:%M:%S"))

plot.xts(ts_pred, main="Bitcoin Price Prediction",
         lwd=c(1, 3), col=c("blue", "pink"))
addLegend("topright", lwd=c(1, 3), ncol=2, bg="white", bty="o",
          legend.names=c("Actual", arima_name))

# using previous price to predict the next day price
mean(diff(c(tail(train_close, 1), ts_pred$test_close))**2, na.rm=TRUE)
mean((ts_pred$test_close - ts_pred$arima_pred)**2)