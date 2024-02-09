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
train_date <- strptime(train_date, format="%Y-%m-%d %H:%M:%S")

par(mfrow=c(1, 1))
plot(xts(train_close, order.by=train_date), type="l", 
     ylab="", main="Bitcoin Price")

win_size <- 30
s_win <- xts(rollapply(tail(train_close, 5000), win_size, sd), 
             order.by=tail(train_date, 5000-win_size+1))
plot(s_win, type="l", ylab="", 
     main=paste0(win_size, "-minute simple moving standard deviation"))

# length(s_win) + win_size - 1 = 5000
(split_idx <- tail(which(s_win > 50), 1) + 10)  # add 10 to adjust
addEventLines(events=xts(x='', order.by=index(s_win)[split_idx]),
              lty=2, col='orange', lwd=2, on=0)

split_train <- tail(train_close, 5000-(split_idx+win_size-1))
tail(train_date, 5000-win_size+1-split_idx)[1]
length(split_train)

################################################################################
acf(split_train, main=paste("ACF Plot of Bitcoin Price"))

par(mfrow=c(1, 2))
lag_price <- diff(split_train)
acf(lag_price, main=paste("ACF Plot of 1-lagged Bitcoin Price"))
pacf(lag_price, main=paste("PACF Plot of 1-lagged Bitcoin Price"))

# 1 lag difference; ACF and PACF: 2 significant lags and looks similar
AIC(Arima(ts(split_train), order=c(2, 1, 0)),
    Arima(ts(split_train), order=c(0, 1, 2)),
    Arima(ts(split_train), order=c(2, 1, 2)))

################################################################################
(best_model <- auto.arima(split_train, max.p=5, max.q=5, max.d=2, 
                          seasonal=FALSE, ic='aic', allowdrift=FALSE))

################################################################################
order <- arimaorder(best_model)
arima_name <- paste0("ARIMA(", paste(order, collapse=", "), ")")

resid <- xts(best_model$residuals, 
             order.by=tail(train_date, 5000-win_size+1-split_idx))
par(mfrow=c(1, 1))
plot(resid, main=paste("Fitted residuals of", arima_name), 
     type="h", ylab="")

# Ljung-Box test on (non-standardized) residuals
Box.test(resid, lag=10, type="Ljung-Box", fitdf=order[1]+order[3])

################################################################################
hist_data <- tail(split_train, 30)
arima_pred <- array(NA, dim=length(test_close))
for (i in 1:length(test_close)) {
  # update the model with old estimates of ARIMA parameters
  best_model <- Arima(hist_data, model=best_model)
  arima_pred[i] <- predict(best_model, n.ahead=1)$pred[1]
  hist_data <- c(hist_data, test_close[i])
}

ts_pred <- xts(cbind(test_close, arima_pred), 
               order.by=strptime(test_date, format="%Y-%m-%d %H:%M:%S"))

plot(ts_pred, main="Bitcoin Price Prediction", lwd=c(1, 3), 
     col=c("blue", "pink"))
addLegend("topright", lwd=c(1, 3), ncol=2, bg="white", bty="o",
          legend.names=c("Actual", arima_name))

# using previous price to predict the next day price
mean(diff(c(tail(train_close, 1), ts_pred$test_close))**2, na.rm=TRUE)
mean((ts_pred$test_close - ts_pred$arima_pred)**2)

################################################################################
### ARCH-GARCH
################################################################################
library(rugarch)

spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1, 1)), 
                   mean.model=list(armaOrder=c(4, 3), include.mean=FALSE), 
                   distribution.model="norm")
(arch_garch <- ugarchfit(spec=spec, data=diff(split_train)))
# Ljung-Box test on standardized residuals
Box.test(arch_garch@fit$residuals/arch_garch@fit$sigma, lag=20, 
         type="Ljung-Box", fitdf=4+3)

(best_arfima <- autoarfima(data=diff(split_train), ar.max=5, ma.max=5, 
                           criterion="AIC", method="full", arfima=FALSE))