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

acf(train_close, main=paste("ACF Plot of Bitcoin Price"))

par(mfrow=c(1, 2))
lag_price <- diff(train_close)
acf(lag_price, main=paste("ACF Plot of 1-lagged Bitcoin Price"))
pacf(lag_price, main=paste("PACF Plot of 1-lagged Bitcoin Price"))

################################################################################
# 1 lag difference; ACF and PACF: 2 significant lags and looks similar
AIC(Arima(ts(train_close), order=c(2, 1, 0)),
    Arima(ts(train_close), order=c(0, 1, 2)))

################################################################################
(best_model <- auto.arima(train_close, max.p=5, max.q=5, max.d=2, 
                          seasonal=FALSE, ic='aic', allowdrift=FALSE))

################################################################################
order <- arimaorder(best_model)
arima_name <- paste0("ARIMA(", paste(order, collapse=", "), ")")

resid <- xts(best_model$residuals, order.by=train_date)
par(mfrow=c(1, 1))
plot(resid/sqrt(best_model$sigma2), type="h", ylab="", 
     main=paste("Standardized Residuals of", arima_name))

# Ljung-Box test on (non-standardized) residuals
Box.test(resid, lag=10, type="Ljung-Box", fitdf=order[1]+order[3])

which(abs(resid)/sqrt(best_model$sigma2) > 12)
# R indexing starts from 1
split_idx <- 115700
addEventLines(events=xts(x='', order.by=index(resid)[split_idx]),
              lty=2, col='orange', lwd=2)
# -1 for d=1
train_date[split_idx-1]

################################################################################
split_train <- train_close[(split_idx-1):length(train_close)]
(split_model <- auto.arima(split_train, max.p=10, max.q=10, max.d=2, 
                           seasonal=FALSE, ic='aic', allowdrift=FALSE))
split_order <- arimaorder(split_model)
split_name <- paste0("ARIMA(", paste(split_order, collapse=", "), ")")
split_date <- train_date[(split_idx-1):length(train_date)]
split_resid <- xts(split_model$residuals, order.by=split_date)
plot(split_resid/sqrt(split_model$sigma2), type="h", ylab="", 
     main=paste("Standardized Residuals of", split_name))
# Ljung-Box test on (non-standardized) residuals
Box.test(split_resid, lag=10, type="Ljung-Box", 
         fitdf=split_order[1]+split_order[3])

################################################################################
# Check against Python
python_order <- c(2, 1, 3)
python_coef <- c(-0.27165673, -0.92135467, 0.24353641, 0.90292576, 
                 -0.03332822)
python_sig2 <- 31.51110009

python_model <- Arima(ts(split_train), order=python_order, fixed=python_coef)
python_model$sigma2 <- python_sig2
# Add 2xm = 2x(2+3) = 10 to AIC, as we fixed them in the first place 
Arima(ts(split_train), model=python_model)    # Python

Arima(ts(split_train), order=python_order)    # R

################################################################################

plot(split_train, type="l", ylab="", 
     main=paste("Price"))

################################################################################
hist_data <- tail(train_close, 30)
arima_pred <- array(NA, dim=length(test_close))
for (i in 1:length(test_close)) {
  # update the model with old estimates of ARIMA parameters
  split_model <- Arima(hist_data, model=split_model)
  arima_pred[i] <- predict(split_model, n.ahead=1)$pred[1]
  hist_data <- c(hist_data, test_close[i])
}

ts_pred <- xts(cbind(test_close, arima_pred), 
               order.by=strptime(test_date, format="%Y-%m-%d %H:%M:%S"))

plot(ts_pred, main="Bitcoin Price Prediction", lwd=c(1, 3), 
     col=c("blue", "pink"))
addLegend("topright", lwd=c(1, 3), ncol=2, bg="white", bty="o",
          legend.names=c("Actual", split_name))

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