set.seed(4002)
AR_2 <- arima.sim(n=200, list(ar=c(0.2897, 0.4858)), sd=1)
set.seed(4002)
MA_2 <- arima.sim(n=200, list(ma=c(0.1897, 0.4858)), sd=1)

par(mfrow=c(1,2))
acf(AR_2); pacf(AR_2)
acf(MA_2); pacf(MA_2)