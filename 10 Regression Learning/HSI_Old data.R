#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
d <- read.csv("../Datasets/fin-ratio.csv")
summary(lm(HSI~EY+CFTP+ln_MV+DY+BTME+DTE, data=d))
summary(lm(HSI~EY+CFTP+ln_MV+DY+BTME, data=d))

################################################################################
summary(lm(HSI~CFTP+ln_MV, data=d))

################################################################################
reg <- lm(HSI~CFTP+ln_MV, data=d)	    # save regression results
names(reg)

par(mfrow=c(3,2))                     # set a 3x2 graphical frame
hist(reg$fitted.values)               # fitted values histogram
hist(reg$residuals)                   # residuals histogram
plot(reg$fitted.values, reg$residuals)# residuals vs fitted values
qqnorm(reg$residuals)                 # qq-normal plot of residuals
qqline(reg$residuals)                 # add reference line
res <- as.ts(reg$residuals)           # change res to time series
plot(res, lag(res))                   # residuals vs lag(residuals)
plot(reg$residuals)                   # residuals vs index number

################################################################################
### Logistic Regression ###
################################################################################
summary(glm(HSI~EY+CFTP+ln_MV+DY+BTME+DTE, data=d, family=binomial))

lreg <- glm(HSI~EY+CFTP+ln_MV+DY+BTME+DTE, data=d, family=binomial)
pr <- (lreg$fitted.values > 0.5)      # pr=TRUE if fitted > 0.5
table(pr, d$HSI)

################################################################################
### Outlier Detection ###
################################################################################
par(mfrow=c(1,1))
################################################################################

mdist <- function(x) {
  t <- as.matrix(x)                   # transform x to a matrix
  m <- apply(t, 2, mean)              # compute column mean
  s <- var(t)                         # compute sample cov. matrix
  mahalanobis(t, m, s)                # built-in mahalanobis func.
}

################################################################################
d <- read.csv("../Datasets/fin-ratio.csv")        # read in dataset
d0 <- d[d$HSI==0,]                    # select HSI=0
d1 <- d[d$HSI==1,]                    # select HSI=1
dim(d0)

################################################################################
x <- d0[,1:6]                         # save d0 to x
md <- mdist(x)                        # compute mdist
plot(md)

################################################################################
(c <- qchisq(0.99, df=6))             # p=6, type-I error = 0.01

d2 <- d0[md<c,]                       # select case in d0 with md<c
dim(d2)                               # throw away 648-626=22 cases
d3 <- rbind(d1, d2)                   # combine d1 with d2 
dim(d3)
# save the cleansed dataset 
write.csv(d3, file="fin-ratio_cleaned.csv", row.names=FALSE)

################################################################################
summary(glm(HSI~CFTP+ln_MV+BTME, data=d3, family=binomial))

lreg <- glm(HSI~CFTP+ln_MV+BTME, data=d3, family=binomial)
pr <- (lreg$fitted.values > 0.5)      # pr=TRUE if fitted > 0.5
table(pr, d3$HSI)

################################################################################
### Threshold Regression ###
################################################################################
# create dummy variable g=2 if ln_MV>9.4776 and g=1 otherwise
g <- (d3$ln_MV > 9.4776) + 1
summary(glm(HSI~EY+CFTP+DY+BTME+DTE+g+EY*g+CFTP*g+DY*g+BTME*g+DTE*g, 
            data=d3, binomial))

################################################################################
summary(glm(HSI~EY+DY+BTME+DTE+g+EY*g+DY*g+BTME*g+DTE*g,
            data=d3, binomial))

################################################################################
summary(glm(HSI~DY+g+DY*g, data=d3, binomial))

threshold_lreg <- glm(HSI~DY+g+DY*g, data=d3, binomial)
pr <- (threshold_lreg$fit>0.5)
table(pr, d3$HSI)

################################################################################
### Multinomial Regression ###
################################################################################
library(nnet)                         # load nnet

names(iris)
mnl <- multinom(Species~., data=iris) # perform MNL
summary(mnl)                          # MNL summary

pred <- predict(mnl)                  # prediction
table(pred, iris$Species)             # tabulate results

################################################################################
d3 <- read.csv("fin-ratio_cleaned.csv")		    
lreg <- glm(HSI~., data=d3, binomial) # save the logistic reg
step(lreg)                            # perform stepwise selection

################################################################################
### Life Chart ###
################################################################################
ysort <- d3$HSI[order(lreg$fit, decreasing=T)] # sort y descending
# ideal case of y
yideal <- c(rep(1, sum(d3$HSI)), rep(0,length(d3$HSI)-sum(d3$HSI)))

n <- length(ysort)                    # length of ysort
perc1 <- cumsum(ysort)/(1:n)          # cum. percentage
plot(perc1, type="l", col="blue")     # plot percentage
abline(h=sum(d3$HSI)/n)               # add horizontal baseline
perc_ideal <- cumsum(yideal)/(1:n)    # ideal cum. percentage 
lines(perc_ideal, type="l", col="red")# plot ideal case

################################################################################
perc2 <- cumsum(ysort)/sum(ysort)     # cum. percentage
pop <- (1:n)/n                        # x-coordinate
plot(pop, perc2, type="l")            # plot percentage
lines(pop, pop)                       # add reference line
# cumulative perc. of success for ideal case
perc2_ideal <- cumsum(yideal)/sum(yideal)
lines(pop, perc2_ideal, type="l", col="red")# plot ideal case