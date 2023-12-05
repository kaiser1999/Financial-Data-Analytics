#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

SPX <- read.csv("../Datasets/SPX.csv", header=TRUE, row.names=1)
HSI <- read.csv("../Datasets/HSI.csv", header=TRUE, row.names=1)
FTSE <- read.csv("../Datasets/FTSE.csv", header=TRUE, row.names=1)

SPX$Year <- format(as.Date(rownames(SPX), format="%d/%m/%Y"), "%Y")
HSI$Year <- format(as.Date(rownames(HSI), format="%d/%m/%Y"), "%Y")
FTSE$Year <- format(as.Date(rownames(FTSE), format="%d/%m/%Y"), "%Y")

SPX$log_return <- c(NA, diff(log(SPX$Price)))
HSI$log_return <- c(NA, diff(log(HSI$Price)))
FTSE$log_return <- c(NA, diff(log(FTSE$Price)))

# Remove first day of data for log return
SPX <- SPX[-1,]
HSI <- HSI[-1,]
FTSE <- FTSE[-1,]

Remove_Outlier <- function(index, outlier_factor=1.5){
  q25 <- quantile(index$log_return, probs=.25, na.rm=FALSE)
  q75 <- quantile(index$log_return, probs=.75, na.rm=FALSE)
  iqr <- q75 - q25  #Inter-quartile range
  lower_bound <- q25 - outlier_factor*iqr
  upper_bound <- q75 + outlier_factor*iqr
  pos <- index$log_return > lower_bound & index$log_return < upper_bound
  return (index[pos,])
}

# Hyperparameter
outlier_removal <- TRUE
start_year <- 2011
end_year <- 2020

if (outlier_removal){
  SPX <- Remove_Outlier(SPX)
  HSI <- Remove_Outlier(HSI)
  FTSE <- Remove_Outlier(FTSE)
}

year_name <- paste0(" Year: ", start_year, "-", end_year)
Chosen_Year <- start_year:end_year
Chosen_SPX <- SPX[SPX$Year %in% Chosen_Year,]
Chosen_HSI <- HSI[HSI$Year %in% Chosen_Year,]
Chosen_FTSE <- FTSE[FTSE$Year %in% Chosen_Year,]

par(mfrow=c(1,3))
hist(Chosen_SPX$log_return, breaks=20, xlab="SPX", cex.lab=1.5, cex.axis=1.5,
     cex.main=1.5, main=paste0("SPX daily log-return.", year_name))
hist(Chosen_HSI$log_return, breaks=20, xlab="HSI", cex.lab=1.5, cex.axis=1.5,
     cex.main=1.5, main=paste0("HSI daily log-return.", year_name))
hist(Chosen_FTSE$log_return, breaks=20, xlab="FTSE", cex.lab=1.5, cex.axis=1.5,
     cex.main=1.5, main=paste0("FTSE daily log-return.", year_name))

u1 <- Chosen_SPX$log_return; u2 <- Chosen_HSI$log_return
u3 <- Chosen_FTSE$log_return
ks.test(u1, y=pnorm, mean=mean(u1), sd=sd(u1))
ks.test(u2, y=pnorm, mean=mean(u2), sd=sd(u2))
ks.test(u3, y=pnorm, mean=mean(u3), sd=sd(u3))

par(mfrow=c(1,3))
boxplot(Chosen_SPX$log_return, Chosen_HSI$log_return, Chosen_FTSE$log_return, 
        names=c("SPX", "HSI", "FTSE"), xlab="Index", ylab="Daily log-return",
        frame=FALSE, col=c("#00AFBB", "#E7B800", "#FC4E07"), boxwex=0.75,
        main=paste0("Boxplot.", year_name))

Chosen_SPX$Index <- "SPX"
Chosen_HSI$Index <- "HSI"
Chosen_FTSE$Index <- "FTSE"
AllIndex <- rbind(Chosen_SPX, Chosen_HSI, Chosen_FTSE)
AllIndex$Index <- factor(AllIndex$Index, c("SPX", "HSI", "FTSE"))

library("gplots")
plotmeans(log_return~Index, data=AllIndex, xlab="Index", ylab="Daily log-return",
          main=paste0("Mean Plot with 95% CI.", year_name))
anova.test <- aov(log_return~Index, data=AllIndex)
summary(anova.test)
# Reject the null hypothesis?
summary(anova.test)[[1]][[1, "Pr(>F)"]] < 0.05

tukey.test <- TukeyHSD(anova.test)
tukey.test
plot(tukey.test)

mu_SPX <- mean(Chosen_SPX$log_return)
mu_HSI <- mean(Chosen_HSI$log_return)
mu_FTSE <- mean(Chosen_FTSE$log_return)

SSE_SPX <- sum((Chosen_SPX$log_return - mu_SPX)^2)
SSE_HSI <- sum((Chosen_HSI$log_return - mu_HSI)^2)
SSE_FTSE <- sum((Chosen_FTSE$log_return - mu_FTSE)^2)

n_SPX <- length(Chosen_SPX$log_return)
n_HSI <- length(Chosen_HSI$log_return)
n_FTSE <- length(Chosen_FTSE$log_return)

df <- length(AllIndex$log_return) - 3
df

Within_group_MSE <- (SSE_SPX + SSE_HSI + SSE_FTSE) / df
Within_group_MSE

# Harmonic mean used for the two sample sizes
SPX_HSI_SE_ANOVA <- sqrt(Within_group_MSE / (2/(1/n_SPX + 1/n_HSI)))
FTSE_SPX_SE_ANOVA <- sqrt(Within_group_MSE / (2/(1/n_FTSE + 1/n_SPX)))
FTSE_HSI_SE_ANOVA <- sqrt(Within_group_MSE / (2/(1/n_FTSE + 1/n_HSI)))

SPX_vs_HSI <- abs(mu_SPX - mu_HSI) / SPX_HSI_SE_ANOVA
FTSE_vs_SPX <- abs(mu_FTSE - mu_SPX) / FTSE_SPX_SE_ANOVA
FTSE_vs_HSI <- abs(mu_FTSE - mu_HSI) / FTSE_HSI_SE_ANOVA

1 - ptukey(q=SPX_vs_HSI, nmeans=3, df=df)
1 - ptukey(q=FTSE_vs_SPX, nmeans=3, df=df)
1 - ptukey(q=FTSE_vs_HSI, nmeans=3, df=df)