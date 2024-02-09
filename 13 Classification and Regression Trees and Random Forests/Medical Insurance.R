#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

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