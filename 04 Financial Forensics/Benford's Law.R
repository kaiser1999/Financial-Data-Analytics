library(benford.analysis)     # US Census (2010) dataset
library(stringr)              # Pad leading zero
data(census.2000_2010)        # load census dataset

head(census.2000_2010, 5)     # print the first 5 rows
census_2010 <- census.2000_2010$pop.2010

################################################################################
### Benford with modified Mean Absolute Deviation ###
################################################################################
Benford_Analysis <- function(Count, EP, DIGITS, x_label=""){
  # Fill in missing digits with zero
  if(length(setdiff(DIGITS, as.numeric(names(Count)))) > 0) {
    missing_digit <- setdiff(DIGITS, as.numeric(names(Count)))
    Count[as.character(missing_digit)] <- 0
    Count <- Count[order(as.numeric(names(Count)))]
  }
  
  # Remove any additional digit in Count not in DIGITS
  if(length(setdiff(as.numeric(names(Count)), DIGITS)) > 0) {
    Count <- Count[as.character(DIGITS)]
  }
  
  AP <- as.numeric(Count/sum(Count))
  N <- sum(Count)
  folded_z <- abs(AP - EP)/sqrt(EP*(1-EP)/N)
  p_val <- 2*dnorm(folded_z)    # p-values for folded z-scores
  print(DIGITS[p_val < 0.05])   # Rejected digits by Benford's law
  
  # Build a bar chart for each digit
  col <- rep("burlywood1", length(DIGITS))
  col[p_val < 0.05] <- "red"
  bar_name <- str_pad(DIGITS, nchar(tail(DIGITS, 1)), pad="0")
  census.barplot <- barplot(AP, names.arg=bar_name, col=col, border=NA,
                            ylab="PROPORTION", xlab=x_label)
  lines(census.barplot, EP, col="blue", lwd=2)
  legend("top", ncol=3, c("Actual", "Rejected", "Benford's Law"), 
         lty=c(0,0,1), lwd=c(0,0,2), fill=c("burlywood1", "red", 0), 
         border=NA, col=c(0, 0, "blue"))
  
  # Return the modified Mean Absolute Deviation
  return(mean(folded_z))
}

################################################################################
### Benford with chi-squared goodness-of-fit test ###
################################################################################
Benford_Analysis <- function(Count, EP, DIGITS, x_label=""){
  # Fill in missing digits with zero
  if(length(setdiff(DIGITS, as.numeric(names(Count)))) > 0) {
    missing_digit <- setdiff(DIGITS, as.numeric(names(Count)))
    Count[as.character(missing_digit)] <- 0
    Count <- Count[order(as.numeric(names(Count)))]
  }
  
  # Remove any additional digit in Count not in the group DIGITS
  if(length(setdiff(as.numeric(names(Count)), DIGITS)) > 0) {
    Count <- Count[as.character(DIGITS)]
  }
  
  AP <- as.numeric(Count/sum(Count))
  N <- sum(Count)
  folded_z <- abs(AP - EP)/sqrt(EP*(1-EP)/N)
  p_val <- 2*dnorm(folded_z)    # p-values for folded z-scores
  print(DIGITS[p_val < 0.05])   # Rejected digits by Benford's law
  
  # Build a bar chart for each digit
  col <- rep("burlywood1", length(DIGITS))
  col[p_val < 0.05] <- "red"
  bar_name <- str_pad(DIGITS, nchar(tail(DIGITS, 1)), pad="0")
  census.barplot <- barplot(AP, names.arg=bar_name, col=col, border=NA,
                            ylab="PROPORTION", xlab=x_label)
  lines(census.barplot, EP, col="blue", lwd=2)
  legend("top", ncol=3, c("Actual", "Rejected", "Benford's Law"), 
         lty=c(0,0,1), lwd=c(0,0,2), fill=c("burlywood1", "red", 0), 
         border=NA, col=c(0, 0, "blue"))
  
  # Return the p-value of chi-squared goodness-of-fit test statistics
  return(dchisq(sum(N*(AP - EP)^2/EP), length(DIGITS)-1))
}

################################################################################
FT_1 <- substr(census_2010, 1, 1)       # Get first digit
Count_1 <- table(as.numeric(FT_1))      # A summary table
DIGITS_1 <- 1:9               # Possible first digit values
EP_1 <- log10(1 + 1/DIGITS_1)           # Expected proportion
Benford_Analysis(Count_1, EP_1, DIGITS_1, "FIRST DIGIT")

################################################################################
FT_2 <- substr(census_2010, 2, 2)       # Get second digit
Count_2 <- table(as.numeric(FT_2))      # A summary table
DIGITS_2 <- 0:9               # Possible second digit values
EP_2 <- rep(0, length(DIGITS_2))        # Expected proportion
for (i in 1:9){
  EP_2 <- EP_2 + log10(1 + 1/((10 * i + 0):(10 * i + 9)))
}
Benford_Analysis(Count_2, EP_2, DIGITS_2, "SECOND DIGIT")

################################################################################
FT_3 <- substr(census_2010, 1, 2)       # Get first-two digits
Count_3 <- table(as.numeric(FT_3))      # A summary table 
DIGITS_3 <- 10:99             # Possible first-two digits values
EP_3 <- log10(1 + 1/DIGITS_3)           # Expected proportion
Benford_Analysis(Count_3, EP_3, DIGITS_3, "FIRST-TWO DIGITS")

################################################################################
FT_4 <- substr(census_2010, 1, 3)       # Get first-three digits
Count_4 <- table(as.numeric(FT_4))      # A summary table 
DIGITS_4 <- 100:999           # Possible first-three digits values
EP_4 <- log10(1 + 1/DIGITS_4)           # Expected proportion
Benford_Analysis(Count_4, EP_4, DIGITS_4, "FIRST-THREE DIGITS")

################################################################################
table(nchar(census_2010))               # Table for digit length
census_4_digit <- census_2010[nchar(census_2010) >= 4]
FT_5 <- substr(census_4_digit, nchar(census_4_digit)-2+1, 
               nchar(census_4_digit))   # Get last-two digits
Count_5 <- table(as.numeric(FT_5))      # A summary table 
DIGITS_5 <- 0:99              # Possible last-two digits values
EP_5 <- rep(1/length(DIGITS_5), length(DIGITS_5)) # EP
Benford_Analysis(Count_5, EP_5, DIGITS_5, "LAST-TWO DIGITS")

################################################################################
EP_6 <- rep(0, 10)
for (i in 10:99){
  EP_6 <- EP_6 + log10(1 + 1/((i * 10 + 0):(i * 10 + 9)))
}

EP_7 <- rep(0, 10)
for (i in 100:999){
  EP_7 <- EP_7 + log10(1 + 1/((i * 10 + 0):(i * 10 + 9)))
}

EP_8 <- rep(0, 10)
for (i in 1000:9999){
  EP_8 <- EP_8 + log10(1 + 1/((i * 10 + 0):(i * 10 + 9)))
}

EP_9 <- rep(0, 10)
for (i in 10000:99999){
  EP_9 <- EP_9 + log10(1 + 1/((i * 10 + 0):(i * 10 + 9)))
}

EP_10 <- rep(0, 10)
for (i in 100000:999999){
  EP_10 <- EP_10 + log10(1 + 1/((i * 10 + 0):(i * 10 + 9)))
}

census.barplot <- barplot(EP_1, col="burlywood1", xpd=F,
                          border="white", ylim=c(0, 0.35), ylab="PROPORTION",
                          xlab="FIRST DIGIT")

census.barplot <- barplot(EP_2, col="coral1", xpd=F,
                          border="white", ylim=c(0, 0.14), ylab="PROPORTION",
                          xlab="SECOND DIGIT")

census.barplot <- barplot(EP_6, col="cadetblue2", xpd=F,
                          border="white", ylim=c(0.09, 0.105), ylab="PROPORTION",
                          xlab="THIRD DIGIT")


EP_78910 <- rbind(EP_7, EP_8, EP_9, EP_10)
rownames(EP_78910) <- c("Forth Digit", "Fifth Digit", "SIXTH Digit", "SEVENTH DIGIT")
colnames(EP_78910) <- 0:9
census.barplot <- barplot(EP_78910, col=c("burlywood1", "coral1", "cadetblue2", "cyan3"), beside=T, xpd=F,
                          border="white", ylim=c(0.099, 0.1005), ylab="PROPORTION",
                          xlab="FOURTH, FIFTH, SIXTH, and SEVENTH DIGITS")
legend("top", ncol=4, c("Forth Digit", "Fifth Digit", "Sixth Digit", "Seventh Digit"), 
       lty=0, lwd=0, fill=c("burlywood1", "coral1", "cadetblue2", "cyan3"), border=NA)


################################################################################
n <- 100000
set.seed(4012)
DIGITS <- DIGITS_4
EP <- log10(1 + 1/DIGITS)

Cov <- diag(EP) - EP %*% t(EP)
inv_Var <- solve(sqrt(diag(EP * (1 - EP))))
Cor <- inv_Var %*% Cov %*% inv_Var
Z <- matrix(rnorm(n*(length(DIGITS))), nrow=n)
X <- Z %*% chol(Cor)
Y <- abs(X)
mMAD <- apply(Y, 1, mean)
#plot(density(MAD))

quantile(mMAD, c(1-0.95, 1-0.97, 1-0.99))
quantile(mMAD, c(0.8, 0.9, 0.95, 0.99))

################################################################################
n <- 10000
set.seed(4012)
DIGITS <- DIGITS_4
EP <- log10(1 + 1/DIGITS)
N <- 10000

mMAD <- rep(0, n)
for (i in 1:n){
  X <- rmultinom(N, 1, EP)
  X <- DIGITS[apply(X, 2, which.max)]
  Count <- table(X)
  # Fill in missing digits with zero
  if(length(setdiff(DIGITS, as.numeric(names(Count)))) > 0) {
    missing_digit <- setdiff(DIGITS, as.numeric(names(Count)))
    Count[as.character(missing_digit)] <- 0
    Count <- Count[order(as.numeric(names(Count)))]
  }
  AP <- as.numeric(Count/sum(Count))
  folded_z <- abs(AP - EP)/sqrt(EP*(1-EP)/N)
  mMAD[i] <- mean(folded_z)
}

quantile(mMAD, c(1-0.95, 1-0.97, 1-0.99))
quantile(mMAD, c(0.8, 0.9, 0.95, 0.99))



