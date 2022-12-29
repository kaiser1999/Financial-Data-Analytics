#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library(readxl)
library(poweRlaw)

################################################################################
get_Zipf <- function(country){
  df <- read_excel("Zipf's Law.xlsx", sheet=country)
  confirmed <- df$Confirmed
  m <- displ$new(confirmed[confirmed > 0])
  x_min <- estimate_xmin(m, pars=NULL)$xmin
  m$setXmin(x_min)
  beta <- estimate_pars(m, pars=NULL)$pars
  m$setPars(beta)

  temp <- bootstrap(m, seed=1)
  sd1 <- sd(temp$bootstraps[,3])
  sd2 <- sd(temp$bootstraps[,2])
  
  ntail <- which(confirmed <= x_min)[1]
  model <- lm(log(df$Rank[1:ntail]) ~ log(df$Confirmed[1:ntail]))
  
  return(list(Country=country, beta=beta, SD=sd1, x_min=x_min, SD=sd2, 
              ntail=ntail, GOF=temp$gof, pvalue=bootstrap_p(m)$p, Country=country,
              alpha_LS=model$coefficients[1], beta_LS=-model$coefficients[2], 
              R2=summary(model)$r.squared, ntail=ntail))
}

################################################################################
COUNTRIES <- c("Brazil", "Canada", "China", "China_Excl_Hubei", "Germany", 
               "India","Italy", "Malaysia", "Mexico", "Netherlands", 
               "Romania", "Russia", "Spain", "Sweden", "UK", "USA")

Results <- list()
for (country in COUNTRIES){
  print(country)
  Results <- append(Results, get_Zipf(country))
}

mat <- matrix(unlist(Results), ncol=13, byrow=TRUE)
colnames(mat) <- c("Country", "beta", "SD", "x_min", "SD", "ntail", "GOF", "pvalue", 
                   "Country", "alpha_LS", "beta_LS", "R2", "ntail")
write.csv(mat, "Zipf's results.csv")

################################################################################
plot_Zipf <- function(countries, colors){
  df_x_min <- read_excel("Zipf's Law.xlsx", sheet="x_min")
  
  df_count <- read_excel("Zipf's Law.xlsx", sheet=countries[1])
  ntail <- df_x_min$ntail[which(df_x_min$Country == countries[1])]
  plot(log(df_count$Confirmed[1:ntail]) ~ log(df_count$Rank[1:ntail]), 
       type="b", pch=19, lwd=1.5, col=colors[1], bg=colors[1], 
       ylim=c(0,15), xlab='log(rank)', ylab='log(cases)')
  
  for (i in 2:length(countries)){
    df_count <- read_excel("Zipf's Law.xlsx", sheet=countries[i])
    ntail <- df_x_min$ntail[which(df_x_min$Country == countries[i])]
    lines(log(df_count$Confirmed[1:ntail]) ~ log(df_count$Rank[1:ntail]), 
          type="b", pch=19, lwd=1.5, col=colors[i], bg=colors[i])
  }
  legend("bottomleft", countries, col=colors, pch=19, lwd=1.5)
}

colors <- c("black", "darkred", "darkblue", "darkgrey", "darkgreen", "gold")

################################################################################
countries <- c("USA", "Canada", "Brazil", "Germany", "India", "Italy")
plot_Zipf(countries, colors)

################################################################################
countries <- c("Malaysia", "Mexico", "Romania", "Spain", "Sweden", "UK")
plot_Zipf(countries, colors)

################################################################################
countries <- c("Russia", "China", "China_Excl_Hubei")
plot_Zipf(countries, colors[1:length(countries)])

curve(11.611+-2.225*x, add=TRUE, xlim=c(0, log(6)), col='red', lwd=3)
curve(8.8709-0.5431*x, add=TRUE, xlim=c(log(7), log(54)), col='red', lwd=3)

curve(7.4982-0.4244*x, add=TRUE, xlim=c(0, log(13)), col='red', lwd=3)
curve(10.317-1.684*x, add=TRUE, xlim=c(log(14), log(26)), col='red', lwd=3)