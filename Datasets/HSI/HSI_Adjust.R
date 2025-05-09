#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
library("readxl")

start_year <- 1999
end_year <- 2002

stocks <- c("HSBC", "CLP", "CK")
adjr_type <- c("Rights Issue", "Stock Split", "Bonus", "Spinoff", "Scrip")
omit_type <- c("Omitted", "Cancelled", "Entitlement")
xlsx_name <- "stocks_1990_2023.xlsx"

start_date <- paste0(start_year, "-01-01")
end_date <- paste0(end_year, "-12-31")

df_price <- read_excel(xlsx_name, sheet="Close")
df_date <- as.Date(df_price$Date)
df_index <- which(df_date >= start_date & df_date <= end_date)
df_price <- df_price[df_index,]
price_date <- df_date[df_index]

results <- c()
for (com in stocks){
  df_dividend <- read_excel(xlsx_name, sheet=paste0(com, " (Dividend)"))
  if (is.character(df_dividend$`Ex Date`)){
    # window "1899-12-30"; macos "1904-01-01"
    if (.Platform$OS.type == "windows"){
      df_date <- as.Date(as.numeric(df_dividend$`Ex Date`), origin="1899-12-30")
    } else {
      df_date <- as.Date(as.numeric(df_dividend$`Ex Date`), origin="1904-01-01")
    }
  } else {
    df_date <- as.Date(df_dividend$`Ex Date`, format="%m/%d/%y")
  }
  df_index <- which(df_date >= start_date & df_date <= end_date)
  df_dividend <- df_dividend[df_index,]
  
  div_date <- df_date[df_index]
  dividend <- as.numeric(df_dividend$Amount)
  dividend[is.na(dividend)] <- 0
  div_type <- df_dividend$Type
  div_adjr <- df_dividend$`Adjustment Ratio`
  
  # Center for Research in Security Prices (CRSP) standards for dividend adjustment
  # https://help.yahoo.com/kb/SLN28256.html
  div_ratio <- 1; split_ratio <- 1; adj_ratio <- 1
  adj_close <- df_price[[com]]
  # loop from the end period to the start period
  for (i in length(price_date):1){
    adj_close[i] <- adj_close[i] * div_ratio * split_ratio
    
    # if we need adjustment in dividend 
    for (j in which(price_date[i] == div_date)){
      if (div_type[j] %in% adjr_type){
        split_ratio <- split_ratio * div_adjr[j]
        
        if (div_type[j] != "Spinoff") adj_ratio <- adj_ratio * div_adjr[j]
      } else if (!(div_type[j] %in% omit_type)){
        div_ratio <- div_ratio * (1 - dividend[j]/adj_ratio/adj_close[i-1])
      }
    }
  }
  results <- cbind(results, adj_close)
}
colnames(results) <- stocks
write.csv(results, file=paste0("stock_", start_year, "_", end_year, ".csv"), 
          row.names=format(price_date, "%d/%m/%Y"))