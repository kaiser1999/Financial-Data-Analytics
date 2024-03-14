#Set directory: Run this on source instead of Console!!
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
df <- read.csv('../Datasets/Mroz.csv')  # read the dataset

# create a new feature: number of children under 18
df$child <- df$child6 + df$child618
# remove the '#','child6', 'child618' features
df <- df[,-c(1,4,5)]                    

# fit the 1st model with all features
model_1 <- glm(formula=child~., family=poisson, data=df)
summary(model_1)            # print summary of the 1st model

################################################################################
# fit the 2nd model with the remaining features
model_2 <- glm(formula=child~work+hoursw+agew+income+experience,
               family=poisson, data=df)

summary(model_2)            # print summary of the 2nd model

# fit the 3rd model with the remaining features
model_3 <- glm(formula=child~hoursw+agew+experience, 
               family=poisson, data=df)

summary(model_3)            # print summary of the 3rd model