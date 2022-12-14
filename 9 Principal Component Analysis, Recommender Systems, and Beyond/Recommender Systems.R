library(recommenderlab)
data("MovieLense")

# only counts rows (users) with more than 100 ratings (non-zero entries)
(MovieLense100 <- MovieLense[rowCounts(MovieLense)>100,])
# 358 users with more than 100 ratings, and 1664 movies with at least one rating

# train the model using the rating data from the first 50 users
(train <- MovieLense100[1:300])

# train the model with the top 10 latent features in SVD
(rec <- Recommender(train, method="SVD", parameter=list(k=10)))

(pre <- predict(rec, MovieLense100[301:302], n=10))
as(pre, "list")

################################################################################
user1 <- as(MovieLense100[301], "data.frame")
user1$item[head(order(user1[,3], decreasing=TRUE), 10)]

user2 <- as(MovieLense100[302], "data.frame")
user2$item[head(order(user2[,3], decreasing=TRUE), 10)]