library(recommenderlab)
data("MovieLense")

# only counts rows (users) with more than 100 ratings (non-zero entries)
idx <- rowCounts(MovieLense) > 100
(MovieLense100 <- MovieLense[idx,])
# 358 users with more than 100 ratings, and 1664 movies with at least one rating

(train <- MovieLense100[1:50])
# train the model using the rating data from the first 50 users with the top 10
# latent features
(rec <- Recommender(train, method="SVD", parameter=list(k=10)))

idx[idx][101:102]     # Users 291 and 292
(pre <- predict(rec, MovieLense100[101:102], n=10))
as(pre, "list")

################################################################################
MovieLense

# only counts rows (users) with more than 100 ratings (non-zero entries)
idx <- rowCounts(MovieLense) > 100
(MovieLense100 <- MovieLense[idx,])

# train the model with the top 10 latent features in SVD
set.seed(4002)
(rec <- Recommender(MovieLense100, method="SVD", parameter=list(k=10)))
(pre <- predict(rec, MovieLense[!idx,], n=10))
as(pre, "list")

################################################################################
user1 <- as(MovieLense100[101], "data.frame")
user1$item[head(order(user1$rating, decreasing=TRUE), 10)]

user2 <- as(MovieLense100[102], "data.frame")
user2$item[head(order(user2$rating, decreasing=TRUE), 10)]