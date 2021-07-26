# 26/7/2021
# Álvaro Amorós Rodríguez
# HarvardX: PH125.9x Data Science: Capstone

# PREPARING LIBRARIES AND DATA
# Loading libraries and data. 
library(ggplot2)
library(scales)
library(ggthemes)
library(data.table)
library(reshape2)
library(tidyverse)
library(caret)
library(readr)
library(lubridate)
library(recosystem)
library(gridExtra )
library(dplyr, warn.conflicts = FALSE)

load("edx.Rda")
load("validation.Rda")

# Suppress summarise info
options(dplyr.summarise.inform = FALSE)

# Creating train and test sets
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, 
                                  list = FALSE)

train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# DATA SUMMARIE
# Summary Statistics
train_set %>%
  group_by(movieId) %>%
  mutate(n_ratings_movie = n()) %>%
  ungroup() %>%
  group_by(userId) %>%
  mutate(n_ratin_user = n ()) %>%
  ungroup() %>%
  summarise(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId),
            n_ratings = nrow(edx),
            avg_rating = mean(as.numeric(rating)),
            sd_rating = sd(as.numeric(rating)),
            max_raing = max(rating),
            min_rating = min(rating),
            mean_n_ratings_movie = round(mean(n_ratings_movie)),
            mean_n_rating_user = round(mean(n_ratin_user))) %>% knitr::kable(caption = "Summary statistics",
                                                                             col.names = c("Users", "Movies", "Ratings", "Avg.", "Sd",
                                                                                           "Max", "Min", "Avg. N. movie", "Avg. N. user"),
                                                                             align = "l")

# The data in a tibble
as_tibble(train_set) %>%
  head() %>% knitr::kable(caption = "Data table", align = "l")

# GRAPHICAL ANALYSIS
# Rating distribution
rating_distribution <-
  train_set %>% group_by(rating) %>% 
  summarise(count=n(), .groups = 'drop') %>%
  ggplot(aes(x=rating, y=count)) + 
  geom_bar(stat = "identity") +
  ggtitle("Rating Distribution") + 
  xlab("Rating") +
  ylab("Count") + 
  theme_hc()

# Number of votes per user
n_ratings_user <- 
  train_set %>% 
  group_by(userId) %>%
  summarise(n=n(), .groups = 'drop') %>%
  ggplot(aes(x = n)) +
  geom_histogram(color = "black", bins = 30) +
  scale_x_log10() + 
  ggtitle("Number of ratings per User") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  scale_y_continuous(labels = comma) +
  theme_hc()

# User average rating
mean_user_vote <-
  train_set %>%
  group_by(userId) %>%
  summarise(mean_vote = mean(rating), .groups = 'drop') %>%
  ggplot(aes(mean_vote)) +
  geom_histogram(bins = 30, color = "black") +
  ggtitle("User average rating") +
  xlab("User average rating") +
  ylab("Count") + 
  scale_y_continuous(labels = comma) +
  theme_hc()

# Number of votes per movie
n_ratings_movie <-
  train_set %>% group_by(movieId) %>%
  summarise(n=n(), .groups = 'drop') %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins = 30) +
  scale_x_log10() + 
  ggtitle("Number of Rarings per movie") +
  xlab("Number of Ratings") +
  ylab("Count") +
  theme_hc()

# Average movie ratings
mean_movie_vote <-
  train_set %>% group_by(movieId) %>%
  summarise(mean_vote = mean(rating), .groups = 'drop') %>%
  ggplot(aes(mean_vote)) +
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Movie average rating") +
  xlab("Movie average rating") +
  ylab("Count") +
  theme_hc()

# Putting all four graphs together in a grid.
grid.arrange( n_ratings_user, n_ratings_movie, mean_user_vote, mean_movie_vote)


# MODEL TESTING

# Defining the function for the RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Naive model
# Mean rating in the train set
mu <- mean(train_set$rating)

# Naive rmse model

naive_rmse <- RMSE(test_set$rating, mu)

# Tibble to store results
rmse_results <- tibble(method = "Objective", RMSE = 0.864900)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Mean",
                                 RMSE = naive_rmse))

rmse_results %>% knitr::kable(caption = "Models", align = "l")

# Movie effect model

# The deviation of each movie from the mean rating
bi <- train_set %>%
  group_by(movieId, .groups = 'drop') %>%
  summarise(b_i = mean(rating - mu))

# Add the movie effect to the model
y_hat_bi <- mu + test_set %>%
  left_join(bi, by = "movieId") %>%
  pull(b_i)

# Calculate RMSE
movie_rmse <- RMSE(test_set$rating, (y_hat_bi))

# Add RMSE to the tibble.
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie Effect",
                                 RMSE = movie_rmse))

rmse_results %>% knitr::kable(caption = "Models", align = "l")

# User effect model

# The deviation of each users from the overall mean + movie effect.
bu <- train_set %>%
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# Adding the user effect to the model.
y_hat_bu <- test_set %>%
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

# Calculating RMSE
user_rmse <- RMSE(test_set$rating, y_hat_bu)

# Adding RMSE to the table.
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie and User Effects",
                                 RMSE = user_rmse))
rmse_results %>% knitr::kable(caption = "Models", align = "l")

# Regularized models

# Deviation of movies and users from the mean.

# Movie Deviations
movie_agvs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

# User deviations
user_agvs <- train_set %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu))

# Graph 1
g_1 <- qplot(b_i, data = movie_agvs, bins = 30, color = I("black"), main = "Deviation of movies from the mean ", ylab = "N. Rarings", xlab = "Movie effect")

# Graph 2
g_2 <- qplot(b_u, data = user_agvs, bins = 30, color = I("black"), main = "Deviation of users from the mean", ylab = "N. Rarings", xlab = "User effect")

# Grid
grid.arrange(g_1, g_2, nrow = 1)

# Table with movies and users with the greatest deviations and its rating

# Movies
movie_titles <- edx %>% 
  select(movieId, title, rating) %>%
  distinct(movieId, title, rating)

train_set %>% dplyr::count(movieId) %>% 
  left_join(bi) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n, rating) %>% 
  slice(1:10) %>% 
  knitr::kable(caption = "Movies with the highest deviation from the mean",
               col.names = c("Title", "Deviation", "N. Votes", "Rating"),
               align = "l")

# Users
user_ids <- edx %>% 
  select(userId, title, rating) %>%
  distinct(userId, rating)

train_set %>% dplyr::count(userId) %>% 
  left_join(bu) %>%
  left_join(user_ids, by="userId") %>%
  arrange((b_u)) %>% 
  select(userId, b_u, n, rating) %>% 
  slice(1:10) %>% 
  knitr::kable(caption = "users with the highest deviation from the mean",
               col.names = c("User", "Deviation", "N. Votes", "Rating"),
               align = "l")

# Regularization of the movie effect.

# Different regularization terms we will test
lambdas_1 <- seq(0, 10, 0.5)

# Regularization function
rmses_1 <- sapply(lambdas_1, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n()))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(predictions = mu + b_i + b_u) %>%
    .$predictions
  
  return(RMSE(test_set$rating, predicted_ratings))
  
})

# Choose lambada which minimizes rmse
l_1 <- lambdas_1[which.min(rmses_1)]


# Regularization of user effect

# Different regularization terms we will test
lambdas_2 <- seq(0, 10, 0.5)

# Regularization function
rmses_2 <- sapply(lambdas_2, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n() + l_1))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n() + l ))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(predictions = mu + b_i + b_u) %>%
    .$predictions
  
  return(RMSE(test_set$rating, predicted_ratings))
  
})

# Choose lambada which minimizes rmse
l_2 <- lambdas_2[which.min(rmses_2)]

# Graphs of different RMSEs for for different  regularization terms 
regu1 <- qplot(lambdas_1, rmses_1, main = "Regularization Movie effect",
               xlab = "Lambda",
               ylab = "RMSE")  

regu2 <- qplot(lambdas_2, rmses_2, main = "Regularization User effect",
               xlab = "Lambda",
               ylab = "RMSE")  

# Grid with both graphs
grid.arrange(regu1, regu2, nrow = 1)


# Add lambdas which minimize rmse to the model which user and movie effects
b_i <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (n() + l_1))

b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n() + l_2))

predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(predictions = mu + b_i + b_u) %>%
  .$predictions

regularization_model <-  RMSE(test_set$rating, predicted_ratings)

# Add results of regularized model to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + User Effects + Regularization",
                                 RMSE = regularization_model))
rmse_results %>% knitr::kable(caption = "Models", align = "l")

# Matrix Factorization

# Prepare the data to be used in with recosystem
train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create a Recommender object
r <- recosystem::Reco()

# create and object to save the the parameters that fit the best: number of latent factors, gradient descent rate and penalty 
# parameter to avoid overfiting.
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 4, niter = 10))
# Train the data with the parameters that maximize the fit of the model
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

# Predict outcomes
predicted_ratings <-  r$predict(test_data, out_memory())

# Test RMSE
factorization_model <- RMSE(test_set$rating, predicted_ratings)
factorization_model
# Train the data with the parameters that maximize the fit of the model
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

# Predict outcomes
predicted_ratings <-  r$predict(test_data, out_memory())

# Test RMSE
factorization_model <- RMSE(test_set$rating, predicted_ratings)
factorization_model

# Add results to table with the other models
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Matrix Factorization",
                                 RMSE = factorization_model))
rmse_results %>% knitr::kable(caption = "Models", align = "l")

# FINAL MODEL

# Repeat all models directly with the data provided by the exercise
# Naive
mu <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu)
rmse_results <- tibble(method = "Objective", RMSE = 0.864900)


# Movie effect
bi <- edx %>%
  group_by(movieId, .groups = 'drop') %>%
  summarise(b_i = mean(rating - mu))

y_hat_bi <- mu + validation %>%
  left_join(bi, by = "movieId") %>%
  pull(b_i)

movie_rmse <- RMSE(validation$rating, y_hat_bi)

# User effect
bu <- edx %>%
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

y_hat_bu <- validation %>%
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

user_rmse <- RMSE(validation$rating, y_hat_bu)


# Regularization
lambdas_1 <- seq(0, 10, 0.5)

rmses_1 <- sapply(lambdas_1, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n()))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(predictions = mu + b_i + b_u) %>%
    .$predictions
  
  return(RMSE(validation$rating, predicted_ratings))
  
})

l_1 <- lambdas_1[which.min(rmses_1)]


lambdas_2 <- seq(0, 10, 0.5)

rmses_2 <- sapply(lambdas_2, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n() + l_1))
  
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n() + l ))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(predictions = mu + b_i + b_u) %>%
    .$predictions
  
  return(RMSE(validation$rating, predicted_ratings))
  
})
l_2 <- lambdas_2[which.min(rmses_2)]


b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (n() + l_1))

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n() + l_2))

predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(predictions = mu + b_i + b_u) %>%
  .$predictions

regularization_model <-  RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + User  + Regularization (edx-validation)",
                                 RMSE = regularization_model))


# Matrix fatorization 

train_data_edx <-  with(edx, data_memory(user_index = userId, 
                                         item_index = movieId, 
                                         rating     = rating))
test_data_validation  <-  with(validation,  data_memory(user_index = userId, 
                                                        item_index = movieId, 
                                                        rating     = rating))

# Create a Recommender object
r <- recosystem::Reco()


# Optimize parametes
opts <- r$tune(train_data_edx, opts = list(dim = c(10, 20, 30), 
                                           lrate = c(0.1, 0.2),
                                           costp_l2 = c(0.01, 0.1), 
                                           costq_l2 = c(0.01, 0.1),
                                           nthread  = 4, niter = 10))

# Run model
r$train(train_data_edx, opts = c(opts$min, nthread = 4, niter = 20))

predicted_ratings <-  r$predict(test_data_validation, out_memory())

factorization_model <- RMSE(validation$rating, predicted_ratings)
factorization_model

# Final results table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Matrix Factorization (edx-validation)",
                                 RMSE = factorization_model))
rmse_results %>% knitr::kable(caption = "Models", align = "l")
