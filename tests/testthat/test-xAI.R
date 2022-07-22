source("utils.R")

testthat::test_that("xAI methods/plots", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(222)
  validation_set<- sample(c(1:nrow(datasets::iris)),25)

  # Build and train  Network
  model <- dnn(Sepal.Length~., data = datasets::iris[-validation_set,], epochs = 1L)
  testthat::expect_error({PDP(model)}, NA)
  testthat::expect_error({ALE(model,K = 3)}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
  testthat::expect_error({PDP(model, variable = c("Sepal.Width",  "Petal.Length"))}, NA)
})
