testthat::test_that("xAI methods/plots", {
  testthat::skip_on_cran()
  set.seed(222)
  validation_set<- sample(c(1:nrow(datasets::iris)),25)

  # Build and train  Network
  nn.fit<- dnn(Sepal.Length~., data = datasets::iris[-validation_set,], epochs = 5L)
  testthat::expect_error({plot(PDP(model))}, NA)
  testthat::expect_error({plot(ALE(model))}, NA)
  testthat::expect_error({plot(PDP(model, variable = c("Sepal.Width",  "Petal.Length")))}, NA)
  testthat::expect_error({plot(PDP(model, variable = c("Sepal.Width",  "Petal.Length")))}, NA)
})
