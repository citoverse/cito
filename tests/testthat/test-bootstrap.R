source("utils.R")
library(cito)
set.seed(42)

df <- data.frame(Y = rnorm(60), X1 = rnorm(60), X2 = rnorm(60),
                 Cat = as.factor(rep(letters[1:3], length.out = 60)))

testthat::test_that("xAI methods on bootstrapped models", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Y ~ X1 + X2 + Cat, data = df, epochs = 3L, bootstrap = 3L, verbose = FALSE, plot = FALSE)

  # PDP for a numeric and for a categorical feature (different code paths)
  testthat::expect_error({.n <- PDP(m, variable = "X1", plot = FALSE)}, NA)
  testthat::expect_error({.n <- PDP(m, variable = "Cat", plot = FALSE)}, NA)
  testthat::expect_error({.n <- PDP(m, variable = "X1", ice = TRUE, plot = FALSE)}, NA)

  testthat::expect_error({.n <- ALE(m, variable = "X1", plot = FALSE)}, NA)
  testthat::expect_error({.n <- ALE(m, variable = "X1", analytical = FALSE, K = 5, plot = FALSE)}, NA)

  ce <- conditionalEffects(m)
  testthat::expect_true(is.list(ce))

  for(imp in c("ce", "permutation")) {
    testthat::expect_error({.n <- summary(m, importance = imp, n_permute = 2L)}, NA, info = imp)
  }
  testthat::expect_error({.n <- predict(m, reduce = "median")}, NA)
  testthat::expect_error({.n <- predict(m, reduce = "none")}, NA)
  testthat::expect_error({.n <- coef(m)}, NA)
  testthat::expect_error({.n <- print(m)}, NA)
})


testthat::test_that("summary of a bootstrapped model fitted with ce = FALSE", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Y ~ X1 + X2, data = df, epochs = 2L, bootstrap = 2L, ce = FALSE, verbose = FALSE, plot = FALSE)
  testthat::expect_null(m$models[[1]]$conditional_effects)
  testthat::expect_error({.n <- summary(m)}, NA)
})


testthat::test_that("blocked bootstrap and blocked tuning", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, epochs = 2L, bootstrap = 2L,
             bootstrap_blocking_variable = df$Cat, verbose = FALSE, plot = FALSE)
  }, NA)

  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, lr = tune(values = c(0.01, 0.1)), epochs = 2L,
             tuning = config_tuning(steps = 2, CV = 2, blocking = df$Cat), verbose = FALSE, plot = FALSE)
  }, NA)

  # CV larger than the number of blocks is reduced with a message
  testthat::expect_message({
    m <- dnn(Y ~ X1 + X2, data = df, lr = tune(values = c(0.01, 0.1)), epochs = 2L,
             tuning = config_tuning(steps = 1, CV = 10, blocking = df$Cat), verbose = FALSE, plot = FALSE)
  })
})


testthat::test_that("out-of-bag permutation importance", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Y ~ X1 + X2, data = df, epochs = 2L, bootstrap = 2L, verbose = FALSE, plot = FALSE)
  imp <- cito:::get_importance(m$models[[1]], n_permute = 2L, importance = "permutation", out_of_bag = TRUE)
  testthat::expect_s3_class(imp, "data.frame")
  testthat::expect_equal(nrow(imp), 2L)
})
