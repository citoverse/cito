source("utils.R")
library(cito)
set.seed(42)

df <- data.frame(Y = rnorm(60), X1 = rnorm(60), X2 = rnorm(60))

testthat::test_that("all optimizers can be configured and used", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  for(opt in c("sgd", "adam", "adadelta", "adagrad", "rmsprop", "rprop", "ignite_adam")) {
    testthat::expect_error({
      cfg <- config_optimizer(opt, verbose = FALSE)
      m <- dnn(Y ~ X1 + X2, data = df, optimizer = cfg, epochs = 2L, verbose = FALSE, plot = FALSE)
    }, NA, info = opt)
  }

  # optimizer specific arguments are picked up from the call
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, optimizer = config_optimizer("sgd", momentum = 0.9, verbose = FALSE),
             epochs = 2L, verbose = FALSE, plot = FALSE)
  }, NA)
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, optimizer = config_optimizer("rmsprop", eps = 1e-7, verbose = FALSE),
             epochs = 2L, verbose = FALSE, plot = FALSE)
  }, NA)
})


testthat::test_that("all learning rate schedulers can be configured and used", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  schedulers <- list(config_lr_scheduler("step", step_size = 2, verbose = FALSE),
                     config_lr_scheduler("multiplicative", lr_lambda = function(epoch) 0.95, verbose = FALSE),
                     config_lr_scheduler("lambda", lr_lambda = function(epoch) 0.95, verbose = FALSE),
                     config_lr_scheduler("reduce_on_plateau", patience = 1, verbose = FALSE),
                     config_lr_scheduler("one_cycle", max_lr = 0.05, total_steps = 4, verbose = FALSE))

  for(s in schedulers) {
    testthat::expect_error({
      m <- dnn(Y ~ X1 + X2, data = df, lr_scheduler = s, epochs = 4L, verbose = FALSE, plot = FALSE)
    }, NA, info = s$lr_scheduler)
  }

  # reduce_on_plateau also has to work on the validation loss
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, lr_scheduler = config_lr_scheduler("reduce_on_plateau", patience = 1, verbose = FALSE),
             validation = 0.3, epochs = 4L, verbose = FALSE, plot = FALSE)
  }, NA)
})


testthat::test_that("classic (non-analytical) ALE", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Sepal.Length ~ Petal.Length + Petal.Width, data = datasets::iris, epochs = 3L, verbose = FALSE, plot = FALSE)

  for(t in c("quantile", "equidistant")) {
    testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", analytical = FALSE, ALE_type = t,
                                      K = 5, plot = FALSE)}, NA, info = t)
  }
  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", analytical = FALSE,
                                    data = datasets::iris[1:60,], K = 5, plot = FALSE)}, NA)

  # multi-output
  m <- dnn(cbind(Sepal.Length, Sepal.Width) ~ Petal.Length + Petal.Width, data = datasets::iris,
           epochs = 3L, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", analytical = FALSE, K = 5, plot = FALSE)}, NA)
  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", plot = FALSE)}, NA)

  # bootstrapped, both paths
  m <- dnn(Sepal.Length ~ Petal.Length, data = datasets::iris, epochs = 2L, bootstrap = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", analytical = FALSE, K = 5, plot = FALSE)}, NA)
  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", plot = FALSE)}, NA)
})


testthat::test_that("architecture plot", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  testthat::skip_if_not_installed("igraph")
  testthat::skip_if_not_installed("ggraph")

  m <- dnn(Y ~ X1 + X2, data = df, hidden = c(5L, 5L), epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n <- plot(m, plot_type = "arch")}, NA)

  # embeddings and custom modules cannot be drawn, but must not fail
  df2 <- df
  df2$Cat <- as.factor(rep(1:3, length.out = nrow(df2)))
  m <- dnn(Y ~ X1 + e(Cat), data = df2, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n <- plot(m, plot_type = "arch")}, NA)
})


testthat::test_that("all hyperparameters can be tuned", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # each tune() argument has its own sampler
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, lambda = tune(), alpha = tune(), epochs = 2L,
             tuning = config_tuning(steps = 2, CV = 2), verbose = FALSE, plot = FALSE)
  }, NA)
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, bias = tune(), dropout = tune(), epochs = 2L,
             tuning = config_tuning(steps = 2, CV = 2), verbose = FALSE, plot = FALSE)
  }, NA)
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, activation = tune(), batchsize = tune(), epochs = 2L,
             tuning = config_tuning(steps = 2, CV = 2), verbose = FALSE, plot = FALSE)
  }, NA)
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, hidden = tune(lower = c(5, 1), upper = c(10, 2)), epochs = 2L,
             tuning = config_tuning(steps = 2, CV = 2, return_models = TRUE), verbose = FALSE, plot = FALSE)
  }, NA)
})


testthat::test_that("continue_training with a validation index vector", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Y ~ X1 + X2, data = df, validation = 1:15, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n <- continue_training(m, epochs = 2L, verbose = FALSE, plot = FALSE)}, NA)

  # new data without new validation indices warns and reuses the old ones
  df2 <- df[1:40, ]
  testthat::expect_warning({.n <- continue_training(m, data = df2, epochs = 2L, verbose = FALSE, plot = FALSE)})

  # validation indices beyond the new sample size are an error
  testthat::expect_error(continue_training(m, data = df[1:10, ], epochs = 2L, verbose = FALSE, plot = FALSE),
                         "Validation indices")

  # proportional validation split is redrawn on new data
  m <- dnn(Y ~ X1 + X2, data = df, validation = 0.2, epochs = 2L, verbose = FALSE, plot = FALSE)
  m2 <- continue_training(m, data = df2, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_true(all(m2$data$validation <= nrow(df2)))
})


testthat::test_that("conditional effects for models without continuous features", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  df2 <- data.frame(Y = rnorm(60), Cat = as.factor(rep(letters[1:3], length.out = 60)))
  m <- dnn(Y ~ Cat, data = df2, epochs = 2L, verbose = FALSE, plot = FALSE)

  testthat::expect_false(m$conditional_effects$any)
  testthat::expect_message({.n <- summary(m)})              # falls back to permutation importance
  testthat::expect_error({.n <- conditionalEffects(m)}, NA)
  testthat::expect_error({.n <- ALE(m, plot = FALSE)}, NA)

  # explicit indices select the columns the effects are computed for
  m2 <- dnn(Y ~ X1 + X2, data = df, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error({.n <- conditionalEffects(m2, indices = 1L)}, NA)
})


testthat::test_that("ALE trimming", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Sepal.Length ~ Petal.Length + Petal.Width, data = datasets::iris,
           epochs = 3L, verbose = FALSE, plot = FALSE)

  full <- ALE(m, variable = "Petal.Length", plot = FALSE)
  trimmed <- ALE(m, variable = "Petal.Length", trim = 0.1, plot = FALSE)
  testthat::expect_true(nrow(trimmed[[1]]$data) < nrow(full[[1]]$data))

  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", center = TRUE, plot = FALSE)}, NA)
  testthat::expect_error({.n <- ALE(m, variable = "Petal.Length", type = "link", plot = FALSE)}, NA)
})
