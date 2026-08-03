source("utils.R")
library(cito)
set.seed(42)

df <- data.frame(Y = rnorm(60), X1 = rnorm(60), X2 = rnorm(60))

testthat::test_that("X/Y interface", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  X <- as.matrix(df[, c("X1", "X2")])

  # single response
  m <- dnn(X = X, Y = df$Y, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(nrow(predict(m)), nrow(df))
  testthat::expect_equal(nrow(predict(m, newdata = X[1:10, ])), 10L)

  # multiple responses get default names
  Y2 <- cbind(df$Y, rnorm(60))
  m <- dnn(X = X, Y = Y2, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(ncol(predict(m)), 2L)

  # data.frame response
  m <- dnn(X = X, Y = data.frame(a = df$Y), epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(ncol(predict(m)), 1L)

  testthat::expect_error(dnn(epochs = 2L, verbose = FALSE, plot = FALSE), "Either formula")
})


testthat::test_that("responses supplied as data.frame or with unused factor levels", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # data.frame targets for the regression-style likelihoods
  testthat::expect_error({.n <- dnn(X = as.matrix(df[, c("X1", "X2")]),
                                    Y = data.frame(a = rnorm(60), b = rnorm(60)),
                                    loss = "mse", epochs = 2L, verbose = FALSE, plot = FALSE)}, NA)

  # a factor with an unused level is reported
  d2 <- data.frame(X = rnorm(60), Y = factor(rep(c("a", "b"), 30), levels = c("a", "b", "c")))
  testthat::expect_warning({.n <- dnn(Y ~ X, data = d2, loss = "cross-entropy", epochs = 2L,
                                      verbose = FALSE, plot = FALSE)})
})


testthat::test_that("mmn formula validation", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  n <- 40
  shapes <- simulate_shapes(n, 20)
  tab <- data.frame(X1 = rnorm(n), X2 = rnorm(n))
  dataList <- list(Y = rnorm(n), tab = tab, img = shapes$data)
  arch <- create_architecture(conv(3), maxPool(), linear(5))
  fit <- function(f) mmn(f, dataList = dataList, loss = "mse", epochs = 1L, verbose = FALSE, plot = FALSE)

  testthat::expect_error(fit(~ dnn(~., data = tab)), "response")
  testthat::expect_error(fit(Y ~ X1), "not allowed")
  testthat::expect_error(fit(Y ~ lm(~., data = tab)), "not supported")
  testthat::expect_error(fit(Y ~ dnn(data = tab)), "Either 'formula' or 'X'")
  testthat::expect_error(fit(Y ~ dnn(~., data = tab) + cnn(X = img)), "'architecture' or 'custom'")
  testthat::expect_error(fit(Y ~ dnn(~., data = tab) + cnn(architecture = arch)), "'X' must be specified")
  testthat::expect_error(fit(Y ~ dnn(~., data = missing_name)), "Couldn't find")
  testthat::expect_error(fit(Y ~ dnn(Y ~ ., data = tab)), "must not be specified")
})


testthat::test_that("architecture printing without shapes", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  arch <- create_architecture(conv(3), maxPool(), avgPool(), linear(5))
  testthat::expect_message({.n <- print(arch)})
  testthat::expect_error({.n <- print(arch, input_shape = c(1, 20, 20), output_shape = 3)}, NA)
})


testthat::test_that("cnn with a custom module", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  n <- 40
  shapes <- simulate_shapes(n, 20)
  custom <- torch::nn_module(initialize = function(output) {self$l <- torch::nn_linear(400, output)},
                             forward = function(x) self$l(torch::torch_flatten(x, start_dim = 2)))

  m <- cnn(shapes$data, shapes$labels, custom = custom, loss = "cross-entropy",
           epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(nrow(predict(m)), n)
  testthat::expect_error({.n <- plot(m)}, NA)   # custom architectures cannot be drawn
  testthat::expect_error(cnn(shapes$data, shapes$labels, architecture = create_architecture(linear(5)),
                             custom = custom, epochs = 1L, verbose = FALSE, plot = FALSE),
                         "either architecture or custom")
})


testthat::test_that("parallel bootstrapping", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()
  # the workers resolve dnn() from their own library, so this only tests the version under
  # development when that is also the installed one (not the case under devtools::load_all())
  desc <- file.path(.libPaths()[1], "cito", "DESCRIPTION")
  installed <- if(file.exists(desc)) as.character(read.dcf(desc, "Version")) else NA
  testthat::skip_if(is.na(installed) || installed != as.character(utils::packageVersion("cito")),
                    "installed cito differs from the version under test")

  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, epochs = 2L, bootstrap = 2L, bootstrap_parallel = 2L,
             verbose = FALSE, plot = FALSE)
  }, NA)
  testthat::expect_equal(length(m$models), 2L)
  testthat::expect_equal(nrow(predict(m)), nrow(df))
})
