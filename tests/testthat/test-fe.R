source("utils.R")

testthat::test_that("fe() recovers linear effects next to a DNN", {
  skip_if_no_torch()
  testthat::skip_on_cran()

  set.seed(1)
  torch::torch_manual_seed(1)
  n = 2000
  d = data.frame(X1 = runif(n, -2, 2), Temp = rnorm(n), Precp = rnorm(n))
  d$Y = sin(2 * d$X1) + 2 * d$Temp - 1 * d$Precp + rnorm(n, sd = 0.2)
  m = dnn(Y ~ X1 + fe(Temp + Precp), data = d, loss = "mse", epochs = 60L, lr = 0.01, verbose = FALSE, plot = FALSE)
  b = coef(m)$fixed_effects
  testthat::expect_equal(rownames(b), c("Temp", "Precp"))
  testthat::expect_equal(as.vector(b), c(2, -1), tolerance = 0.1)
  testthat::expect_equal(m$model_properties$input, 3L)
  testthat::expect_equal(dim(m$net$l_1$weight)[2], 1L)
})

testthat::test_that("fe() formula handling: '.', I(), lambda/alpha and errors", {
  skip_if_no_torch()

  d = data.frame(X1 = rnorm(80), X2 = rnorm(80), Temp = rnorm(80), Y = rnorm(80))
  m = dnn(Y ~ . + fe(Temp + I(Temp^2), lambda = 0.01, alpha = 0.5), data = d, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(rownames(coef(m)$fixed_effects), c("Temp", "I(Temp^2)"))
  testthat::expect_equal(dim(m$net$l_1$weight)[2], 2L)
  testthat::expect_equal(m$model_properties$fe$lambda, 0.01)
  testthat::expect_equal(nrow(predict(m, newdata = d[1:5, ])), 5L)
  testthat::expect_error(dnn(Y ~ Temp + fe(Temp), data = d, epochs = 1L, verbose = FALSE, plot = FALSE), "either the DNN or fe")
  testthat::expect_error(dnn(Y ~ X1 + fe(Temp) + fe(X2), data = d, epochs = 1L, verbose = FALSE, plot = FALSE), "one fe")
  testthat::expect_error(dnn(Y ~ fe(Temp), data = d, epochs = 1L, verbose = FALSE, plot = FALSE), "glm")
})

testthat::test_that("the global lambda does not regularize the fixed effects", {
  skip_if_no_torch()

  set.seed(2)
  d = data.frame(X1 = rnorm(500), Temp = rnorm(500))
  d$Y = 3 * d$Temp + rnorm(500, sd = 0.1)
  m = dnn(Y ~ X1 + fe(Temp), data = d, loss = "mse", lambda = 1, alpha = 0, epochs = 40L, lr = 0.05, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(as.vector(coef(m)$fixed_effects), 3, tolerance = 0.1)
  m2 = dnn(Y ~ X1 + fe(Temp, lambda = 1, alpha = 0), data = d, loss = "mse", epochs = 40L, lr = 0.05, verbose = FALSE, plot = FALSE)
  testthat::expect_lt(abs(as.vector(coef(m2)$fixed_effects)), 2.5)
})

testthat::test_that("fe() combines with e(), and survives saveRDS, continue_training and bootstrap", {
  skip_if_no_torch()

  d = data.frame(X1 = rnorm(120), Temp = rnorm(120), g = factor(sample(letters[1:5], 120, TRUE)), Y = rnorm(120))
  m = dnn(Y ~ . + fe(Temp) + e(g, dim = 2), data = d, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(dim(m$net$l_1$weight)[2], 1L + 2L)
  testthat::expect_equal(rownames(coef(m)$fixed_effects), "Temp")
  f = tempfile(fileext = ".rds")
  saveRDS(m, f)
  m_r = readRDS(f)
  testthat::expect_equal(predict(m_r, newdata = d), predict(m, newdata = d), tolerance = 1e-6)
  testthat::expect_error(continue_training(m, epochs = 1L, verbose = FALSE), NA)
  testthat::expect_error(summary(m), NA)
  mb = dnn(Y ~ X1 + fe(Temp), data = d, epochs = 1L, bootstrap = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_error(predict(mb, newdata = d[1:3, ]), NA)
  testthat::expect_error(mmn(Y ~ dnn(~ X1 + fe(Temp), data = d), dataList = list(Y = d$Y), epochs = 1L, verbose = FALSE, plot = FALSE), "fe")
})
