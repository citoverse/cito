source("utils.R")
library(cito)
set.seed(42)

testthat::test_that("analytical conditional effects match a central difference", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  n <- 300
  d <- data.frame(X1 = rnorm(n), X2 = rnorm(n))
  d$Y <- 1.5*d$X1 - 2*d$X2 + rnorm(n, sd = .3)
  m <- dnn(Y ~ ., data = d, hidden = c(20,20), epochs = 50, verbose = FALSE, plot = FALSE, lr = .05)

  ref <- sapply(c("X1","X2"), function(v) {
    h <- 1e-2*sd(d[[v]]); dp <- dm <- d
    dp[[v]] <- dp[[v]] + h; dm[[v]] <- dm[[v]] - h
    (predict(m, dp, type = "response")[,1] - predict(m, dm, type = "response")[,1])/(2*h)
  })
  ce <- conditionalEffects(m, method = "analytical", type = "response")
  ana <- sapply(1:2, function(j) ce[[1]]$result[,j,j])
  testthat::expect_lt(mean(abs(ana - ref)), 0.01)

  fin <- conditionalEffects(m, method = "finite", epsilon = 0.01, type = "response")
  testthat::expect_lt(mean(abs(sapply(1:2, function(j) fin[[1]]$result[,j,j]) - ref)), 0.05)
})


testthat::test_that("finite interactions are scaled by h_i * h_j", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # the two features deliberately have different spread; with the wrong scaling the
  # finite cross derivative is off by sd(B)/sd(A)
  n <- 400
  d <- data.frame(A = rnorm(n), B = rnorm(n, sd = 3))
  d$Y <- d$A*d$B + rnorm(n, sd = .2)
  m <- dnn(Y ~ ., data = d, hidden = c(32,32), epochs = 250, verbose = FALSE, plot = FALSE, lr = .01)

  ana <- mean(conditionalEffects(m, interactions = TRUE)[[1]]$result[,1,2])
  fin <- mean(conditionalEffects(m, method = "finite", interactions = TRUE)[[1]]$result[,1,2])
  testthat::expect_lt(abs(fin/ana - 1), 0.25)
})


testthat::test_that("ALE variance importance recovers a known decomposition", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  n <- 2000
  d <- data.frame(X1 = rnorm(n), X2 = rnorm(n), X3 = rnorm(n))
  d$Y <- 1*d$X1 - 2*d$X2 + rnorm(n, sd = .3)          # X3 irrelevant
  m <- dnn(Y ~ ., data = d, hidden = c(50,50), epochs = 200, verbose = FALSE, plot = FALSE, lr = .01)

  truth <- c(var(d$X1), var(-2*d$X2), 0)
  imp <- summary(m, importance = "ale")$importance[,2]
  testthat::expect_lt(mean(abs(imp/sum(imp) - truth/sum(truth))), 0.05)
  # the parts of an additive model with independent features add up to Var(prediction)
  testthat::expect_lt(abs(sum(imp)/var(predict(m)[,1]) - 1), 0.15)
})


testthat::test_that("ale weights are normalised", {
  x <- c(1, 1, 2, 3, 3, 3)
  ux <- sort(unique(x))
  for(w in list(FALSE, "frequency", "density")) {
    testthat::expect_equal(sum(cito:::ale_weights(x, ux, w)), 1)
  }
  testthat::expect_equal(cito:::ale_weights(x, ux, "frequency"), c(2,1,3)/6)
})


testthat::test_that("conditional effects are subsampled on large data", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  testthat::expect_equal(cito:::ce_subsample(TRUE, 100, 5, 1), FALSE)      # small, keep all
  testthat::expect_equal(cito:::ce_subsample(TRUE, 20000, 5, 1), 0.25)     # capped at 5000
  testthat::expect_equal(cito:::ce_subsample(500, 20000, 5, 1), 0.025)     # user supplied
  testthat::expect_true(is.na(cito:::ce_subsample(TRUE, 1e6, 4000, 1)))    # too wide

  n <- 8000
  d <- data.frame(X1 = rnorm(n), X2 = rnorm(n)); d$Y <- d$X1 + rnorm(n)
  m <- dnn(Y ~ ., data = d, epochs = 2, verbose = FALSE, plot = FALSE, batchsize = 256)
  testthat::expect_equal(length(m$conditional_effects$row_indices), 5000L)
  testthat::expect_error({.n <- summary(m, importance = "ale")}, NA)
  testthat::expect_error({.n <- ALE(m, plot = FALSE)}, NA)

  m2 <- dnn(Y ~ ., data = d, epochs = 2, ce = 200, verbose = FALSE, plot = FALSE, batchsize = 256)
  testthat::expect_equal(length(m2$conditional_effects$row_indices), 200L)
})


testthat::test_that("conditionalEffects accepts subsample together with new data", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  n <- 300
  d <- data.frame(X1 = rnorm(n), X2 = rnorm(n)); d$Y <- d$X1 + rnorm(n)
  m <- dnn(Y ~ ., data = d, epochs = 5, verbose = FALSE, plot = FALSE)
  ce <- conditionalEffects(m, data = d[1:50,], subsample = .5)
  testthat::expect_equal(dim(ce[[1]]$result)[1], 25L)
})
