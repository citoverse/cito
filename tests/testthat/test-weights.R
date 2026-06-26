source("utils.R")
library(cito)
set.seed(42)

# Observation weights in cito are made available to the (custom) likelihood. A
# custom loss that declares a 'weights' argument receives them. cito always
# passes this argument when it is present in the signature, supplying NULL when
# the model carries no weights (e.g. continuing training on new, unweighted
# data), so a robust weighted loss must handle weights = NULL. The helper also
# accepts 'weights' either as an R matrix (used when the baseline loss is
# computed) or as a torch_tensor (used during training).
weighted_mse <- function(pred, true, weights = NULL) {
  l <- torch::nnf_mse_loss(pred, true, reduction = "none")
  if(is.null(weights)) return(l)
  if(!inherits(weights, "torch_tensor")) {
    weights <- torch::torch_tensor(as.matrix(weights), dtype = pred$dtype)
  }
  l * weights
}

n <- 60
df <- data.frame(Y  = rnorm(n),
                 X1 = rnorm(n),
                 X2 = rnorm(n),
                 Cat = as.factor(rep(1:3, length.out = n)))
w <- runif(n, 0.5, 1.5)


testthat::test_that("DNN weights are accepted, stored and reused (vector and matrix)", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # vector weights
  m <- dnn(Y ~ X1 + X2, data = df, loss = weighted_mse, weights = w,
           epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_true(m$training_properties$has_weights)
  testthat::expect_equal(nrow(m$data$weights), n)
  testthat::expect_error({.p <- predict(m)}, NA)

  # matrix weights
  m2 <- dnn(Y ~ X1 + X2, data = df, loss = weighted_mse, weights = matrix(w, ncol = 1L),
            epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_true(m2$training_properties$has_weights)
  testthat::expect_equal(nrow(m2$data$weights), n)
})


testthat::test_that("DNN weights work with a validation split (with and without embeddings)", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # validation split, no embeddings
  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, loss = weighted_mse, weights = w,
             validation = 0.3, epochs = 2L, verbose = FALSE, plot = FALSE)
  }, NA)

  # validation split + embeddings (exercises the X / Z / weights / Y data loader path)
  testthat::expect_error({
    m <- dnn(Y ~ X1 + e(Cat), data = df, loss = weighted_mse, weights = w,
             validation = 0.3, epochs = 2L, verbose = FALSE, plot = FALSE)
  }, NA)
})


testthat::test_that("continue_training reuses stored weights and accepts new ones", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Y ~ X1 + X2, data = df, loss = weighted_mse, weights = w,
           epochs = 2L, verbose = FALSE, plot = FALSE)

  # (a) continue on original data -> weights are reused automatically
  testthat::expect_error({m_a <- continue_training(m, epochs = 2L)}, NA)
  testthat::expect_true(m_a$training_properties$has_weights)

  # (b) continue with explicitly supplied new weights
  w_new <- runif(n, 0.5, 1.5)
  testthat::expect_error({m_b <- continue_training(m, epochs = 2L, weights = w_new)}, NA)
  testthat::expect_true(m_b$training_properties$has_weights)

  # (c) continue on new data together with new weights
  df_new <- df[1:40, ]
  w_sub  <- runif(40, 0.5, 1.5)
  testthat::expect_error({m_c <- continue_training(m, epochs = 2L, data = df_new, weights = w_sub)}, NA)
  testthat::expect_true(m_c$training_properties$has_weights)

  # (d) continue on new data without weights -> unweighted, must not crash
  testthat::expect_error({m_d <- continue_training(m, epochs = 2L, data = df_new)}, NA)
  testthat::expect_false(m_d$training_properties$has_weights)

  # (e) mismatched weight length is reported clearly
  testthat::expect_error(continue_training(m, epochs = 1L, weights = runif(5)))
})


testthat::test_that("DNN weights work with bootstrapping", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m <- dnn(Y ~ X1 + X2, data = df, loss = weighted_mse, weights = w,
           bootstrap = 2L, epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_s3_class(m, "citodnnBootstrap")
  testthat::expect_true(all(sapply(m$models, function(mod) mod$training_properties$has_weights)))
  testthat::expect_error({.p <- predict(m)}, NA)
  testthat::expect_error({m2 <- continue_training(m, epochs = 1L)}, NA)
})


testthat::test_that("DNN weights work with hyperparameter tuning", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  testthat::expect_error({
    m <- dnn(Y ~ X1 + X2, data = df, loss = weighted_mse, weights = w,
             lr = tune(values = c(0.01, 0.1)), epochs = 2L,
             tuning = config_tuning(steps = 2, CV = 2), verbose = FALSE, plot = FALSE)
  }, NA)
  testthat::expect_error({.p <- predict(m)}, NA)
})


testthat::test_that("CNN weights are accepted, stored and reused", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(42)
  architecture <- create_architecture(conv(), maxPool(), linear())
  X <- array(runif(80 * 1 * 8 * 8), dim = c(80, 1, 8, 8))
  Y <- rnorm(80)
  wc <- runif(80, 0.5, 1.5)

  m <- cnn(X, Y, architecture, loss = weighted_mse, weights = wc,
           epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_true(m$training_properties$has_weights)
  testthat::expect_equal(nrow(m$data$weights), 80)
  testthat::expect_error({.p <- predict(m)}, NA)

  # continue on original data reuses the stored weights
  testthat::expect_error({m_a <- continue_training(m, epochs = 1L)}, NA)
  testthat::expect_true(m_a$training_properties$has_weights)

  # continue with a validation split + reused weights
  testthat::expect_error({
    m_v <- cnn(X, Y, architecture, loss = weighted_mse, weights = wc,
               validation = 0.25, epochs = 2L, verbose = FALSE, plot = FALSE)
    continue_training(m_v, epochs = 1L)
  }, NA)
})
