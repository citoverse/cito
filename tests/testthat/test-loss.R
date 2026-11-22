source("utils.R")
library(cito)
set.seed(42)

X_cnn <- array(runif(prod(c(100, 3, 10, 10))), dim = c(100, 3, 10, 10))
X_dnn <- matrix(runif(300), 100, 3)

architecture <- create_architecture(conv(), maxPool(), conv(), avgPool(), linear())

test_loss <- function(loss, Y) {
  inner <- function(device) {
    for (y in Y) {
      if(ncol(as.matrix(y)) == 1) {
        formula <- stats::formula("Y ~ .")
        data <- data.frame(Y=y, X_dnn)
      } else {
        formula <- stats::formula(paste0("cbind(", paste(paste0("Y.", 1:ncol(as.matrix(y))), collapse=","), ") ~ ."))
        data <- data.frame(y, X_dnn)
      }

      .n <- wrap_dnn(list(formula=formula, data=data, loss=loss, device=device, epochs=1, plot=FALSE, verbose=FALSE))
      .n <- wrap_dnn(list(formula=formula, data=data, loss=loss, device=device, epochs=1, plot=FALSE, verbose=FALSE, bootstrap=2))
      # .n <- wrap_cnn(list(X=X_cnn, Y=y, architecture=architecture, loss=loss, device=device, epochs=1, plot=FALSE, verbose=FALSE))
      # .n <- wrap_mmn(list(formula=Y ~ cnn(X=X_cnn, architecture=architecture) + dnn(X=X_dnn), dataList=list(X_cnn=X_cnn, X_dnn=X_dnn, Y=y), loss=loss, device=device, epochs=1, plot=FALSE, verbose=FALSE))
    }
  }

  inner("cpu")
  if(torch::cuda_is_available()) inner("cuda")
  if(torch::backends_mps_is_available()) inner("mps")
}

testthat::test_that("test cross-entropy", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)),
            sample(c("a","b","c"), 100, replace=TRUE),
            matrix(sample(c("a","b","c"), 100, replace=TRUE), nrow=100, ncol=1),
            data.frame(Y=sample(c("a","b","c"), 100, replace=TRUE)))
  test_loss("cross-entropy", Y)
})

testthat::test_that("test mse", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(runif(100),
            matrix(runif(100), nrow=100, ncol=1),
            matrix(runif(300), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y=runif(100)),
            data.frame(Y.1=runif(100), Y.2=runif(100), Y.3=runif(100)))

  test_loss("mse", Y)
})

testthat::test_that("test mae", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(runif(100),
            matrix(runif(100), nrow=100, ncol=1),
            matrix(runif(300), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y=runif(100)),
            data.frame(Y.1=runif(100), Y.2=runif(100), Y.3=runif(100)))

  test_loss("mae", Y)
})

testthat::test_that("test bernoulli", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(sample(c(0,1), 100, replace = TRUE),
            matrix(sample(c(0,1), 100, replace = TRUE), nrow=100, ncol=1),
            matrix(sample(c(0,1), 300, replace = TRUE), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y=sample(c(0,1), 100, replace = TRUE)),
            data.frame(Y.1=sample(c(0,1), 100, replace = TRUE), Y.2=sample(c(0,1), 100, replace = TRUE), Y.3=sample(c(0,1), 100, replace = TRUE)))

  test_loss("bernoulli", Y)
})

testthat::test_that("test gaussian", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(runif(100),
            matrix(runif(100), nrow=100, ncol=1),
            matrix(runif(300), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y=runif(100)),
            data.frame(Y.1=runif(100), Y.2=runif(100), Y.3=runif(100)))

  test_loss("gaussian", Y)
})

testthat::test_that("test poisson", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(rpois(100, 10),
            matrix(rpois(100, 10), nrow=100, ncol=1),
            matrix(rpois(300, 10), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y=rpois(100, 10)),
            data.frame(Y.1=rpois(100, 10), Y.2=rpois(100, 20), Y.3=rpois(100, 30)))

  test_loss("poisson", Y)
})

testthat::test_that("test binomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b"), 100, replace=TRUE)),
            sample(c("a","b"), 100, replace=TRUE),
            matrix(sample(c("a","b"), 100, replace=TRUE), nrow=100, ncol=1),
            data.frame(Y=sample(c("a","b"), 100, replace=TRUE)),
            matrix(sample(0:10, 200, replace = TRUE), nrow=100, ncol=2, dimnames = list(NULL, c("Y.1","Y.2"))),
            data.frame(Y.1=sample(0:10, 100, replace = TRUE), Y.2=sample(0:10, 100, replace = TRUE)))

  test_loss("binomial", Y)
})

testthat::test_that("test nbinom", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(rnbinom(100, size=0.5, mu=10),
            matrix(rnbinom(100, size=0.5, mu=10), nrow=100, ncol=1),
            matrix(rnbinom(300, size=0.5, mu=10), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y=rnbinom(100, size=0.5, mu=10)),
            data.frame(Y.1=rnbinom(100, size=0.5, mu=10), Y.2=rnbinom(100, size=0.5, mu=20), Y.3=rnbinom(100, size=0.5, mu=30)))

  test_loss("nbinom", Y)
})

testthat::test_that("test multinomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)),
            sample(c("a","b","c"), 100, replace=TRUE),
            matrix(sample(c("a","b","c"), 100, replace=TRUE), nrow=100, ncol=1),
            data.frame(Y=sample(c("a","b","c"), 100, replace=TRUE)),
            matrix(sample(0:10, 300, replace = TRUE), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y.1=sample(0:10, 100, replace = TRUE), Y.2=sample(0:10, 100, replace = TRUE), Y.3=sample(0:10, 100, replace = TRUE)))

  test_loss("multinomial", Y)
})

testthat::test_that("test mvp", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(matrix(sample(c(0,1), 300, replace = TRUE), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y.1=sample(c(0,1), 100, replace = TRUE), Y.2=sample(c(0,1), 100, replace = TRUE), Y.3=sample(c(0,1), 100, replace = TRUE)))

  test_loss("mvp", Y)
})

testthat::test_that("test clogit", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)),
            sample(c("a","b","c"), 100, replace=TRUE),
            matrix(sample(c("a","b","c"), 100, replace=TRUE), nrow=100, ncol=1),
            data.frame(Y=sample(c("a","b","c"), 100, replace=TRUE)),
            matrix(sample(c(0,1), 300, replace = TRUE), nrow=100, ncol=3, dimnames = list(NULL, c("Y.1","Y.2","Y.3"))),
            data.frame(Y.1=sample(c(0,1), 100, replace = TRUE), Y.2=sample(c(0,1), 100, replace = TRUE), Y.3=sample(c(0,1), 100, replace = TRUE)))

  test_loss("clogit", Y)
})



testthat::test_that("nbinom exposes the dispersion parameter and can simulate", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(42)
  df <- data.frame(X = runif(80))
  df$Y <- rpois(80, exp(1 + 0.5 * df$X))

  m <- dnn(Y ~ X, data = df, loss = "nbinom", epochs = 5L, verbose = FALSE, plot = FALSE)

  theta <- m$loss$parameter_link()
  testthat::expect_true(is.numeric(theta))
  testthat::expect_length(theta, 1L)
  testthat::expect_true(all(theta > 0))

  pred <- predict(m, type = "response")
  sims <- m$loss$simulate(pred)
  testthat::expect_equal(length(sims), nrow(pred))
  testthat::expect_true(all(sims >= 0))

  # multi-response
  df2 <- data.frame(X = df$X, Y1 = df$Y, Y2 = rpois(80, 3))
  m2 <- dnn(cbind(Y1, Y2) ~ X, data = df2, loss = "nbinom", epochs = 3L, verbose = FALSE, plot = FALSE)
  testthat::expect_length(m2$loss$parameter_link(), 2L)
  testthat::expect_equal(dim(m2$loss$simulate(predict(m2, type = "response"))), c(80L, 2L))
})


testthat::test_that("loss = 'binomial' accepts a 0/1 vector and matches bernoulli", {
  skip_if_no_torch()

  df <- data.frame(X = rnorm(60), Y = rbinom(60, 1, 0.5))
  torch::torch_manual_seed(1)
  mb <- dnn(Y ~ X, data = df, loss = "binomial", epochs = 2L, verbose = FALSE, plot = FALSE)
  torch::torch_manual_seed(1)
  mr <- dnn(Y ~ X, data = df, loss = "bernoulli", epochs = 2L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(as.vector(predict(mb, type = "response")), as.vector(predict(mr, type = "response")), tolerance = 1e-5)
  testthat::expect_error(dnn(Y ~ X, data = transform(df, Y = 1), loss = "binomial", epochs = 1L, verbose = FALSE, plot = FALSE), NA)
})


testthat::test_that("binomial stays finite on separable 0/1 data", {
  skip_if_no_torch()

  set.seed(1)
  df <- data.frame(x1 = rnorm(300), x2 = rnorm(300))
  df$Y <- as.integer(df$x1 + df$x2 > 0)
  torch::torch_manual_seed(1)
  # log(1 - sigmoid(eta)) underflowed to -Inf at eta >= 17 and the loss became NaN within the first epoch
  m <- dnn(Y ~ ., data = df, loss = "binomial", optimizer = "adam", lr = 0.05, epochs = 50L, verbose = FALSE, plot = FALSE)
  testthat::expect_false(anyNA(m$losses$train_l))
  testthat::expect_false(anyNA(predict(m)))
  eta <- torch::torch_tensor(matrix(c(-40, 0, 17, 40), ncol = 1))
  nll <- m$loss$forward(eta, torch::torch_tensor(cbind(c(1, 1, 1, 0), c(0, 0, 0, 1))))
  testthat::expect_true(all(is.finite(as.numeric(nll))))
})


testthat::test_that("an ordered response stays a factor, ordered predictors become integers", {
  skip_if_no_torch()

  df <- data.frame(X = rnorm(60), Z = factor(sample(c("a", "b", "c"), 60, TRUE), ordered = TRUE),
                   Y = factor(sample(c("low", "mid", "high"), 60, TRUE), levels = c("low", "mid", "high"), ordered = TRUE))
  m <- dnn(Y ~ X + Z, data = df, loss = "cross-entropy", epochs = 1L, verbose = FALSE, plot = FALSE)
  testthat::expect_equal(ncol(predict(m, type = "response")), 3L)
  testthat::expect_true(all(predict(m, type = "class") %in% levels(df$Y)))
})


testthat::test_that("ordered predictors in newdata get the integer codes of the training levels", {
  skip_if_no_torch()

  set.seed(2)
  for(k in 2:3) {
    lv <- letters[1:k]
    df <- data.frame(X = rnorm(80), O = factor(sample(lv, 80, TRUE), levels = lv, ordered = TRUE))
    df$Y <- as.integer(df$O) + df$X
    m <- dnn(Y ~ X + O, data = df, loss = "mse", epochs = 5L, verbose = FALSE, plot = FALSE)
    testthat::expect_equal(as.vector(predict(m, newdata = df)), as.vector(predict(m)), tolerance = 1e-5)
    # a subset whose levels were dropped must keep the training codes
    sub <- droplevels(df[df$O == lv[k], ])
    testthat::expect_equal(as.vector(predict(m, newdata = sub)), as.vector(predict(m))[df$O == lv[k]], tolerance = 1e-5)
    testthat::expect_error(conditionalEffects(m, data = df), NA)
    testthat::expect_error(ALE(m, variable = "X", data = df, plot = FALSE), NA)
  }
})


testthat::test_that("e() levels of newdata are coded on the training levels", {
  skip_if_no_torch()

  set.seed(3)
  df <- data.frame(x = rnorm(100), g = sample(c("a", "b", "c", "d", "e"), 100, TRUE))
  df$y <- df$x + as.integer(factor(df$g))
  m <- dnn(y ~ x + e(g, dim = 2), data = df, loss = "mse", epochs = 5L, verbose = FALSE, plot = FALSE)
  # characters were accepted at fit but rejected by predict()
  testthat::expect_equal(as.vector(predict(m, newdata = df)), as.vector(predict(m)), tolerance = 1e-5)
  i <- which(df$g == "d")
  sub <- df[i, ]
  sub$g <- droplevels(factor(sub$g))
  testthat::expect_equal(as.vector(predict(m, newdata = sub)), as.vector(predict(m))[i], tolerance = 1e-5)
  testthat::expect_equal(as.vector(conditionalEffects(m, data = sub)[[1]]$mean), as.vector(conditionalEffects(m, data = df[i, ])[[1]]$mean), tolerance = 1e-5)
  testthat::expect_error(predict(m, newdata = transform(df[1:3, ], g = "z")), "not seen during training")
  m2 <- continue_training(m, epochs = 1L, data = df[i, ], verbose = FALSE)
  testthat::expect_equal(as.vector(m2$data$Z), as.vector(m$data$Z)[i])
})


testthat::test_that("gaussian: the sd stays positive and is estimated, also when the mean fits fast", {
  skip_if_no_torch()

  set.seed(2)
  d <- data.frame(X1 = rnorm(500), Temp = rnorm(500))
  d$Y <- 3 * d$Temp + rnorm(500, sd = 0.1)
  for(seed in 1:3) {
    torch::torch_manual_seed(seed)
    m <- dnn(Y ~ X1 + Temp, data = d, loss = "gaussian", hidden = NULL, epochs = 40L, lr = 0.05, verbose = FALSE, plot = FALSE)
    testthat::expect_false(anyNA(m$losses$train_l))
    testthat::expect_true(m$loss$parameter_link() > 0 && m$loss$parameter_link() < 0.3)
  }
})
