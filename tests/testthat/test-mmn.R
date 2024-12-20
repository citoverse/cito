source("utils.R")
library(cito)
set.seed(42)

wrap_mmn = function(pars) {
  testthat::expect_error({model = do.call(mmn, pars)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = continue_training(model, epochs = 1L, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = predict(model, type = "response")}, NA)
  testthat::expect_error({.n = coef(model)}, NA)
  testthat::expect_error({.n = residuals(model)}, NA)
  testthat::expect_error({.n = summary(model)}, NA)
}

array_dims <- list(c(100,1,10),
                   c(100,1,10,10),
                   c(100,1,10,10,10),
                   c(100,3,10),
                   c(100,3,10,10),
                   c(100,3,10,10,10))
X_cnn <- lapply(array_dims, function(x) array(runif(prod(x)), dim = x))
X_dnn <- matrix(runif(300), 100, 3)

architecture <- create_architecture(conv(), maxPool(), conv(), avgPool(), linear())

scenarios <- lapply(X_cnn, function(x) {
  list(formula=stats::as.formula("Y~cnn(X=X1, architecture=architecture) + dnn(X=X2)"), dataList=list(X1=x, X2=X_dnn), epochs=1, plot=FALSE, verbose=FALSE)
})

test_loss <- function(loss, Y) {
  for(i in 1:length(scenarios)) {
    sc <- scenarios[[i]]
    sc <- append(sc, list(loss=loss))
    for(j in 1:length(Y)) {
      sc$dataList$Y <- Y[[j]]

      .n <- wrap_mmn(append(sc, list(device="cpu")))
      if(torch::cuda_is_available()) {
        .n <- wrap_mmn(append(sc, list(device="cuda")))
      }
      if(torch::backends_mps_is_available()) {
        .n <- wrap_mmn(append(sc, list(device="mps")))
      }
    }
  }
}




testthat::test_that("MMN softmax", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)))

  test_loss("softmax", Y)
})

testthat::test_that("MMN poisson", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(rpois(100, 10),
            matrix(rpois(100, 10), nrow=100, ncol=1),
            matrix(rpois(300, 10), nrow=100, ncol=3))
  test_loss("poisson", Y)
})

testthat::test_that("MMN nbinom", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(rpois(100, 10),
            matrix(rpois(100, 10), nrow=100, ncol=1),
            matrix(rpois(300, 10), nrow=100, ncol=3))
  test_loss("nbinom", Y)
})

testthat::test_that("MMN binomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)),
            matrix(sample(c(0,1), 300, replace=TRUE), nrow=100, ncol=3),
            matrix(sample(c(FALSE,TRUE), 300, replace=TRUE), nrow=100, ncol=3))
  test_loss("binomial", Y)
})

testthat::test_that("MMN multinomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)),
            #matrix(rpois(100,10),100,1),
            matrix(rpois(300,10),100,3))
  test_loss("multinomial", Y)
})

testthat::test_that("MMN mse/mae/gaussian", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(runif(100),
            matrix(runif(100), nrow=100, ncol=1),
            matrix(runif(300), nrow=100, ncol=3))
  test_loss("mse", Y)
  test_loss("mae", Y)
  test_loss("gaussian", Y)
})
