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
      .n <- wrap_cnn(list(X=X_cnn, Y=y, architecture=architecture, loss=loss, device=device, epochs=1, plot=FALSE, verbose=FALSE))
      .n <- wrap_mmn(list(formula=Y ~ cnn(X=X_cnn, architecture=architecture) + dnn(X=X_dnn), dataList=list(X_cnn=X_cnn, X_dnn=X_dnn, Y=y), loss=loss, device=device, epochs=1, plot=FALSE, verbose=FALSE))
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

