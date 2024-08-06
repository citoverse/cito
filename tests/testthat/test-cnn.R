source("utils.R")

wrap_cnn = function(pars) {
  device <- ifelse(torch::cuda_is_available(), "cuda", "cpu")
  pars <- append(pars, list(device=device))
  testthat::expect_error({model = do.call(cnn, pars)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = continue_training(model, epochs = 1L, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = predict(model, type = "response")}, NA)
  testthat::expect_error({.n = coef(model)}, NA)
  testthat::expect_error({.n = plot(model)}, NA)
  testthat::expect_error({.n = residuals(model)}, NA)
  testthat::expect_error({.n = summary(model)}, NA)
}


array_dims <- list(c(100,1,10),
                   c(100,1,10,10),
                   c(100,1,10,10,10),
                   c(100,3,10),
                   c(100,3,10,10),
                   c(100,3,10,10,10))
X <- lapply(array_dims, function(x) array(runif(prod(x)), dim = x))

architecture <- create_architecture(conv(), maxPool(), conv(), avgPool(), linear())

testthat::test_that("CNN softmax", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- sample(c("a","b","c"), 100, replace=TRUE)
  Y <- list(#Y,
            #matrix(Y, nrow=100, ncol=1),
            factor(Y))
  for(i in 1:length(X)) {
    for(j in 1:length(Y)) {
      wrap_cnn(list(X=X[[i]], Y=Y[[j]], architecture=architecture, epochs=1, loss="softmax", plot=FALSE, verbose=FALSE))
    }
  }
})

testthat::test_that("CNN poisson", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- rpois(100, 10)
  Y <- list(Y,
            matrix(Y, nrow=100, ncol=1))
  for(i in 1:length(X)) {
    for(j in 1:length(Y)) {
      wrap_cnn(list(X=X[[i]], Y=Y[[j]], architecture=architecture, epochs=1, loss="poisson", plot=FALSE, verbose=FALSE))
    }
  }
})

testthat::test_that("CNN binomial", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(factor(sample(c("a","b","c"), 100, replace=TRUE)),
            matrix(sample(c(0,1), 300, replace=TRUE), nrow=100, ncol=3),
            matrix(sample(c(FALSE,TRUE), 300, replace=TRUE), nrow=100, ncol=3))
  for(i in 1:length(X)) {
    for(j in 1:length(Y)) {
      wrap_cnn(list(X=X[[i]], Y=Y[[j]], architecture=architecture, epochs=1, loss="binomial", plot=FALSE, verbose=FALSE))
    }
  }
})

testthat::test_that("CNN mse/mae/gaussian", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  Y <- list(runif(100),
            matrix(runif(100), nrow=100, ncol=1),
            matrix(runif(300), nrow=100, ncol=3))
  for(i in 1:length(X)) {
    for(j in 1:length(Y)) {
      for (loss in c("mse", "mae", "gaussian")) {
        wrap_cnn(list(X=X[[i]], Y=Y[[j]], architecture=architecture, epochs=1, loss=loss, plot=FALSE, verbose=FALSE))
      }
    }
  }
})

testthat::test_that("CNN accuracy", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  set.seed(123)
  n <- 1000
  shapes <- simulate_shapes(n, 50)
  test <- sample.int(n, 0.1*n)
  train <- c(1:n)[-test]
  cnn.fit <- cnn(X=shapes$data[train,,,,drop=F], Y=shapes$labels[train], architecture=architecture, loss="softmax", validation=0.1, epochs=200, plot=FALSE, verbose=FALSE)
  pred <- predict(cnn.fit, newdata=shapes$data[test,,,,drop=F], type="class")
  true <- shapes$labels[test]
  accuracy <- length(which(pred==true))/length(test)
  testthat::expect_gt(accuracy, 0.80)
})



testthat::test_that("CNN transfer learning", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  shapes <- simulate_shapes(10, 100, 3)

  models <- list(
    "alexnet",
    "inception_v3",
    "mobilenet_v2",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext50_32x4d",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "wide_resnet101_2",
    "wide_resnet50_2"
  )

  for (transfer_model in models) {
    architecture <- create_architecture(transfer(transfer_model, pretrained=FALSE))
    wrap_cnn(list(X=shapes$data, Y=shapes$labels, architecture=architecture, epochs=1, batchsize = 5L, loss="softmax", plot=FALSE, verbose=FALSE))

    architecture <- create_architecture(transfer(transfer_model, pretrained=FALSE), linear(), linear())
    wrap_cnn(list(X=shapes$data, Y=shapes$labels, architecture=architecture, epochs=1, batchsize = 5L, loss="softmax", plot=FALSE, verbose=FALSE))
  }
})

testthat::test_that("CNN pretrained", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  architecture <- create_architecture(transfer("alexnet"))

  n <- 500
  shapes <- simulate_shapes(n, 100, 3)
  test <- sample.int(n, 0.1*n)
  train <- c(1:n)[-test]
  cnn.fit <- cnn(X=shapes$data[train,,,,drop=F], Y=shapes$labels[train], architecture=architecture, loss="softmax", epochs=10)
  pred <- predict(cnn.fit, newdata=shapes$data[test,,,,drop=F], type="class")
  true <- shapes$labels[test]
  accuracy <- length(which(pred==true))/length(test)
  testthat::expect_gt(accuracy, 0.95)
})








