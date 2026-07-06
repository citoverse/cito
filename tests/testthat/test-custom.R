source("utils.R")
library(cito)
set.seed(42)

X = matrix(runif(3*50), 50, 3)
Y = runif(50)
data = data.frame(Y = Y, X, Cat = as.factor(rep(1:5, 10)))

# input and output are not hard coded, they are filled in automatically by cito
custom_mlp = torch::nn_module(
  initialize = function(input, output, hidden = 20L) {
    self$l1 = torch::nn_linear(input, hidden)
    self$l2 = torch::nn_linear(hidden, output)
  },
  forward = function(x) {
    return(self$l2(torch::nnf_relu(self$l1(x))))
  }
)

# embeddings receives the named vector of e() levels
custom_mlp_embed = torch::nn_module(
  initialize = function(input, output, embeddings, dim = 3L, hidden = 20L) {
    self$embed = torch::nn_embedding(embeddings[1], dim)
    self$l1 = torch::nn_linear(input + dim, hidden)
    self$l2 = torch::nn_linear(hidden, output)
  },
  forward = function(x, z) {
    e = torch::torch_squeeze(self$embed(z[,1,drop=FALSE]), 2)
    return(self$l2(torch::nnf_relu(self$l1(torch::torch_cat(list(x, e), dim = 2L)))))
  }
)

testthat::test_that("DNN custom module", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # generator
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp, epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(m)}, NA)
  testthat::expect_error({.n = predict(m, newdata = data, type = "response")}, NA)
  testthat::expect_error({.n = coef(m)}, NA)
  testthat::expect_error({.n = residuals(m)}, NA)
  testthat::expect_error({.n = plot(m, plot_type = "arch")}, NA)
  testthat::expect_error({.n = summary(m)}, NA)
  testthat::expect_error({.n = conditionalEffects(m)}, NA)
  suppressWarnings(testthat::expect_error({.n = PDP(m, variable = "X1")}, NA))
  suppressWarnings(testthat::expect_error({.n = ALE(m, variable = "X1")}, NA))
  testthat::expect_error({m = continue_training(m, epochs = 2L)}, NA)

  # list form with args
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = list(module = custom_mlp, args = list(hidden = 10L)), epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(m)}, NA)

  # instantiated module (no automatic argument injection possible, dimensions must be provided)
  net_inst = custom_mlp(input = 3L, output = 1L)
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = net_inst, epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(m)}, NA)

  # regularization is applied generically to custom parameters
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp, lambda = 0.01, alpha = 0.5, epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
})

testthat::test_that("DNN custom module save and reload", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp, epochs = 2L, plot = FALSE, verbose = FALSE)
  p_before = predict(m)
  saveRDS(m, "test_custom_model.RDS")
  m = readRDS("test_custom_model.RDS")
  testthat::expect_error({p_after = predict(m)}, NA)
  testthat::expect_true(all(abs(p_before - p_after) < 1e-6))
  testthat::expect_error({.n = continue_training(m, epochs = 2L)}, NA)
  file.remove("test_custom_model.RDS")

  # instantiated module (torch_serialize path)
  m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp(input = 3L, output = 1L), epochs = 2L, plot = FALSE, verbose = FALSE)
  p_before = predict(m)
  saveRDS(m, "test_custom_model.RDS")
  m = readRDS("test_custom_model.RDS")
  testthat::expect_error({p_after = predict(m)}, NA)
  testthat::expect_true(all(abs(p_before - p_after) < 1e-6))
  file.remove("test_custom_model.RDS")
})

testthat::test_that("DNN custom module with embeddings", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  testthat::expect_error({m = dnn(Y ~ X1+X2+X3+e(Cat), data = data, custom = custom_mlp_embed, epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(m)}, NA)
  testthat::expect_error({.n = predict(m, newdata = data)}, NA)
  testthat::expect_error({.n = summary(m)}, NA)
  testthat::expect_error({m = continue_training(m, epochs = 2L)}, NA)

  # unsupported e() features must fail
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3+e(Cat, lambda = 0.1), data = data, custom = custom_mlp_embed, epochs = 2L, plot = FALSE, verbose = FALSE)})
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3+e(Cat, train = FALSE), data = data, custom = custom_mlp_embed, epochs = 2L, plot = FALSE, verbose = FALSE)})
})

testthat::test_that("DNN custom module errors", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  # wrong output dimension
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = list(module = custom_mlp, args = list(output = 5L)), epochs = 2L, plot = FALSE, verbose = FALSE)})

  # tuning of architecture hyperparameters
  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp, hidden = tune(), epochs = 2L, plot = FALSE, verbose = FALSE)})

  # return_embeddings not supported without a return_embeddings argument in forward
  m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp, epochs = 2L, plot = FALSE, verbose = FALSE)
  testthat::expect_error({.n = predict(m, return_embeddings = TRUE)})

  # softmax in forward must raise a warning, the cross-entropy loss expects logits
  softmax_mlp = torch::nn_module(
    initialize = function(input, output, hidden = 20L) {
      self$fc1 = torch::nn_sequential(torch::nn_linear(input, hidden), torch::nn_relu(), torch::nn_linear(hidden, output))
    },
    forward = function(x) {
      return(torch::nnf_softmax(self$fc1(x), 2L))
    }
  )
  testthat::expect_warning({m = dnn(Species ~ ., data = datasets::iris, loss = "cross-entropy", custom = softmax_mlp, epochs = 2L, plot = FALSE, verbose = FALSE)})

  # without the softmax no warning and proper convergence
  logit_mlp = torch::nn_module(
    initialize = function(input, output, hidden = 20L) {
      self$fc1 = torch::nn_sequential(torch::nn_linear(input, hidden), torch::nn_relu(), torch::nn_linear(hidden, output))
    },
    forward = function(x) {
      return(self$fc1(x))
    }
  )
  testthat::expect_warning({m = dnn(Species ~ ., data = datasets::iris, loss = "cross-entropy", custom = logit_mlp, epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
})

testthat::test_that("DNN custom module bootstrap", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  testthat::expect_error({m = dnn(Y ~ X1+X2+X3, data = data, custom = custom_mlp, epochs = 2L, bootstrap = 2L, plot = FALSE, verbose = FALSE)}, NA)
  # the bootstrap models must use the custom module, not the default MLP
  testthat::expect_true(!is.null(m$models[[1]]$model_properties$custom))
  testthat::expect_error({.n = predict(m)}, NA)
})

testthat::test_that("CNN custom module", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  custom_cnn = torch::nn_module(
    initialize = function(input, output) { # input = c(channels, height, width), filled in automatically
      self$conv1 = torch::nn_conv2d(input[1], 5L, 3L)
      self$flatten = torch::nn_flatten()
      self$l1 = torch::nn_linear(80L, output) # 10x10 -> conv(k=3) -> 8x8 -> maxPool(2) -> 4x4, 5*4*4 = 80
    },
    forward = function(x) {
      x = torch::nnf_relu(self$conv1(x))
      x = torch::nnf_max_pool2d(x, 2L)
      return(self$l1(self$flatten(x)))
    }
  )

  XC = array(runif(100*3*10*10), dim = c(100, 3, 10, 10))
  YC = runif(100)

  testthat::expect_error({m = cnn(X = XC, Y = YC, custom = custom_cnn, loss = "mse", epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(m)}, NA)
  testthat::expect_error({.n = predict(m, newdata = XC)}, NA)
  testthat::expect_error({.n = coef(m)}, NA)
  testthat::expect_error({.n = print(m)}, NA)
  testthat::expect_error({m = continue_training(m, epochs = 2L)}, NA)

  # save and reload
  p_before = predict(m)
  saveRDS(m, "test_custom_cnn.RDS")
  m = readRDS("test_custom_cnn.RDS")
  testthat::expect_error({p_after = predict(m)}, NA)
  testthat::expect_true(all(abs(p_before - p_after) < 1e-6))
  file.remove("test_custom_cnn.RDS")

  # architecture and custom are mutually exclusive
  testthat::expect_error({m = cnn(X = XC, Y = YC, architecture = create_architecture(conv(), linear()), custom = custom_cnn, loss = "mse", epochs = 2L, plot = FALSE, verbose = FALSE)})
})

testthat::test_that("MMN with custom submodule", {
  testthat::skip_on_cran()
  testthat::skip_on_ci()
  skip_if_no_torch()

  custom_sub = torch::nn_module(
    initialize = function(input, output) { # within mmn(): input filled in automatically, output from output_dim
      self$l1 = torch::nn_linear(input, output)
    },
    forward = function(x) {
      return(torch::nnf_relu(self$l1(x)))
    }
  )

  dataList = list(
    df = data.frame(X1 = runif(50), X2 = runif(50), X3 = runif(50)),
    A = array(runif(50*3*10*10), dim = c(50, 3, 10, 10)),
    Y = runif(50)
  )
  architecture = create_architecture(conv(3L), maxPool(), linear(5L))

  # output_dim is required for custom submodules
  testthat::expect_error({m = mmn(Y ~ dnn(~ X1+X2+X3, data = df, custom = custom_sub) + cnn(X = A, architecture = architecture), dataList = dataList, loss = "mse", epochs = 2L, plot = FALSE, verbose = FALSE)})

  testthat::expect_error({m = mmn(Y ~ dnn(~ X1+X2+X3, data = df, custom = list(module = custom_sub, output_dim = 10L)) + cnn(X = A, architecture = architecture), dataList = dataList, loss = "mse", epochs = 2L, plot = FALSE, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(m)}, NA)
  testthat::expect_error({.n = print(m)}, NA)
  testthat::expect_error({m = continue_training(m, epochs = 2L)}, NA)
})
