skip_if_no_torch = function() {
  if (!torch::torch_is_installed())
    skip("required torch version not available for testing")
}

wrap_dnn = function(pars) {
  testthat::expect_error({model = do.call(dnn, pars)}, NA)
  testthat::expect_error({.n = predict(model, newdata=pars$data)}, NA)
  testthat::expect_error({.n = continue_training(model, epochs = 1L, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = predict(model, type = "response")}, NA)
  testthat::expect_error({.n = coef(model)}, NA)
  testthat::expect_error({.n = plot(model)}, NA)
  testthat::expect_error({.n = residuals(model)}, NA)
  testthat::expect_error({.n = summary(model)}, NA)
  suppressWarnings(testthat::expect_error({.n = PDP(model, variable = "X.1")}, NA))
  suppressWarnings(testthat::expect_error({.n = ALE(model, variable = "X.1")}, NA))
}

wrap_cnn = function(pars) {
  testthat::expect_error({model = do.call(cnn, pars)}, NA)
  testthat::expect_error({.n = predict(model, newdata=pars$X)}, NA)
  testthat::expect_error({.n = continue_training(model, epochs = 1L, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = predict(model, type = "response")}, NA)
  testthat::expect_error({.n = coef(model)}, NA)
  testthat::expect_error({.n = plot(model)}, NA)
  testthat::expect_error({.n = residuals(model)}, NA)
  testthat::expect_error({.n = summary(model)}, NA)
}

wrap_mmn = function(pars) {
  testthat::expect_error({model = do.call(mmn, pars)}, NA)
  testthat::expect_error({.n = predict(model, newdata=pard$dataList)}, NA)
  testthat::expect_error({.n = continue_training(model, epochs = 1L, verbose = FALSE)}, NA)
  testthat::expect_error({.n = predict(model)}, NA)
  testthat::expect_error({.n = predict(model, type = "response")}, NA)
  testthat::expect_error({.n = coef(model)}, NA)
  testthat::expect_error({.n = residuals(model)}, NA)
  testthat::expect_error({.n = summary(model)}, NA)
}
